#!/usr/bin/env python3
"""Publish laboratory-like simulated MoCap pose + gyro from PX4 SITL.

PX4 VehicleOdometry is first converted through the trusted project state
adapter and then re-expressed as the raw convention observed in the pool:

    raw MoCap world: NED
    raw rigid body:  FRD

The downstream ``mocap_odom_ekf`` therefore exercises exactly the same
NED/FRD -> core-NWU/FLU conversion in SITL and in the real experiment.

Two pose-delivery modes are supported:

``ideal``
    Publish every valid simulated MoCap pose.

``intermittent``
    Periodically suppress only the MoCap pose stream. The pseudo-IMU continues
    to publish, which permits testing estimator coasting with a fresh gyro.
    Setting ``use_imu_gyro:=false`` in the EKF wrapper tests the same dropout
    without gyro assistance.

This adapter is a software-path validator, not an independent ground-truth
sensor.
"""

from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from px4_msgs.msg import VehicleOdometry
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import Imu

from .state_adapter import px4_vehicle_odometry_to_core_state


def _expand(template: str, robot: str) -> str:
    return (
        template
        .replace("{robot}", robot)
        .replace("{robot_lower}", robot.lower())
    )


def pose_available(
    elapsed_sec: float,
    *,
    mode: str,
    dropout_start_sec: float,
    dropout_period_sec: float,
    dropout_duration_sec: float,
) -> bool:
    """Return whether the simulated MoCap pose is available at this time."""
    if mode == "ideal":
        return True
    if elapsed_sec < dropout_start_sec:
        return True
    phase = (elapsed_sec - dropout_start_sec) % dropout_period_sec
    return phase >= dropout_duration_sec


class SimulatedMocapNode(Node):
    def __init__(self) -> None:
        super().__init__("simulated_mocap")

        self.declare_parameter("robots", ["itrl_rov_1"])
        self.declare_parameter(
            "input_topic_template",
            "/{robot}/fmu/out/vehicle_odometry",
        )
        self.declare_parameter(
            "output_pose_topic_template",
            "/mocap/{robot}/pose",
        )
        self.declare_parameter(
            "output_imu_topic_template",
            "/mocap/{robot}/imu",
        )
        self.declare_parameter("pose_frame_id", "mocap_ned")
        self.declare_parameter(
            "imu_frame_id_template",
            "{robot}/base_link_frd",
        )
        self.declare_parameter("measurement_mode", "ideal")
        self.declare_parameter("dropout_start_sec", 5.0)
        self.declare_parameter("dropout_period_sec", 10.0)
        self.declare_parameter("dropout_duration_sec", 2.0)
        self.declare_parameter("status_period_sec", 2.0)

        self.robots = [
            str(value).strip()
            for value in self.get_parameter("robots").value
            if str(value).strip()
        ]
        if not self.robots:
            raise ValueError("robots must not be empty")
        if len(set(self.robots)) != len(self.robots):
            raise ValueError("robots must be unique")

        self._input_template = str(
            self.get_parameter("input_topic_template").value
        )
        self._pose_template = str(
            self.get_parameter("output_pose_topic_template").value
        )
        self._imu_template = str(
            self.get_parameter("output_imu_topic_template").value
        )
        self._pose_frame = str(
            self.get_parameter("pose_frame_id").value
        )
        self._imu_frame_template = str(
            self.get_parameter("imu_frame_id_template").value
        )

        self._measurement_mode = str(
            self.get_parameter("measurement_mode").value
        ).strip().lower()
        if self._measurement_mode not in ("ideal", "intermittent"):
            raise ValueError(
                "measurement_mode must be 'ideal' or 'intermittent'"
            )

        self._dropout_start_sec = float(
            self.get_parameter("dropout_start_sec").value
        )
        self._dropout_period_sec = float(
            self.get_parameter("dropout_period_sec").value
        )
        self._dropout_duration_sec = float(
            self.get_parameter("dropout_duration_sec").value
        )
        if self._dropout_start_sec < 0.0:
            raise ValueError("dropout_start_sec must be nonnegative")
        if self._measurement_mode == "intermittent":
            if self._dropout_period_sec <= 0.0:
                raise ValueError(
                    "dropout_period_sec must be positive in intermittent mode"
                )
            if not (
                0.0 < self._dropout_duration_sec
                < self._dropout_period_sec
            ):
                raise ValueError(
                    "dropout_duration_sec must lie strictly between zero and "
                    "dropout_period_sec in intermittent mode"
                )

        status_period = float(
            self.get_parameter("status_period_sec").value
        )
        if status_period <= 0.0:
            raise ValueError("status_period_sec must be positive")

        output_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self._pose_publishers = {}
        self._imu_publishers = {}
        self._subscriptions = []

        self._received = {robot: 0 for robot in self.robots}
        self._pose_published = {robot: 0 for robot in self.robots}
        self._pose_suppressed = {robot: 0 for robot in self.robots}
        self._imu_published = {robot: 0 for robot in self.robots}
        self._rejected = {robot: 0 for robot in self.robots}

        self._last_status_received = {
            robot: 0 for robot in self.robots
        }
        self._last_status_pose_published = {
            robot: 0 for robot in self.robots
        }
        self._last_status_pose_suppressed = {
            robot: 0 for robot in self.robots
        }
        self._last_status_imu_published = {
            robot: 0 for robot in self.robots
        }
        self._last_status_rejected = {
            robot: 0 for robot in self.robots
        }

        self._start_sec = self._now_sec()

        lines = []
        for robot in self.robots:
            input_topic = _expand(self._input_template, robot)
            pose_topic = _expand(self._pose_template, robot)
            imu_topic = _expand(self._imu_template, robot)

            self._pose_publishers[robot] = self.create_publisher(
                PoseStamped,
                pose_topic,
                output_qos,
            )
            self._imu_publishers[robot] = self.create_publisher(
                Imu,
                imu_topic,
                output_qos,
            )
            self._subscriptions.append(
                self.create_subscription(
                    VehicleOdometry,
                    input_topic,
                    self._callback(robot),
                    qos_profile_sensor_data,
                )
            )
            lines.append(
                f"  {robot}: {input_topic} -> {pose_topic}, {imu_topic}"
            )

        self.create_timer(status_period, self._status)

        schedule = "continuous"
        if self._measurement_mode == "intermittent":
            schedule = (
                f"first dropout at {self._dropout_start_sec:.1f} s, "
                f"{self._dropout_duration_sec:.1f} s off every "
                f"{self._dropout_period_sec:.1f} s"
            )

        self.get_logger().info(
            "Simulated MoCap configured:\n"
            + "\n".join(lines)
            + "\n  raw pose contract: world=NED, body=FRD"
            + "\n  raw gyro contract: body=FRD"
            + f"\n  pose measurement mode: {self._measurement_mode}"
            + f"\n  pose schedule: {schedule}"
        )

    def _now_sec(self) -> float:
        return 1e-9 * float(self.get_clock().now().nanoseconds)

    def _pose_available_now(self) -> bool:
        return pose_available(
            max(0.0, self._now_sec() - self._start_sec),
            mode=self._measurement_mode,
            dropout_start_sec=self._dropout_start_sec,
            dropout_period_sec=self._dropout_period_sec,
            dropout_duration_sec=self._dropout_duration_sec,
        )

    def _callback(self, robot: str):
        def callback(message: VehicleOdometry) -> None:
            self._received[robot] += 1
            try:
                state = np.asarray(
                    px4_vehicle_odometry_to_core_state(message),
                    dtype=float,
                )
            except ValueError as error:
                self._rejected[robot] += 1
                self.get_logger().error(
                    f"Cannot convert PX4 state for {robot}: {error}",
                    throttle_duration_sec=2.0,
                )
                return

            if state.shape != (13,) or not np.all(np.isfinite(state)):
                self._rejected[robot] += 1
                self.get_logger().error(
                    f"Invalid converted state for {robot}",
                    throttle_duration_sec=2.0,
                )
                return

            stamp = self.get_clock().now().to_msg()

            # Trusted adapter output: core NWU / body FLU.
            # Re-express it as the raw laboratory convention:
            #
            #   p_NED = S p_NWU,
            #   R_NED<-FRD = S R_NWU<-FLU S,
            #   S = diag(1,-1,-1).
            position_ned = np.array(
                [state[0], -state[1], -state[2]],
                dtype=float,
            )
            quaternion_ned_frd = np.array(
                [state[3], state[4], -state[5], -state[6]],
                dtype=float,
            )  # scalar-first [w,x,y,z]

            if self._pose_available_now():
                pose = PoseStamped()
                pose.header.stamp = stamp
                pose.header.frame_id = self._pose_frame
                pose.pose.position.x = float(position_ned[0])
                pose.pose.position.y = float(position_ned[1])
                pose.pose.position.z = float(position_ned[2])
                pose.pose.orientation.w = float(
                    quaternion_ned_frd[0]
                )
                pose.pose.orientation.x = float(
                    quaternion_ned_frd[1]
                )
                pose.pose.orientation.y = float(
                    quaternion_ned_frd[2]
                )
                pose.pose.orientation.z = float(
                    quaternion_ned_frd[3]
                )
                self._pose_publishers[robot].publish(pose)
                self._pose_published[robot] += 1
            else:
                self._pose_suppressed[robot] += 1

            # The pseudo-IMU is intentionally independent of MoCap visibility.
            # This lets intermittent-pose tests retain a fresh gyro, exactly as
            # a robot-mounted IMU would.
            imu = Imu()
            imu.header.stamp = stamp
            imu.header.frame_id = _expand(
                self._imu_frame_template,
                robot,
            )
            imu.orientation_covariance[0] = -1.0
            imu.linear_acceleration_covariance[0] = -1.0
            imu.angular_velocity.x = float(state[10])
            imu.angular_velocity.y = float(-state[11])
            imu.angular_velocity.z = float(-state[12])
            variance = 1e-4
            imu.angular_velocity_covariance = [
                variance, 0.0, 0.0,
                0.0, variance, 0.0,
                0.0, 0.0, variance,
            ]
            self._imu_publishers[robot].publish(imu)
            self._imu_published[robot] += 1

        return callback

    def _status(self) -> None:
        """Warn only for genuine adapter failures, not intentional dropouts."""
        for robot in self.robots:
            received_delta = (
                self._received[robot]
                - self._last_status_received[robot]
            )
            pose_published_delta = (
                self._pose_published[robot]
                - self._last_status_pose_published[robot]
            )
            pose_suppressed_delta = (
                self._pose_suppressed[robot]
                - self._last_status_pose_suppressed[robot]
            )
            imu_published_delta = (
                self._imu_published[robot]
                - self._last_status_imu_published[robot]
            )
            rejected_delta = (
                self._rejected[robot]
                - self._last_status_rejected[robot]
            )

            self._last_status_received[robot] = self._received[robot]
            self._last_status_pose_published[robot] = (
                self._pose_published[robot]
            )
            self._last_status_pose_suppressed[robot] = (
                self._pose_suppressed[robot]
            )
            self._last_status_imu_published[robot] = (
                self._imu_published[robot]
            )
            self._last_status_rejected[robot] = self._rejected[robot]

            if received_delta <= 0:
                self.get_logger().warn(
                    f"No new PX4 VehicleOdometry received for {robot}.",
                    throttle_duration_sec=10.0,
                )
                continue

            accounted_pose = (
                pose_published_delta + pose_suppressed_delta
            )
            if accounted_pose <= 0:
                self.get_logger().warn(
                    f"PX4 state is arriving for {robot}, but simulated "
                    "MoCap pose processing is stalled.",
                    throttle_duration_sec=10.0,
                )

            if imu_published_delta <= 0:
                self.get_logger().warn(
                    f"PX4 state is arriving for {robot}, but the simulated "
                    "IMU stream is stalled.",
                    throttle_duration_sec=10.0,
                )

            if rejected_delta > 0:
                self.get_logger().warn(
                    f"Rejected {rejected_delta} PX4 state sample(s) for "
                    f"{robot} during the latest health interval.",
                    throttle_duration_sec=5.0,
                )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SimulatedMocapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
