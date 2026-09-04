#!/usr/bin/env python3
"""Publish simulated MoCap pose + gyro from PX4 SITL odometry.

For each robot:

    PX4 VehicleOdometry (NED/FRD)
        -> formation_control_ros state adapter
        -> PoseStamped in core NWU / FLU
        -> Imu angular velocity in body FLU

Default outputs:

    /mocap/<robot>/pose
    /mocap/<robot>/imu

This is a software-path validator, not an independent ground-truth sensor.
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
        self.declare_parameter("pose_frame_id", "core_nwu")
        self.declare_parameter(
            "imu_frame_id_template",
            "{robot}/base_link",
        )
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
            self.get_parameter(
                "output_pose_topic_template"
            ).value
        )
        self._imu_template = str(
            self.get_parameter(
                "output_imu_topic_template"
            ).value
        )
        self._pose_frame = str(
            self.get_parameter("pose_frame_id").value
        )
        self._imu_frame_template = str(
            self.get_parameter(
                "imu_frame_id_template"
            ).value
        )
        status_period = float(
            self.get_parameter("status_period_sec").value
        )

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
        self._published = {robot: 0 for robot in self.robots}
        self._rejected = {robot: 0 for robot in self.robots}
        self._last_status_received = {
            robot: 0 for robot in self.robots
        }
        self._last_status_published = {
            robot: 0 for robot in self.robots
        }
        self._last_status_rejected = {
            robot: 0 for robot in self.robots
        }

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
                f"  {robot}: {input_topic} -> "
                f"{pose_topic}, {imu_topic}"
            )

        self.create_timer(status_period, self._status)
        self.get_logger().info(
            "Simulated MoCap configured:\n"
            + "\n".join(lines)
            + "\n  output pose contract: core NWU / FLU"
            + "\n  output gyro contract: body FLU"
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

            if state.shape != (13,) or not np.all(
                np.isfinite(state)
            ):
                self._rejected[robot] += 1
                self.get_logger().error(
                    f"Invalid converted state for {robot}",
                    throttle_duration_sec=2.0,
                )
                return

            stamp = self.get_clock().now().to_msg()

            pose = PoseStamped()
            pose.header.stamp = stamp
            pose.header.frame_id = self._pose_frame
            pose.pose.position.x = float(state[0])
            pose.pose.position.y = float(state[1])
            pose.pose.position.z = float(state[2])
            # state adapter quaternion is scalar-first [w,x,y,z].
            pose.pose.orientation.w = float(state[3])
            pose.pose.orientation.x = float(state[4])
            pose.pose.orientation.y = float(state[5])
            pose.pose.orientation.z = float(state[6])
            self._pose_publishers[robot].publish(pose)

            imu = Imu()
            imu.header.stamp = stamp
            imu.header.frame_id = _expand(
                self._imu_frame_template,
                robot,
            )
            imu.orientation_covariance[0] = -1.0
            imu.linear_acceleration_covariance[0] = -1.0
            imu.angular_velocity.x = float(state[10])
            imu.angular_velocity.y = float(state[11])
            imu.angular_velocity.z = float(state[12])
            # SITL pseudo-gyro: small nominal covariance; this field is only
            # diagnostic because estimator tuning is parameterized separately.
            variance = 1e-4
            imu.angular_velocity_covariance = [
                variance, 0.0, 0.0,
                0.0, variance, 0.0,
                0.0, 0.0, variance,
            ]
            self._imu_publishers[robot].publish(imu)
            self._published[robot] += 1

        return callback

    def _status(self) -> None:
        """Report only broken/stalled simulated sensor streams."""
        for robot in self.robots:
            received_delta = (
                self._received[robot]
                - self._last_status_received[robot]
            )
            published_delta = (
                self._published[robot]
                - self._last_status_published[robot]
            )
            rejected_delta = (
                self._rejected[robot]
                - self._last_status_rejected[robot]
            )

            self._last_status_received[robot] = self._received[robot]
            self._last_status_published[robot] = self._published[robot]
            self._last_status_rejected[robot] = self._rejected[robot]

            if received_delta <= 0:
                self.get_logger().warn(
                    f"No new PX4 VehicleOdometry received for {robot}.",
                    throttle_duration_sec=10.0,
                )
                continue

            if published_delta <= 0:
                self.get_logger().warn(
                    f"PX4 state is arriving for {robot}, but simulated "
                    "MoCap pose/gyro is not being published.",
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
