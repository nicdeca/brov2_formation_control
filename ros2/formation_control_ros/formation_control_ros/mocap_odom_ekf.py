#!/usr/bin/env python3
"""MoCap-based state estimator with an explicit core-NWU / FLU contract.

The controller-facing output of this node is ALWAYS

    world: core NWU
    body:  FLU
    quaternion: body-FLU -> core-NWU
    twist: body FLU

The incoming real MoCap PoseStamped is converted at this node's input boundary.
The conversion is configurable so the physical MoCap bridge does not need to
share the controller's axes or origin.

Estimation structure
--------------------
* Translation: constant-velocity Kalman filter in core-NWU coordinates.
* Attitude: direct gated MoCap quaternion update (no slow EKF attitude lag).
* Angular velocity:
    - primary: body-FLU gyro when enabled and fresh;
    - fallback: finite difference of accepted MoCap attitudes, low-pass filtered.
* Linear velocity output: translational-KF world velocity rotated to body FLU.

A transformed raw pose is also published on ``core_pose_topic``.  Recording
this topic makes it possible to compare PX4, the MoCap estimator, and raw MoCap
in exactly the same core frame during real experiments.

Quaternion arrays in this file use ROS ordering [x, y, z, w].
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import Imu
from tf2_ros import TransformBroadcaster


S_FLU_FRD = np.diag([1.0, -1.0, -1.0])
R_CORE_FROM_NED = np.diag([1.0, -1.0, -1.0])
R_CORE_FROM_ROS_ENU = np.array(
    [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=float,
)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("quaternion must have finite nonzero norm")
    q = q / norm
    if q[3] < 0.0:
        q = -q
    return q


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    x, y, z, w = quat_normalize(q)
    return np.array([-x, -y, -z, w], dtype=float)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = np.asarray(q1, dtype=float)
    x2, y2, z2, w2 = np.asarray(q2, dtype=float)
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=float,
    )


def quat_exp(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=float).reshape(3)
    angle = float(np.linalg.norm(rotvec))
    if angle <= 1e-12:
        return quat_normalize(
            np.array(
                [
                    0.5 * rotvec[0],
                    0.5 * rotvec[1],
                    0.5 * rotvec[2],
                    1.0,
                ],
                dtype=float,
            )
        )
    axis = rotvec / angle
    half = 0.5 * angle
    return quat_normalize(
        np.concatenate([axis * math.sin(half), [math.cos(half)]])
    )


def quat_log(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    vector = q[:3]
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm <= 1e-12:
        return 2.0 * vector
    angle = 2.0 * math.atan2(vector_norm, q[3])
    if angle > math.pi:
        angle -= 2.0 * math.pi
    return vector * (angle / vector_norm)


def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = quat_normalize(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


def rotation_matrix_to_quat(rotation: np.ndarray) -> np.ndarray:
    r = np.asarray(rotation, dtype=float).reshape(3, 3)
    trace = float(np.trace(r))
    if trace > 0.0:
        scale = 2.0 * math.sqrt(trace + 1.0)
        q = np.array(
            [
                (r[2, 1] - r[1, 2]) / scale,
                (r[0, 2] - r[2, 0]) / scale,
                (r[1, 0] - r[0, 1]) / scale,
                0.25 * scale,
            ],
            dtype=float,
        )
    else:
        index = int(np.argmax(np.diag(r)))
        if index == 0:
            scale = 2.0 * math.sqrt(
                max(1e-16, 1.0 + r[0, 0] - r[1, 1] - r[2, 2])
            )
            q = np.array(
                [
                    0.25 * scale,
                    (r[0, 1] + r[1, 0]) / scale,
                    (r[0, 2] + r[2, 0]) / scale,
                    (r[2, 1] - r[1, 2]) / scale,
                ],
                dtype=float,
            )
        elif index == 1:
            scale = 2.0 * math.sqrt(
                max(1e-16, 1.0 + r[1, 1] - r[0, 0] - r[2, 2])
            )
            q = np.array(
                [
                    (r[0, 1] + r[1, 0]) / scale,
                    0.25 * scale,
                    (r[1, 2] + r[2, 1]) / scale,
                    (r[0, 2] - r[2, 0]) / scale,
                ],
                dtype=float,
            )
        else:
            scale = 2.0 * math.sqrt(
                max(1e-16, 1.0 + r[2, 2] - r[0, 0] - r[1, 1])
            )
            q = np.array(
                [
                    (r[0, 2] + r[2, 0]) / scale,
                    (r[1, 2] + r[2, 1]) / scale,
                    0.25 * scale,
                    (r[1, 0] - r[0, 1]) / scale,
                ],
                dtype=float,
            )
    return quat_normalize(q)


def attitude_difference_rad(q1: np.ndarray, q2: np.ndarray) -> float:
    q_error = quat_multiply(q1, quat_conjugate(q2))
    return float(np.linalg.norm(quat_log(q_error)))


def body_z_axis_angle_rad(q_core_flu: np.ndarray) -> float:
    rotation = quat_to_rotation_matrix(q_core_flu)
    return float(
        math.acos(float(np.clip(rotation[2, 2], -1.0, 1.0)))
    )


def first_order_lpf(
    previous: np.ndarray | None,
    measurement: np.ndarray,
    *,
    dt: float,
    time_constant: float,
) -> np.ndarray:
    measurement = np.asarray(measurement, dtype=float)
    if previous is None or time_constant <= 0.0 or dt <= 0.0:
        return measurement.copy()
    alpha = float(dt / (time_constant + dt))
    return previous + alpha * (measurement - previous)


@dataclass(frozen=True)
class MocapFrameTransform:
    """Rigid transform from incoming MoCap pose to core-NWU / body-FLU.

    Let M be the incoming MoCap world and Bm the incoming rigid-body frame.

    ``r_core_from_input_world`` maps vectors M -> core-NWU.
    ``t_core_from_input_world`` is the input-world origin expressed in core.
    ``r_input_body_from_flu`` maps FLU vectors -> incoming rigid-body vectors.
    ``body_origin_offset_input_body`` is the vector from the incoming pose
    origin to the controller body origin, expressed in Bm.
    """

    r_core_from_input_world: np.ndarray
    t_core_from_input_world: np.ndarray
    r_input_body_from_flu: np.ndarray
    body_origin_offset_input_body: np.ndarray

    def pose_to_core(
        self,
        position_input: np.ndarray,
        quaternion_input_body_to_world: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        position_input = np.asarray(position_input, dtype=float).reshape(3)
        r_input_world_from_body = quat_to_rotation_matrix(
            quaternion_input_body_to_world
        )

        position_body_origin_input_world = (
            position_input
            + r_input_world_from_body
            @ self.body_origin_offset_input_body
        )
        position_core = (
            self.t_core_from_input_world
            + self.r_core_from_input_world
            @ position_body_origin_input_world
        )
        r_core_from_flu = (
            self.r_core_from_input_world
            @ r_input_world_from_body
            @ self.r_input_body_from_flu
        )
        return position_core, rotation_matrix_to_quat(r_core_from_flu)


class TranslationalCvKalman:
    """Six-state [position, world velocity] constant-velocity Kalman filter."""

    def __init__(
        self,
        *,
        position_std: float,
        linear_accel_std: float,
        initial_velocity_std: float,
        max_position_innovation_m: float,
    ) -> None:
        self.position_std = float(position_std)
        self.linear_accel_std = float(linear_accel_std)
        self.initial_velocity_std = float(initial_velocity_std)
        self.max_position_innovation_m = float(max_position_innovation_m)

        self.state = np.zeros(6, dtype=float)
        self.covariance = np.eye(6, dtype=float)
        self.initialized = False

    @property
    def position(self) -> np.ndarray:
        return self.state[:3]

    @property
    def velocity_world(self) -> np.ndarray:
        return self.state[3:6]

    def initialize(self, position: np.ndarray) -> None:
        self.state[:] = 0.0
        self.state[:3] = np.asarray(position, dtype=float)
        self.covariance = np.diag(
            [
                self.position_std**2,
                self.position_std**2,
                self.position_std**2,
                self.initial_velocity_std**2,
                self.initial_velocity_std**2,
                self.initial_velocity_std**2,
            ]
        )
        self.initialized = True

    def predict(self, dt: float) -> None:
        if not self.initialized:
            return
        dt = max(float(dt), 0.0)
        if dt <= 1e-9:
            return

        f = np.eye(6, dtype=float)
        f[:3, 3:6] = np.eye(3) * dt

        variance = self.linear_accel_std**2
        q = np.zeros((6, 6), dtype=float)
        q[:3, :3] = np.eye(3) * variance * dt**4 / 4.0
        q[:3, 3:6] = np.eye(3) * variance * dt**3 / 2.0
        q[3:6, :3] = q[:3, 3:6]
        q[3:6, 3:6] = np.eye(3) * variance * dt**2

        self.state = f @ self.state
        self.covariance = f @ self.covariance @ f.T + q
        self.covariance = 0.5 * (
            self.covariance + self.covariance.T
        )

    def update(self, position: np.ndarray) -> bool:
        position = np.asarray(position, dtype=float).reshape(3)
        if not self.initialized:
            self.initialize(position)
            return True

        residual = position - self.position
        innovation = float(np.linalg.norm(residual))
        if (
            self.max_position_innovation_m > 0.0
            and innovation > self.max_position_innovation_m
        ):
            return False

        h = np.zeros((3, 6), dtype=float)
        h[:, :3] = np.eye(3)
        r = np.eye(3, dtype=float) * self.position_std**2

        pht = self.covariance @ h.T
        innovation_covariance = h @ pht + r
        gain = pht @ np.linalg.inv(innovation_covariance)

        self.state = self.state + gain @ residual
        identity = np.eye(6, dtype=float)
        ikh = identity - gain @ h
        self.covariance = (
            ikh @ self.covariance @ ikh.T + gain @ r @ gain.T
        )
        self.covariance = 0.5 * (
            self.covariance + self.covariance.T
        )
        return True


class MocapStateEstimatorNode(Node):
    """Produce controller-ready core-NWU / FLU odometry from MoCap + gyro."""

    def __init__(self) -> None:
        super().__init__("mocap_odom_ekf")

        # Topics and output frames.
        self.declare_parameter("pose_topic", "/mocap/robot/pose")
        self.declare_parameter(
            "core_pose_topic",
            "/mocap/robot/pose_core",
        )
        self.declare_parameter("odom_topic", "/mocap/robot/odom_ekf")
        self.declare_parameter("imu_topic", "/robot/mavros/imu/data")
        self.declare_parameter("parent_frame", "core_nwu")
        self.declare_parameter("child_frame", "robot/base_link_ekf")
        self.declare_parameter("publish_rate_hz", 80.0)
        self.declare_parameter("publish_tf", False)

        # Incoming MoCap pose convention.
        self.declare_parameter(
            "input_world_frame",
            "ned",
        )  # ned | core_nwu | ros_enu | custom
        self.declare_parameter(
            "input_body_frame",
            "frd",
        )  # flu | frd | custom
        self.declare_parameter(
            "world_to_core_translation",
            [0.0, 0.0, 0.0],
        )
        self.declare_parameter(
            "world_to_core_quaternion_xyzw",
            [0.0, 0.0, 0.0, 1.0],
        )
        self.declare_parameter(
            "body_flu_to_input_quaternion_xyzw",
            [0.0, 0.0, 0.0, 1.0],
        )
        self.declare_parameter(
            "body_origin_offset_input_body",
            [0.0, 0.0, 0.0],
        )

        # Gyro convention. ROS/MAVROS base_link is normally FLU.
        self.declare_parameter("use_imu_gyro", True)
        self.declare_parameter(
            "imu_body_frame",
            "flu",
        )  # flu | frd | custom
        self.declare_parameter(
            "body_flu_to_imu_quaternion_xyzw",
            [0.0, 0.0, 0.0, 1.0],
        )
        self.declare_parameter("gyro_timeout_sec", 0.20)
        self.declare_parameter("gyro_time_constant_sec", 0.03)
        self.declare_parameter("gyro_std", 0.03)
        self.declare_parameter("max_gyro_abs_rad_s", 5.0)

        # Translational filter and pose gates.
        self.declare_parameter("position_std", 0.01)
        self.declare_parameter("linear_accel_std", 0.7)
        self.declare_parameter("initial_velocity_std", 0.5)
        self.declare_parameter("orientation_std", 0.015)
        self.declare_parameter("orientation_measurement_gain", 1.0)
        self.declare_parameter(
            "mocap_angular_velocity_time_constant_sec",
            0.05,
        )
        self.declare_parameter("max_position_innovation_m", 0.50)
        self.declare_parameter("max_orientation_innovation_rad", 1.20)
        self.declare_parameter("max_body_z_axis_angle_rad", 1.20)
        self.declare_parameter("max_coast_sec", 1.0)
        self.declare_parameter("max_rejected_samples", 200)
        self.declare_parameter("status_period_sec", 2.0)

        self._pose_topic = str(self.get_parameter("pose_topic").value)
        self._core_pose_topic = str(
            self.get_parameter("core_pose_topic").value
        )
        self._odom_topic = str(self.get_parameter("odom_topic").value)
        self._imu_topic = str(self.get_parameter("imu_topic").value)
        self._parent_frame = str(
            self.get_parameter("parent_frame").value
        )
        self._child_frame = str(
            self.get_parameter("child_frame").value
        )
        self._publish_rate_hz = float(
            self.get_parameter("publish_rate_hz").value
        )
        self._publish_tf = bool(
            self.get_parameter("publish_tf").value
        )

        self._use_imu_gyro = bool(
            self.get_parameter("use_imu_gyro").value
        )
        self._gyro_timeout_sec = float(
            self.get_parameter("gyro_timeout_sec").value
        )
        self._gyro_tau = float(
            self.get_parameter("gyro_time_constant_sec").value
        )
        self._gyro_std = float(self.get_parameter("gyro_std").value)
        self._max_gyro_abs = float(
            self.get_parameter("max_gyro_abs_rad_s").value
        )

        self._orientation_std = float(
            self.get_parameter("orientation_std").value
        )
        self._orientation_gain = float(
            self.get_parameter("orientation_measurement_gain").value
        )
        if not 0.0 < self._orientation_gain <= 1.0:
            raise ValueError(
                "orientation_measurement_gain must lie in (0, 1]"
            )

        self._mocap_omega_tau = float(
            self.get_parameter(
                "mocap_angular_velocity_time_constant_sec"
            ).value
        )
        self._max_orientation_innovation = float(
            self.get_parameter(
                "max_orientation_innovation_rad"
            ).value
        )
        self._max_body_z_axis_angle = float(
            self.get_parameter(
                "max_body_z_axis_angle_rad"
            ).value
        )
        self._max_coast_sec = float(
            self.get_parameter("max_coast_sec").value
        )
        self._max_rejected_samples = int(
            self.get_parameter("max_rejected_samples").value
        )
        self._status_period_sec = float(
            self.get_parameter("status_period_sec").value
        )

        if self._publish_rate_hz <= 0.0:
            raise ValueError("publish_rate_hz must be positive")
        if self._max_coast_sec <= 0.0:
            raise ValueError("max_coast_sec must be positive")
        if self._status_period_sec <= 0.0:
            raise ValueError("status_period_sec must be positive")

        self._frame_transform = self._make_frame_transform()
        self._r_flu_from_imu = self._make_imu_transform()

        self._translation = TranslationalCvKalman(
            position_std=float(
                self.get_parameter("position_std").value
            ),
            linear_accel_std=float(
                self.get_parameter("linear_accel_std").value
            ),
            initial_velocity_std=float(
                self.get_parameter("initial_velocity_std").value
            ),
            max_position_innovation_m=float(
                self.get_parameter(
                    "max_position_innovation_m"
                ).value
            ),
        )

        self._orientation_core_flu: np.ndarray | None = None
        self._mocap_omega_body: np.ndarray | None = None
        self._gyro_body_flu: np.ndarray | None = None
        self._last_filter_sec: float | None = None
        self._last_pose_rx_sec: float | None = None
        self._last_gyro_rx_sec: float | None = None
        self._last_mocap_pose_sec: float | None = None
        self._last_mocap_orientation: np.ndarray | None = None
        self._consecutive_pose_rejections = 0

        self._pose_rx = 0
        self._pose_accepted = 0
        self._pose_rejected = 0
        self._gyro_rx = 0
        self._odom_tx = 0
        self._last_status_pose_rx = 0
        self._last_status_rejected = 0
        self._last_status_gyro_rx = 0
        self._last_status_angular_source: str | None = None

        input_qos = qos_profile_sensor_data
        output_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self._odom_pub = self.create_publisher(
            Odometry,
            self._odom_topic,
            output_qos,
        )
        self._core_pose_pub = self.create_publisher(
            PoseStamped,
            self._core_pose_topic,
            output_qos,
        )
        self._pose_subscription = self.create_subscription(
            PoseStamped,
            self._pose_topic,
            self._pose_cb,
            input_qos,
        )
        self._imu_subscription = None
        if self._use_imu_gyro:
            self._imu_subscription = self.create_subscription(
                Imu,
                self._imu_topic,
                self._imu_cb,
                input_qos,
            )

        self._tf_broadcaster = (
            TransformBroadcaster(self) if self._publish_tf else None
        )

        self.create_timer(
            1.0 / self._publish_rate_hz,
            self._tick,
        )
        self.create_timer(
            self._status_period_sec,
            self._status,
        )

        input_world = str(
            self.get_parameter("input_world_frame").value
        )
        input_body = str(
            self.get_parameter("input_body_frame").value
        )
        imu_body = str(
            self.get_parameter("imu_body_frame").value
        )
        self.get_logger().info(
            "mocap_odom_ekf configured:\n"
            f"  pose input: {self._pose_topic}\n"
            f"  transformed raw pose: {self._core_pose_topic}\n"
            f"  odometry output: {self._odom_topic}\n"
            f"  output contract: world=core_nwu, body=FLU\n"
            f"  input MoCap world: {input_world}\n"
            f"  input MoCap body: {input_body}\n"
            f"  gyro enabled: {self._use_imu_gyro}\n"
            f"  gyro topic: {self._imu_topic}\n"
            f"  gyro body frame: {imu_body}\n"
            f"  orientation measurement gain: "
            f"{self._orientation_gain:.3f}\n"
            f"  publish rate: {self._publish_rate_hz:.1f} Hz\n"
            f"  publish TF: {self._publish_tf}"
        )

    def _make_frame_transform(self) -> MocapFrameTransform:
        world_mode = str(
            self.get_parameter("input_world_frame").value
        ).strip().lower()
        if world_mode == "core_nwu":
            r_core_input = np.eye(3, dtype=float)
        elif world_mode == "ned":
            r_core_input = R_CORE_FROM_NED.copy()
        elif world_mode == "ros_enu":
            r_core_input = R_CORE_FROM_ROS_ENU.copy()
        elif world_mode == "custom":
            r_core_input = quat_to_rotation_matrix(
                np.asarray(
                    self.get_parameter(
                        "world_to_core_quaternion_xyzw"
                    ).value,
                    dtype=float,
                )
            )
        else:
            raise ValueError(
                "input_world_frame must be core_nwu, ned, ros_enu, or custom"
            )

        body_mode = str(
            self.get_parameter("input_body_frame").value
        ).strip().lower()
        if body_mode == "flu":
            r_input_body_flu = np.eye(3, dtype=float)
        elif body_mode == "frd":
            r_input_body_flu = S_FLU_FRD.copy()
        elif body_mode == "custom":
            r_input_body_flu = quat_to_rotation_matrix(
                np.asarray(
                    self.get_parameter(
                        "body_flu_to_input_quaternion_xyzw"
                    ).value,
                    dtype=float,
                )
            )
        else:
            raise ValueError(
                "input_body_frame must be flu, frd, or custom"
            )

        translation = np.asarray(
            self.get_parameter(
                "world_to_core_translation"
            ).value,
            dtype=float,
        ).reshape(3)
        offset = np.asarray(
            self.get_parameter(
                "body_origin_offset_input_body"
            ).value,
            dtype=float,
        ).reshape(3)
        if not np.all(np.isfinite(translation)):
            raise ValueError(
                "world_to_core_translation must be finite"
            )
        if not np.all(np.isfinite(offset)):
            raise ValueError(
                "body_origin_offset_input_body must be finite"
            )

        return MocapFrameTransform(
            r_core_from_input_world=r_core_input,
            t_core_from_input_world=translation,
            r_input_body_from_flu=r_input_body_flu,
            body_origin_offset_input_body=offset,
        )

    def _make_imu_transform(self) -> np.ndarray:
        mode = str(
            self.get_parameter("imu_body_frame").value
        ).strip().lower()
        if mode == "flu":
            return np.eye(3, dtype=float)
        if mode == "frd":
            # v_FRD = S v_FLU -> v_FLU = S v_FRD.
            return S_FLU_FRD.copy()
        if mode == "custom":
            r_imu_from_flu = quat_to_rotation_matrix(
                np.asarray(
                    self.get_parameter(
                        "body_flu_to_imu_quaternion_xyzw"
                    ).value,
                    dtype=float,
                )
            )
            return r_imu_from_flu.T
        raise ValueError(
            "imu_body_frame must be flu, frd, or custom"
        )

    def _now_sec(self) -> float:
        return 1e-9 * float(self.get_clock().now().nanoseconds)

    def _active_angular_velocity_body(
        self,
        now: float,
    ) -> tuple[np.ndarray, str]:
        if (
            self._use_imu_gyro
            and self._gyro_body_flu is not None
            and self._last_gyro_rx_sec is not None
            and now - self._last_gyro_rx_sec
            <= self._gyro_timeout_sec
        ):
            return self._gyro_body_flu.copy(), "gyro"
        if self._mocap_omega_body is not None:
            return self._mocap_omega_body.copy(), "mocap_fd"
        return np.zeros(3, dtype=float), "zero"

    def _predict_to(self, now: float) -> None:
        if self._last_filter_sec is None:
            self._last_filter_sec = now
            return

        dt = max(0.0, now - self._last_filter_sec)
        if dt <= 1e-9:
            return

        self._translation.predict(dt)
        if self._orientation_core_flu is not None:
            omega_body, _ = self._active_angular_velocity_body(now)
            self._orientation_core_flu = quat_normalize(
                quat_multiply(
                    self._orientation_core_flu,
                    quat_exp(omega_body * dt),
                )
            )
        self._last_filter_sec = now

    def _transform_pose(
        self,
        msg: PoseStamped,
    ) -> tuple[np.ndarray, np.ndarray]:
        position = np.array(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ],
            dtype=float,
        )
        quaternion = np.array(
            [
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(position)):
            raise ValueError("MoCap position contains non-finite values")
        quaternion = quat_normalize(quaternion)
        return self._frame_transform.pose_to_core(
            position,
            quaternion,
        )

    def _core_pose_message(
        self,
        stamp,
        position: np.ndarray,
        orientation: np.ndarray,
    ) -> PoseStamped:
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self._parent_frame
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])
        msg.pose.orientation.x = float(orientation[0])
        msg.pose.orientation.y = float(orientation[1])
        msg.pose.orientation.z = float(orientation[2])
        msg.pose.orientation.w = float(orientation[3])
        return msg

    def _pose_cb(self, msg: PoseStamped) -> None:
        self._pose_rx += 1
        now = self._now_sec()

        try:
            position, orientation = self._transform_pose(msg)
        except ValueError as error:
            self._pose_rejected += 1
            self.get_logger().error(
                f"Rejected malformed MoCap pose: {error}",
                throttle_duration_sec=1.0,
            )
            return

        # Always publish the transformed raw pose. This is intentionally before
        # estimator gating so offline diagnostics can see rejected measurements.
        self._core_pose_pub.publish(
            self._core_pose_message(
                self.get_clock().now().to_msg(),
                position,
                orientation,
            )
        )

        self._predict_to(now)

        if (
            self._max_body_z_axis_angle > 0.0
            and body_z_axis_angle_rad(orientation)
            > self._max_body_z_axis_angle
        ):
            self._reject_pose(
                "body-z tilt gate",
                position,
                orientation,
            )
            return

        if self._orientation_core_flu is not None:
            orientation_innovation = attitude_difference_rad(
                orientation,
                self._orientation_core_flu,
            )
            if (
                self._max_orientation_innovation > 0.0
                and orientation_innovation
                > self._max_orientation_innovation
            ):
                self._reject_pose(
                    "orientation innovation gate",
                    position,
                    orientation,
                )
                return

        if not self._translation.update(position):
            self._reject_pose(
                "position innovation gate",
                position,
                orientation,
            )
            return

        # Finite-difference MoCap body angular velocity, expressed in the
        # CURRENT FLU body frame. Use a world-frame finite difference first,
        # then rotate into the measured current body.
        if (
            self._last_mocap_orientation is not None
            and self._last_mocap_pose_sec is not None
        ):
            dt_pose = now - self._last_mocap_pose_sec
            if 1e-4 < dt_pose < 0.25:
                q_world_increment = quat_multiply(
                    orientation,
                    quat_conjugate(
                        self._last_mocap_orientation
                    ),
                )
                omega_world = quat_log(q_world_increment) / dt_pose
                rotation_core_flu = quat_to_rotation_matrix(
                    orientation
                )
                omega_body = rotation_core_flu.T @ omega_world
                self._mocap_omega_body = first_order_lpf(
                    self._mocap_omega_body,
                    omega_body,
                    dt=dt_pose,
                    time_constant=self._mocap_omega_tau,
                )

        self._last_mocap_orientation = orientation.copy()
        self._last_mocap_pose_sec = now

        if self._orientation_core_flu is None:
            self._orientation_core_flu = orientation.copy()
        else:
            residual_world = quat_log(
                quat_multiply(
                    orientation,
                    quat_conjugate(
                        self._orientation_core_flu
                    ),
                )
            )
            self._orientation_core_flu = quat_normalize(
                quat_multiply(
                    quat_exp(
                        self._orientation_gain * residual_world
                    ),
                    self._orientation_core_flu,
                )
            )

        self._last_pose_rx_sec = now
        self._pose_accepted += 1
        self._consecutive_pose_rejections = 0

    def _reject_pose(
        self,
        reason: str,
        position: np.ndarray,
        orientation: np.ndarray,
    ) -> None:
        self._pose_rejected += 1
        self._consecutive_pose_rejections += 1
        self.get_logger().warn(
            "Rejected MoCap pose "
            f"({reason}); consecutive="
            f"{self._consecutive_pose_rejections}",
            throttle_duration_sec=1.0,
        )

        if (
            self._max_rejected_samples > 0
            and self._consecutive_pose_rejections
            >= self._max_rejected_samples
        ):
            # Hard re-anchor after a prolonged rejection sequence, but only
            # when the orientation itself satisfies the tilt gate.
            if (
                self._max_body_z_axis_angle <= 0.0
                or body_z_axis_angle_rad(orientation)
                <= self._max_body_z_axis_angle
            ):
                self._translation.initialize(position)
                self._orientation_core_flu = orientation.copy()
                self._last_mocap_orientation = orientation.copy()
                self._mocap_omega_body = None
                self._consecutive_pose_rejections = 0
                self.get_logger().warn(
                    "Re-anchored MoCap estimator after repeated "
                    "pose rejections."
                )

    def _imu_cb(self, msg: Imu) -> None:
        self._gyro_rx += 1
        now = self._now_sec()

        raw = np.array(
            [
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(raw)):
            self.get_logger().warn(
                "Ignoring non-finite IMU angular velocity.",
                throttle_duration_sec=1.0,
            )
            return

        omega_flu = self._r_flu_from_imu @ raw
        magnitude = float(np.linalg.norm(omega_flu))
        if (
            self._max_gyro_abs > 0.0
            and magnitude > self._max_gyro_abs
        ):
            self.get_logger().warn(
                f"Rejecting gyro magnitude {magnitude:.3f} rad/s "
                f"> {self._max_gyro_abs:.3f} rad/s",
                throttle_duration_sec=1.0,
            )
            return

        dt = (
            0.0
            if self._last_gyro_rx_sec is None
            else now - self._last_gyro_rx_sec
        )
        self._gyro_body_flu = first_order_lpf(
            self._gyro_body_flu,
            omega_flu,
            dt=dt,
            time_constant=self._gyro_tau,
        )
        self._last_gyro_rx_sec = now

    def _reset_after_coast(self) -> None:
        self._translation = TranslationalCvKalman(
            position_std=self._translation.position_std,
            linear_accel_std=self._translation.linear_accel_std,
            initial_velocity_std=self._translation.initial_velocity_std,
            max_position_innovation_m=(
                self._translation.max_position_innovation_m
            ),
        )
        self._orientation_core_flu = None
        self._mocap_omega_body = None
        self._last_filter_sec = None
        self._last_pose_rx_sec = None
        self._last_mocap_pose_sec = None
        self._last_mocap_orientation = None
        self._consecutive_pose_rejections = 0

    def _tick(self) -> None:
        if (
            not self._translation.initialized
            or self._orientation_core_flu is None
            or self._last_pose_rx_sec is None
        ):
            return

        now = self._now_sec()
        if now - self._last_pose_rx_sec > self._max_coast_sec:
            self._reset_after_coast()
            self.get_logger().warn(
                "MoCap coast timeout exceeded; estimator will "
                "reinitialize on the next accepted pose."
            )
            return

        self._predict_to(now)
        stamp = self.get_clock().now().to_msg()
        odom = self._odom_message(stamp, now)
        self._odom_pub.publish(odom)
        self._odom_tx += 1
        if self._tf_broadcaster is not None:
            self._tf_broadcaster.sendTransform(
                self._transform_message(stamp)
            )

    def _odom_message(self, stamp, now: float) -> Odometry:
        assert self._orientation_core_flu is not None

        rotation = quat_to_rotation_matrix(
            self._orientation_core_flu
        )
        linear_body = (
            rotation.T @ self._translation.velocity_world
        )
        angular_body, angular_source = (
            self._active_angular_velocity_body(now)
        )

        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = self._parent_frame
        msg.child_frame_id = self._child_frame

        msg.pose.pose.position.x = float(
            self._translation.position[0]
        )
        msg.pose.pose.position.y = float(
            self._translation.position[1]
        )
        msg.pose.pose.position.z = float(
            self._translation.position[2]
        )
        msg.pose.pose.orientation.x = float(
            self._orientation_core_flu[0]
        )
        msg.pose.pose.orientation.y = float(
            self._orientation_core_flu[1]
        )
        msg.pose.pose.orientation.z = float(
            self._orientation_core_flu[2]
        )
        msg.pose.pose.orientation.w = float(
            self._orientation_core_flu[3]
        )

        msg.twist.twist.linear.x = float(linear_body[0])
        msg.twist.twist.linear.y = float(linear_body[1])
        msg.twist.twist.linear.z = float(linear_body[2])
        msg.twist.twist.angular.x = float(angular_body[0])
        msg.twist.twist.angular.y = float(angular_body[1])
        msg.twist.twist.angular.z = float(angular_body[2])

        pose_cov = np.zeros((6, 6), dtype=float)
        pose_cov[:3, :3] = self._translation.covariance[:3, :3]
        pose_cov[3:6, 3:6] = (
            np.eye(3) * self._orientation_std**2
        )
        msg.pose.covariance = pose_cov.reshape(-1).tolist()

        twist_cov = np.zeros((6, 6), dtype=float)
        twist_cov[:3, :3] = (
            rotation.T
            @ self._translation.covariance[3:6, 3:6]
            @ rotation
        )
        angular_std = (
            self._gyro_std
            if angular_source == "gyro"
            else max(3.0 * self._gyro_std, 0.08)
        )
        twist_cov[3:6, 3:6] = (
            np.eye(3) * angular_std**2
        )
        msg.twist.covariance = twist_cov.reshape(-1).tolist()
        return msg

    def _transform_message(self, stamp) -> TransformStamped:
        assert self._orientation_core_flu is not None
        msg = TransformStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self._parent_frame
        msg.child_frame_id = self._child_frame
        msg.transform.translation.x = float(
            self._translation.position[0]
        )
        msg.transform.translation.y = float(
            self._translation.position[1]
        )
        msg.transform.translation.z = float(
            self._translation.position[2]
        )
        msg.transform.rotation.x = float(
            self._orientation_core_flu[0]
        )
        msg.transform.rotation.y = float(
            self._orientation_core_flu[1]
        )
        msg.transform.rotation.z = float(
            self._orientation_core_flu[2]
        )
        msg.transform.rotation.w = float(
            self._orientation_core_flu[3]
        )
        return msg

    def _status(self) -> None:
        """Report estimator health only when something is wrong.

        The timer is intentionally retained so missing/stale streams are still
        detected, but healthy operation does not generate periodic INFO spam.
        """
        now = self._now_sec()
        angular_source = self._active_angular_velocity_body(now)[1]
        initialized = (
            self._translation.initialized
            and self._orientation_core_flu is not None
        )

        pose_delta = self._pose_rx - self._last_status_pose_rx
        rejected_delta = (
            self._pose_rejected - self._last_status_rejected
        )
        gyro_delta = self._gyro_rx - self._last_status_gyro_rx

        self._last_status_pose_rx = self._pose_rx
        self._last_status_rejected = self._pose_rejected
        self._last_status_gyro_rx = self._gyro_rx

        if pose_delta <= 0:
            self.get_logger().warn(
                f"No new MoCap poses on {self._pose_topic}.",
                throttle_duration_sec=10.0,
            )
            return

        if not initialized:
            self.get_logger().warn(
                "MoCap poses are arriving, but the estimator is not "
                "initialized.",
                throttle_duration_sec=10.0,
            )
            return

        if rejected_delta > 0:
            self.get_logger().warn(
                f"Rejected {rejected_delta} MoCap pose sample(s) during "
                "the latest health interval.",
                throttle_duration_sec=5.0,
            )

        if self._use_imu_gyro:
            if gyro_delta <= 0:
                self.get_logger().warn(
                    f"No new gyro samples on {self._imu_topic}; "
                    f"angular source is {angular_source}.",
                    throttle_duration_sec=10.0,
                )
            elif angular_source != "gyro":
                self.get_logger().warn(
                    "Gyro is enabled but not currently used; "
                    f"angular source is {angular_source}.",
                    throttle_duration_sec=10.0,
                )

        # Log source changes only when they indicate a degradation.
        if (
            self._last_status_angular_source == "gyro"
            and angular_source != "gyro"
        ):
            self.get_logger().warn(
                f"Angular-velocity source changed from gyro to "
                f"{angular_source}.",
                throttle_duration_sec=5.0,
            )
        self._last_status_angular_source = angular_source


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MocapStateEstimatorNode()
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
