"""PX4 ROS 2 wrench interface.

This intentionally preserves the proven ROS 2/PX4 interface from the previous
workspace: ``VehicleThrustSetpoint`` + ``VehicleTorqueSetpoint`` while
``OffboardControlMode.thrust_and_torque`` is active.

The core controller should normally be built in *wrench space* for this
interface.  Its wrench-polytope constraint represents the BlueROV2 actuator
feasibility set, while PX4 performs the final allocation.

The axis scaling below is an interface normalization inherited from the old
workspace; it is NOT the controller's actuator model.  Keep these constants
documented and verify them against the PX4/SmarcSim vehicle configuration
before hardware experiments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from px4_msgs.msg import (
    VehicleControlMode,
    VehicleThrustSetpoint,
    VehicleTorqueSetpoint,
)
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from .frame_conventions import FrameConvention


def px4_qos_profile() -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
    )


@dataclass(frozen=True, slots=True)
class PX4WrenchNormalization:
    """Physical wrench used for normalized PX4 thrust/torque commands."""

    force_max: tuple[float, float, float] = (88.0, 88.0, 137.0)
    torque_max: tuple[float, float, float] = (30.0, 16.5, 21.0)
    thrust_command_limit: float = 0.10
    torque_command_limit: float = 0.10

    def __post_init__(self) -> None:
        force = np.asarray(self.force_max, dtype=float)
        torque = np.asarray(self.torque_max, dtype=float)
        if (
            force.shape != (3,)
            or torque.shape != (3,)
            or np.any(force <= 0.0)
            or np.any(torque <= 0.0)
        ):
            raise ValueError("PX4 wrench normalization scales must be positive.")
        if not 0.0 < self.thrust_command_limit <= 1.0:
            raise ValueError("thrust_command_limit must lie in (0, 1].")
        if not 0.0 < self.torque_command_limit <= 1.0:
            raise ValueError("torque_command_limit must lie in (0, 1].")


class PX4WrenchInterface:
    """Publish core FLU wrench commands to PX4 FRD setpoint topics."""

    def __init__(
        self,
        node: Node,
        *,
        robot_name: str,
        frames: FrameConvention,
        normalization: PX4WrenchNormalization,
    ) -> None:
        self._node = node
        self._frames = frames
        self._normalization = normalization
        self.enabled = False

        namespace = f"/{robot_name}"
        qos = px4_qos_profile()
        self._thrust_pub = node.create_publisher(
            VehicleThrustSetpoint,
            f"{namespace}/fmu/in/vehicle_thrust_setpoint",
            qos,
        )
        self._torque_pub = node.create_publisher(
            VehicleTorqueSetpoint,
            f"{namespace}/fmu/in/vehicle_torque_setpoint",
            qos,
        )
        node.create_subscription(
            VehicleControlMode,
            f"{namespace}/fmu/out/vehicle_control_mode",
            self._control_mode_callback,
            qos,
        )

    def _control_mode_callback(self, message: VehicleControlMode) -> None:
        was_enabled = self.enabled
        self.enabled = bool(message.flag_armed) and bool(
            message.flag_control_offboard_enabled
        )
        if self.enabled and not was_enabled:
            self._node.get_logger().info("PX4 offboard wrench control enabled.")
        elif was_enabled and not self.enabled:
            self._node.get_logger().warn("PX4 offboard wrench control disabled.")
            self.publish_zero()

    def publish_core_wrench(self, wrench_core_flu: np.ndarray) -> None:
        wrench_frd = self._frames.core_wrench_to_px4_frd(
            np.asarray(wrench_core_flu, dtype=float)
        )
        force_scale = np.asarray(
            self._normalization.force_max,
            dtype=float,
        )
        torque_scale = np.asarray(
            self._normalization.torque_max,
            dtype=float,
        )

        thrust = np.clip(
            wrench_frd[:3] / force_scale,
            -self._normalization.thrust_command_limit,
            self._normalization.thrust_command_limit,
        )
        torque = np.clip(
            wrench_frd[3:] / torque_scale,
            -self._normalization.torque_command_limit,
            self._normalization.torque_command_limit,
        )
        self._publish(thrust, torque)

    def publish_zero(self) -> None:
        self._publish(np.zeros(3), np.zeros(3))

    def _publish(
        self,
        thrust_normalized: np.ndarray,
        torque_normalized: np.ndarray,
    ) -> None:
        now_us = int(self._node.get_clock().now().nanoseconds / 1000)

        thrust = VehicleThrustSetpoint()
        thrust.timestamp = now_us
        thrust.timestamp_sample = now_us
        thrust.xyz = [float(value) for value in thrust_normalized]

        torque = VehicleTorqueSetpoint()
        torque.timestamp = now_us
        torque.timestamp_sample = now_us
        torque.xyz = [float(value) for value in torque_normalized]

        self._thrust_pub.publish(thrust)
        self._torque_pub.publish(torque)
