"""ROS 2 follower node for one directed sensing edge."""

from __future__ import annotations

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from px4_msgs.msg import VehicleOdometry
from rclpy.node import Node

from .core_runtime import FollowerCoreRuntime, FollowerTaskConfig
from .diagnostics import DiagnosticsPublisher
from .node_helpers import (
    controller_config_from_parameters,
    declare_common_parameters,
    frame_convention_from_parameters,
    px4_normalization_from_parameters,
)
from .px4_interface import PX4WrenchInterface, px4_qos_profile
from .snapshot_publisher import DiagnosticSnapshotPublisher
from .state_adapter import (
    odometry_to_core_state,
    px4_vehicle_odometry_to_core_state,
)


class FollowerControllerNode(Node):
    """Thin ROS 2 wrapper around ``FollowerCoreRuntime``.

    ROS-1 milestone: the relative observation is synthesized from two MoCap
    Odometry topics.  The controller itself only receives converted core
    states.  Later, the parent MoCap subscription can be replaced by an
    onboard relative-pose source without changing ``FollowerCoreRuntime``.
    """

    def __init__(self) -> None:
        super().__init__("formation_follower_controller")
        declare_common_parameters(self)

        self.declare_parameter("parent_robot_name", "itrl_rov_1")
        self.declare_parameter("self_odometry_topic", "")
        self.declare_parameter("parent_odometry_topic", "")
        self.declare_parameter("desired_relative_position", [1.8, 0.0, 0.0])

        self.declare_parameter("d_min", 0.5)
        self.declare_parameter("d_max", 3.6)
        self.declare_parameter("d_min_conservative", 0.8)
        self.declare_parameter("d_max_conservative", 3.0)
        self.declare_parameter("alpha_h_conservative", 0.72)
        self.declare_parameter("alpha_v_conservative", 0.72)
        self.declare_parameter("horizontal_half_angle_deg", 45.0)
        self.declare_parameter("vertical_half_angle_deg", 30.0)

        self.declare_parameter("formation_gain", 1.0)
        self.declare_parameter("image_horizontal_gain", 0.7)
        self.declare_parameter("image_vertical_gain", 0.7)
        self.declare_parameter("collision_barrier_weight", 0.18)
        self.declare_parameter("range_barrier_weight", 0.18)
        self.declare_parameter("horizontal_fov_barrier_weight", 0.25)
        self.declare_parameter("vertical_fov_barrier_weight", 0.25)

        self.declare_parameter("adaptive", True)
        self.declare_parameter("relaxation_recovery_gain", 0.8)
        self.declare_parameter("relaxation_domain_margin_ratio", 0.1)
        self.declare_parameter("use_parent_velocity_in_clf", False)

        self.robot_name = str(self.get_parameter("robot_name").value)
        self.parent_robot_name = str(
            self.get_parameter("parent_robot_name").value
        )
        self.state_source = str(
            self.get_parameter("state_source").value
        )
        if self.state_source not in ("px4", "nav_msgs"):
            raise ValueError(
                "state_source must be 'px4' or 'nav_msgs'."
            )

        self.self_odometry_topic = str(
            self.get_parameter("self_odometry_topic").value
        )
        parent_odometry_topic = str(
            self.get_parameter("parent_odometry_topic").value
        )
        if not self.self_odometry_topic:
            self.self_odometry_topic = (
                f"/{self.robot_name}/fmu/out/vehicle_odometry"
                if self.state_source == "px4"
                else f"/mocap/{self.robot_name.lower()}/odom"
            )
        if not parent_odometry_topic:
            parent_odometry_topic = (
                f"/{self.parent_robot_name}/fmu/out/vehicle_odometry"
                if self.state_source == "px4"
                else f"/mocap/{self.parent_robot_name.lower()}/odom"
            )
        self.parent_odometry_topic = parent_odometry_topic

        self.dt = float(self.get_parameter("dt").value)
        self.measurement_timeout = float(
            self.get_parameter("measurement_timeout").value
        )
        self.dry_run = bool(self.get_parameter("dry_run").value)
        if self.dt <= 0.0:
            raise ValueError("dt must be positive.")

        desired = tuple(
            float(value)
            for value in self.get_parameter(
                "desired_relative_position"
            ).value
        )
        if len(desired) != 3:
            raise ValueError(
                "desired_relative_position must contain three values."
            )

        self.frames = frame_convention_from_parameters(self)
        self.runtime = FollowerCoreRuntime(
            controller_config_from_parameters(self),
            FollowerTaskConfig(
                desired_relative_position=desired,
                d_min=float(self.get_parameter("d_min").value),
                d_max=float(self.get_parameter("d_max").value),
                d_min_conservative=float(
                    self.get_parameter("d_min_conservative").value
                ),
                d_max_conservative=float(
                    self.get_parameter("d_max_conservative").value
                ),
                alpha_h_conservative=float(
                    self.get_parameter("alpha_h_conservative").value
                ),
                alpha_v_conservative=float(
                    self.get_parameter("alpha_v_conservative").value
                ),
                horizontal_half_angle_deg=float(
                    self.get_parameter("horizontal_half_angle_deg").value
                ),
                vertical_half_angle_deg=float(
                    self.get_parameter("vertical_half_angle_deg").value
                ),
                formation_gain=float(
                    self.get_parameter("formation_gain").value
                ),
                image_horizontal_gain=float(
                    self.get_parameter("image_horizontal_gain").value
                ),
                image_vertical_gain=float(
                    self.get_parameter("image_vertical_gain").value
                ),
                collision_barrier_weight=float(
                    self.get_parameter("collision_barrier_weight").value
                ),
                range_barrier_weight=float(
                    self.get_parameter("range_barrier_weight").value
                ),
                horizontal_fov_barrier_weight=float(
                    self.get_parameter(
                        "horizontal_fov_barrier_weight"
                    ).value
                ),
                vertical_fov_barrier_weight=float(
                    self.get_parameter(
                        "vertical_fov_barrier_weight"
                    ).value
                ),
                adaptive=bool(self.get_parameter("adaptive").value),
                relaxation_recovery_gain=float(
                    self.get_parameter(
                        "relaxation_recovery_gain"
                    ).value
                ),
                relaxation_domain_margin_ratio=float(
                    self.get_parameter(
                        "relaxation_domain_margin_ratio"
                    ).value
                ),
                use_parent_velocity_in_clf=bool(
                    self.get_parameter(
                        "use_parent_velocity_in_clf"
                    ).value
                ),
            ),
        )

        self.px4 = PX4WrenchInterface(
            self,
            robot_name=self.robot_name,
            frames=self.frames,
            normalization=px4_normalization_from_parameters(self),
        )
        self.diagnostics = DiagnosticsPublisher(self)
        self.snapshot_publisher = DiagnosticSnapshotPublisher(self)

        self._self_state: np.ndarray | None = None
        self._parent_state: np.ndarray | None = None
        self._self_receipt = -np.inf
        self._parent_receipt = -np.inf

        if self.state_source == "px4":
            self.create_subscription(
                VehicleOdometry,
                self.self_odometry_topic,
                self._self_px4_odom_callback,
                px4_qos_profile(),
            )
            self.create_subscription(
                VehicleOdometry,
                self.parent_odometry_topic,
                self._parent_px4_odom_callback,
                px4_qos_profile(),
            )
        else:
            self.create_subscription(
                Odometry,
                self.self_odometry_topic,
                self._self_odom_callback,
                10,
            )
            self.create_subscription(
                Odometry,
                self.parent_odometry_topic,
                self._parent_odom_callback,
                10,
            )
        self.create_timer(self.dt, self._control_step)

        self.get_logger().info(
            "ROS 2 follower controller configured:\n"
            f"  robot: {self.robot_name}\n"
            f"  parent: {self.parent_robot_name}\n"
            f"  desired parent-minus-follower vector: {desired}\n"
            f"  state source: {self.state_source}\n"
            f"  self odometry: {self.self_odometry_topic}\n"
            f"  parent odometry: {self.parent_odometry_topic}\n"
            f"  generic-odom world frame: {self.frames.world_frame}\n"
            f"  odom twist frame: {self.frames.twist_frame}\n"
            f"  core control space: "
            f"{self.runtime.controller_config.control_space}\n"
            f"  dry run: {self.dry_run}"
        )

    def _now_seconds(self) -> float:
        return 1e-9 * float(self.get_clock().now().nanoseconds)

    def _self_px4_odom_callback(
        self,
        message: VehicleOdometry,
    ) -> None:
        try:
            self._self_state = px4_vehicle_odometry_to_core_state(
                message
            )
            self._self_receipt = self._now_seconds()
        except ValueError as error:
            self.get_logger().error(
                f"Invalid self VehicleOdometry frame: {error}",
                throttle_duration_sec=2.0,
            )

    def _parent_px4_odom_callback(
        self,
        message: VehicleOdometry,
    ) -> None:
        try:
            self._parent_state = px4_vehicle_odometry_to_core_state(
                message
            )
            self._parent_receipt = self._now_seconds()
        except ValueError as error:
            self.get_logger().error(
                f"Invalid parent VehicleOdometry frame: {error}",
                throttle_duration_sec=2.0,
            )

    def _self_odom_callback(self, message: Odometry) -> None:
        self._self_state = odometry_to_core_state(message, self.frames)
        self._self_receipt = self._now_seconds()

    def _parent_odom_callback(self, message: Odometry) -> None:
        self._parent_state = odometry_to_core_state(message, self.frames)
        self._parent_receipt = self._now_seconds()

    def _measurements_fresh(self) -> bool:
        now = self._now_seconds()
        return (
            self._self_state is not None
            and self._parent_state is not None
            and now - self._self_receipt <= self.measurement_timeout
            and now - self._parent_receipt <= self.measurement_timeout
        )

    def _control_step(self) -> None:
        if not self.px4.enabled and not self.dry_run:
            return

        if not self._measurements_fresh():
            self.get_logger().warn(
                "Waiting for fresh self/parent odometry.",
                throttle_duration_sec=2.0,
            )
            if not self.dry_run:
                self.px4.publish_zero()
            self.diagnostics.publish_fallback()
            self.snapshot_publisher.publish_fallback(role=0.0)
            return

        assert self._self_state is not None
        assert self._parent_state is not None

        try:
            result = self.runtime.step(
                self._self_state,
                self._parent_state,
                dt=self.dt,
            )
        except Exception as error:
            self.get_logger().error(
                f"Follower core evaluation failed: {error}",
                throttle_duration_sec=1.0,
            )
            if not self.dry_run:
                self.px4.publish_zero()
            self.diagnostics.publish_fallback()
            self.snapshot_publisher.publish_fallback(role=0.0)
            return

        if not self.dry_run:
            self.px4.publish_core_wrench(result.wrench_body)
        self.diagnostics.publish(
            result.diagnostics,
            fallback=False,
        )
        if result.snapshot is not None:
            self.snapshot_publisher.publish(result.snapshot)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FollowerControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.px4.publish_zero()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
