"""ROS 2 follower node for one directed sensing edge."""

from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import Vector3Stamped
from nav_msgs.msg import Odometry
from px4_msgs.msg import VehicleOdometry
from rclpy.node import Node
from std_msgs.msg import String

from formation_control.control.bluerov2_leader import LeaderTrajectorySample

from .core_runtime import (
    FollowerCoreRuntime,
    FollowerTaskConfig,
    LeaderCoreRuntime,
)
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
from .workspace_parameters import (
    declare_workspace_parameters,
    workspace_config_from_parameters,
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
        declare_workspace_parameters(self)

        self.declare_parameter("parent_robot_name", "itrl_rov_1")
        self.declare_parameter("self_odometry_topic", "")
        self.declare_parameter("parent_odometry_topic", "")
        self.declare_parameter("desired_relative_position", [1.8, 0.0, 0.0])
        self.declare_parameter(
            "desired_relative_position_topic",
            "formation_control/desired_relative_position",
        )
        self.declare_parameter(
            "desired_formation_topic",
            "formation_control/desired_formation",
        )
        self.declare_parameter(
            "formation_names",
            ["close", "nominal", "wide", "elevated"],
        )
        self.declare_parameter(
            "formation_relative_positions",
            [
                0.0, -1.25, 0.0,
                0.0, -1.80, 0.0,
                0.0, -2.40, 0.0,
                0.0, -1.80, 0.40,
            ],
        )

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

        # Optional experiment-phase gate.  When a non-empty topic is supplied,
        # the follower first tracks an absolute initialization pose with the
        # same validated leader CLF-QP and only evaluates camera-dependent
        # formation control after the manager publishes FORMATION.
        self.declare_parameter("experiment_phase_topic", "")
        self.declare_parameter(
            "initialization_position",
            [0.0, 0.0, 0.0],
        )
        self.declare_parameter("initialization_position_gain", 1.0)
        self.declare_parameter("initialization_attitude_gain", 1.0)

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

        formation_names = [
            str(value)
            for value in self.get_parameter("formation_names").value
        ]
        flat_formations = np.asarray(
            self.get_parameter("formation_relative_positions").value,
            dtype=float,
        ).reshape(-1)
        if len(set(formation_names)) != len(formation_names):
            raise ValueError("formation_names must be unique.")
        if flat_formations.size != 3 * len(formation_names):
            raise ValueError(
                "formation_relative_positions must contain exactly three "
                "values for every entry in formation_names."
            )
        formation_vectors = flat_formations.reshape(-1, 3)
        self._formation_library = {
            name: formation_vectors[index].copy()
            for index, name in enumerate(formation_names)
        }

        self.frames = frame_convention_from_parameters(self)
        controller_config = controller_config_from_parameters(self)
        workspace_config = workspace_config_from_parameters(self)
        self.runtime = FollowerCoreRuntime(
            controller_config,
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
            workspace_config=workspace_config,
        )

        initialization_position = np.asarray(
            self.get_parameter("initialization_position").value,
            dtype=float,
        ).reshape(-1)
        if initialization_position.shape != (3,):
            raise ValueError(
                "initialization_position must contain three values."
            )
        if not np.all(np.isfinite(initialization_position)):
            raise ValueError(
                "initialization_position must be finite."
            )
        self._initialization_position = initialization_position.copy()
        self._initialization_runtime = LeaderCoreRuntime(
            controller_config,
            workspace_config=workspace_config,
            position_gain=float(
                self.get_parameter("initialization_position_gain").value
            ),
            attitude_gain=float(
                self.get_parameter("initialization_attitude_gain").value
            ),
        )
        self._initialization_reference = LeaderTrajectorySample(
            position=self._initialization_position.copy(),
            velocity=np.zeros(3),
            acceleration=np.zeros(3),
        )

        phase_topic = str(
            self.get_parameter("experiment_phase_topic").value
        ).strip()
        self._experiment_phase = (
            "FORMATION" if not phase_topic else "INITIALIZE"
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

        desired_topic = str(
            self.get_parameter("desired_relative_position_topic").value
        )
        formation_topic = str(
            self.get_parameter("desired_formation_topic").value
        )
        self.create_subscription(
            Vector3Stamped,
            desired_topic,
            self._desired_relative_position_callback,
            10,
        )
        self.create_subscription(
            String,
            formation_topic,
            self._desired_formation_callback,
            10,
        )

        if phase_topic:
            self.create_subscription(
                String,
                phase_topic,
                self._experiment_phase_callback,
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
            f"  online relative-position topic: {desired_topic}\n"
            f"  named-formation topic: {formation_topic}\n"
            f"  formation library: {tuple(self._formation_library)}\n"
            f"  experiment phase: {self._experiment_phase}\n"
            f"  experiment phase topic: {phase_topic or '(disabled)'}\n"
            f"  absolute initialization position: "
            f"{self._initialization_position.tolist()}\n"
            f"  workspace barrier enabled: {workspace_config.enabled}\n"
            f"  workspace adaptive: {workspace_config.adaptive}\n"
            f"  dry run: {self.dry_run}"
        )

    def _set_desired_relative_position(
        self,
        desired_core: np.ndarray,
        *,
        source: str,
    ) -> None:
        try:
            self.runtime.set_desired_relative_position(desired_core)
        except ValueError as error:
            self.get_logger().error(
                f"Rejected desired formation from {source}: {error}"
            )
            return

        desired = self.runtime.desired_relative_position
        self.get_logger().info(
            "Updated desired parent-minus-follower vector from "
            f"{source}: {desired.tolist()}"
        )

    def _desired_relative_position_callback(
        self,
        message: Vector3Stamped,
    ) -> None:
        incoming_world = np.array(
            [
                message.vector.x,
                message.vector.y,
                message.vector.z,
            ],
            dtype=float,
        )
        desired_core = self.frames.world_vector_to_core(incoming_world)
        self._set_desired_relative_position(
            desired_core,
            source="desired_relative_position topic",
        )

    def _desired_formation_callback(
        self,
        message: String,
    ) -> None:
        name = message.data.strip()
        desired = self._formation_library.get(name)
        if desired is None:
            self.get_logger().error(
                f"Unknown formation {name!r}. Available formations: "
                f"{tuple(self._formation_library)}"
            )
            return
        self._set_desired_relative_position(
            desired,
            source=f"formation library entry {name!r}",
        )

    def _experiment_phase_callback(self, message: String) -> None:
        phase = message.data.strip().upper()
        if phase not in ("INITIALIZE", "FORMATION"):
            self.get_logger().error(
                f"Unknown experiment phase {message.data!r}; expected "
                "INITIALIZE or FORMATION."
            )
            return

        if phase == self._experiment_phase:
            return

        if (
            self._experiment_phase == "FORMATION"
            and phase == "INITIALIZE"
        ):
            self.get_logger().error(
                "Ignoring unsupported FORMATION -> INITIALIZE transition. "
                "Restart the experiment to reinitialize cleanly."
            )
            return

        if phase == "FORMATION":
            # Preserve wall-domain relaxation accumulated by the absolute
            # initialization controller while allowing the formation command
            # filter itself to initialize from the switch state.
            self.runtime.set_workspace_relaxation_state(
                self._initialization_runtime.workspace_relaxation_state
            )

        self._experiment_phase = phase
        self.get_logger().info(
            f"Experiment phase changed to {self._experiment_phase}."
        )

    def _initialization_step(self):
        assert self._self_state is not None

        return self._initialization_runtime.step(
            self._self_state,
            self._initialization_reference,
            dt=self.dt,
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
            if self._experiment_phase == "INITIALIZE":
                result = self._initialization_step()
            else:
                # FollowerCoreRuntime remains uninitialized during INITIALIZE,
                # so its command filter and adaptive-domain state are initialized
                # from the actual state at the FORMATION transition.
                result = self.runtime.step(
                    self._self_state,
                    self._parent_state,
                    dt=self.dt,
                )
        except Exception as error:
            phase_label = self._experiment_phase.lower()
            self.get_logger().error(
                f"Follower {phase_label} core evaluation failed: {error}",
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
        if (
            self._experiment_phase == "FORMATION"
            and result.snapshot is not None
        ):
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
