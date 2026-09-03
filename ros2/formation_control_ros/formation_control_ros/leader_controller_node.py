"""ROS 2 leader node using the same CLF-QP dynamics controller."""

from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
from px4_msgs.msg import VehicleOdometry
from rclpy.node import Node
from std_msgs.msg import String
from trajectory_msgs.msg import MultiDOFJointTrajectoryPoint

from formation_control.control.bluerov2_leader import LeaderTrajectorySample
from formation_control.geometry import rotation_matrix_from_quaternion

from .core_runtime import LeaderCoreRuntime
from .diagnostics import DiagnosticsPublisher
from .node_helpers import (
    controller_config_from_parameters,
    declare_common_parameters,
    frame_convention_from_parameters,
    px4_normalization_from_parameters,
)
from .px4_interface import PX4WrenchInterface, px4_qos_profile
from .reference_runtime import VelocityCommandReference
from .snapshot_publisher import DiagnosticSnapshotPublisher
from .state_adapter import (
    odometry_to_core_state,
    px4_vehicle_odometry_to_core_state,
)
from .workspace_parameters import (
    declare_workspace_parameters,
    workspace_config_from_parameters,
)


class LeaderControllerNode(Node):
    """Leader accepting stationary, velocity, or full p/v/a references."""

    def __init__(self) -> None:
        super().__init__("formation_leader_controller")
        declare_common_parameters(self)
        declare_workspace_parameters(self)

        self.declare_parameter("reference_mode", "stationary")
        self.declare_parameter("odometry_topic", "")
        self.declare_parameter(
            "velocity_command_topic",
            "formation_control/velocity_command",
        )
        self.declare_parameter(
            "cmd_vel_topic",
            "formation_control/cmd_vel",
        )
        self.declare_parameter("velocity_command_timeout", 0.35)
        self.declare_parameter(
            "trajectory_point_topic",
            "formation_control/trajectory_point",
        )
        self.declare_parameter("velocity_command_bandwidth", 0.2)
        self.declare_parameter("position_gain", 1.0)
        self.declare_parameter("attitude_gain", 1.0)
        self.declare_parameter("experiment_phase_topic", "")
        self.declare_parameter(
            "initialization_position",
            [0.0, 0.0, 0.0],
        )

        self.robot_name = str(self.get_parameter("robot_name").value)
        self.state_source = str(
            self.get_parameter("state_source").value
        )
        if self.state_source not in ("px4", "nav_msgs"):
            raise ValueError(
                "state_source must be 'px4' or 'nav_msgs'."
            )
        self.odometry_topic = str(
            self.get_parameter("odometry_topic").value
        )
        if not self.odometry_topic:
            self.odometry_topic = (
                f"/{self.robot_name}/fmu/out/vehicle_odometry"
                if self.state_source == "px4"
                # else f"/mocap/{self.robot_name.lower()}/odom"
                else f"/{self.robot_name.lower()}/odom_ekf"
            )

        self.dt = float(self.get_parameter("dt").value)
        self.measurement_timeout = float(
            self.get_parameter("measurement_timeout").value
        )
        self.dry_run = bool(self.get_parameter("dry_run").value)
        self.reference_mode = str(
            self.get_parameter("reference_mode").value
        )
        if self.reference_mode not in (
            "stationary",
            "velocity",
            "trajectory",
        ):
            raise ValueError(
                "reference_mode must be stationary, velocity, or trajectory."
            )

        self.frames = frame_convention_from_parameters(self)
        workspace_config = workspace_config_from_parameters(self)
        self.runtime = LeaderCoreRuntime(
            controller_config_from_parameters(self),
            workspace_config=workspace_config,
            position_gain=float(
                self.get_parameter("position_gain").value
            ),
            attitude_gain=float(
                self.get_parameter("attitude_gain").value
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

        self._state: np.ndarray | None = None
        self._state_receipt = -np.inf
        self._stationary_reference: LeaderTrajectorySample | None = None

        self._velocity_command = np.zeros(3)
        self._velocity_command_receipt = -np.inf
        self._velocity_reference: VelocityCommandReference | None = None

        self._trajectory_reference: LeaderTrajectorySample | None = None
        self._trajectory_receipt = -np.inf

        phase_topic = str(
            self.get_parameter("experiment_phase_topic").value
        ).strip()
        self._experiment_phase = (
            "FORMATION" if not phase_topic else "INITIALIZE"
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
        self._initialization_reference = LeaderTrajectorySample(
            position=initialization_position.copy(),
            velocity=np.zeros(3),
            acceleration=np.zeros(3),
        )

        if self.state_source == "px4":
            self.create_subscription(
                VehicleOdometry,
                self.odometry_topic,
                self._px4_odom_callback,
                px4_qos_profile(),
            )
        else:
            self.create_subscription(
                Odometry,
                self.odometry_topic,
                self._odom_callback,
                10,
            )

        if self.reference_mode == "velocity":
            topic = str(
                self.get_parameter("velocity_command_topic").value
            )
            cmd_vel_topic = str(
                self.get_parameter("cmd_vel_topic").value
            )
            self.create_subscription(
                TwistStamped,
                topic,
                self._velocity_command_callback,
                10,
            )
            self.create_subscription(
                Twist,
                cmd_vel_topic,
                self._cmd_vel_callback,
                10,
            )
        elif self.reference_mode == "trajectory":
            topic = str(
                self.get_parameter("trajectory_point_topic").value
            )
            self.create_subscription(
                MultiDOFJointTrajectoryPoint,
                topic,
                self._trajectory_point_callback,
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
            "ROS 2 leader controller configured:\n"
            f"  robot: {self.robot_name}\n"
            f"  reference mode: {self.reference_mode}\n"
            f"  state source: {self.state_source}\n"
            f"  odometry topic: {self.odometry_topic}\n"
            f"  generic-odom world frame: {self.frames.world_frame}\n"
            f"  odom twist frame: {self.frames.twist_frame}\n"
            f"  experiment phase: {self._experiment_phase}\n"
            f"  experiment phase topic: {phase_topic or '(disabled)'}\n"
            f"  absolute initialization position: "
            f"{self._initialization_reference.position.tolist()}\n"
            f"  workspace barrier enabled: {workspace_config.enabled}\n"
            f"  workspace adaptive: {workspace_config.adaptive}\n"
            f"  dry run: {self.dry_run}"
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
                "Ignoring unsupported FORMATION -> INITIALIZE transition."
            )
            return

        previous_phase = self._experiment_phase
        self._experiment_phase = phase

        # INITIALIZE uses an explicit absolute initialization reference.
        # Any stationary/velocity reference created earlier may therefore still
        # point to the original spawn state.  When FORMATION is released,
        # re-anchor the active leader reference at the ACTUAL current state so
        # the phase transition itself is reference-continuous.
        if previous_phase == "INITIALIZE" and phase == "FORMATION":
            if self._state is None:
                self.get_logger().warn(
                    "FORMATION released before a leader state was available; "
                    "the leader reference could not be reset."
                )
            else:
                current_position = self._state[:3].copy()
                rotation = rotation_matrix_from_quaternion(
                    self._state[3:7]
                )
                current_velocity = rotation @ self._state[7:10]

                if self.reference_mode == "stationary":
                    self._stationary_reference = LeaderTrajectorySample(
                        position=current_position,
                        velocity=np.zeros(3),
                        acceleration=np.zeros(3),
                    )
                    self.get_logger().info(
                        "Reset stationary leader reference at FORMATION "
                        f"transition to {current_position.tolist()}."
                    )

                elif self.reference_mode == "velocity":
                    self._velocity_reference = (
                        VelocityCommandReference.initialize(
                            current_position,
                            current_velocity,
                            float(
                                self.get_parameter(
                                    "velocity_command_bandwidth"
                                ).value
                            ),
                        )
                    )
                    # Do not allow a command received during INITIALIZE to be
                    # applied immediately when FORMATION starts.
                    self._velocity_command = np.zeros(3, dtype=float)
                    self._velocity_command_receipt = -np.inf

                    # Respect the currently available workspace immediately.
                    bounds = self.runtime.workspace_reference_bounds()
                    if bounds is not None:
                        self._velocity_reference.project_to_box(*bounds)

                    self.get_logger().info(
                        "Reset leader velocity reference at FORMATION "
                        "transition from current state: "
                        f"position={current_position.tolist()}, "
                        f"velocity={current_velocity.tolist()}."
                    )

        self.get_logger().info(
            f"Experiment phase changed to {self._experiment_phase}."
        )

    def _now_seconds(self) -> float:
        return 1e-9 * float(self.get_clock().now().nanoseconds)

    def _px4_odom_callback(
        self,
        message: VehicleOdometry,
    ) -> None:
        try:
            state = px4_vehicle_odometry_to_core_state(message)
        except ValueError as error:
            self.get_logger().error(
                f"Invalid VehicleOdometry frame: {error}",
                throttle_duration_sec=2.0,
            )
            return
        self._accept_state(state)

    def _odom_callback(self, message: Odometry) -> None:
        self._accept_state(
            odometry_to_core_state(message, self.frames)
        )

    def _accept_state(self, state: np.ndarray) -> None:
        self._state = state
        self._state_receipt = self._now_seconds()

        if self._stationary_reference is None:
            self._stationary_reference = LeaderTrajectorySample(
                position=self._state[:3].copy(),
                velocity=np.zeros(3),
                acceleration=np.zeros(3),
            )

        if (
            self.reference_mode == "velocity"
            and self._velocity_reference is None
        ):
            rotation = rotation_matrix_from_quaternion(
                self._state[3:7]
            )
            inertial_velocity = rotation @ self._state[7:10]
            self._velocity_reference = (
                VelocityCommandReference.initialize(
                    self._state[:3],
                    inertial_velocity,
                    float(
                        self.get_parameter(
                            "velocity_command_bandwidth"
                        ).value
                    ),
                )
            )

    def _accept_velocity_command(
        self,
        incoming_world: np.ndarray,
    ) -> None:
        if not np.all(np.isfinite(incoming_world)):
            self.get_logger().error(
                "Rejected non-finite leader velocity command."
            )
            return
        self._velocity_command = self.frames.world_vector_to_core(
            incoming_world
        )
        self._velocity_command_receipt = self._now_seconds()

    def _velocity_command_callback(
        self,
        message: TwistStamped,
    ) -> None:
        self._accept_velocity_command(
            np.array(
                [
                    message.twist.linear.x,
                    message.twist.linear.y,
                    message.twist.linear.z,
                ],
                dtype=float,
            )
        )

    def _cmd_vel_callback(
        self,
        message: Twist,
    ) -> None:
        self._accept_velocity_command(
            np.array(
                [
                    message.linear.x,
                    message.linear.y,
                    message.linear.z,
                ],
                dtype=float,
            )
        )

    def _active_velocity_command(self) -> np.ndarray:
        timeout = float(
            self.get_parameter("velocity_command_timeout").value
        )
        if timeout <= 0.0:
            raise ValueError("velocity_command_timeout must be positive.")
        if (
            self._now_seconds() - self._velocity_command_receipt
            > timeout
        ):
            return np.zeros(3, dtype=float)
        return self._velocity_command

    def _trajectory_point_callback(
        self,
        message: MultiDOFJointTrajectoryPoint,
    ) -> None:
        if not message.transforms:
            self.get_logger().error(
                "Trajectory point has no transform."
            )
            return
        if not message.velocities:
            self.get_logger().error(
                "Trajectory point has no velocity."
            )
            return
        if not message.accelerations:
            self.get_logger().error(
                "Trajectory point has no acceleration."
            )
            return

        transform = message.transforms[0]
        velocity = message.velocities[0]
        acceleration = message.accelerations[0]

        position = self.frames.world_position_to_core(
            np.array(
                [
                    transform.translation.x,
                    transform.translation.y,
                    transform.translation.z,
                ]
            )
        )
        linear_velocity = self.frames.world_vector_to_core(
            np.array(
                [
                    velocity.linear.x,
                    velocity.linear.y,
                    velocity.linear.z,
                ]
            )
        )
        linear_acceleration = self.frames.world_vector_to_core(
            np.array(
                [
                    acceleration.linear.x,
                    acceleration.linear.y,
                    acceleration.linear.z,
                ]
            )
        )

        self._trajectory_reference = LeaderTrajectorySample(
            position=position,
            velocity=linear_velocity,
            acceleration=linear_acceleration,
        )
        self._trajectory_receipt = self._now_seconds()

    def _state_fresh(self) -> bool:
        return (
            self._state is not None
            and self._now_seconds() - self._state_receipt
            <= self.measurement_timeout
        )

    def _reference(self) -> LeaderTrajectorySample | None:
        if self._experiment_phase == "INITIALIZE":
            return self._initialization_reference

        if self.reference_mode == "stationary":
            return self._stationary_reference

        if self.reference_mode == "velocity":
            if self._velocity_reference is None:
                return None
            return self._velocity_reference.sample(
                self._active_velocity_command()
            )

        if self._trajectory_reference is None:
            return None
        if (
            self._now_seconds() - self._trajectory_receipt
            > self.measurement_timeout
        ):
            return None
        return self._trajectory_reference

    def _control_step(self) -> None:
        if not self.px4.enabled and not self.dry_run:
            return

        if not self._state_fresh():
            self.get_logger().warn(
                "Waiting for fresh leader odometry.",
                throttle_duration_sec=2.0,
            )
            if not self.dry_run:
                self.px4.publish_zero()
            self.diagnostics.publish_fallback()
            self.snapshot_publisher.publish_fallback(role=1.0)
            return

        reference = self._reference()
        if reference is None:
            self.get_logger().warn(
                "Waiting for leader reference.",
                throttle_duration_sec=2.0,
            )
            if not self.dry_run:
                self.px4.publish_zero()
            self.diagnostics.publish_fallback()
            self.snapshot_publisher.publish_fallback(role=1.0)
            return

        assert self._state is not None
        try:
            result = self.runtime.step(
                self._state,
                reference,
                dt=self.dt,
            )
        except Exception as error:
            self.get_logger().error(
                f"Leader core evaluation failed: {error}",
                throttle_duration_sec=1.0,
            )
            if not self.dry_run:
                self.px4.publish_zero()
            self.diagnostics.publish_fallback()
            self.snapshot_publisher.publish_fallback(role=1.0)
            return

        if not self.dry_run:
            self.px4.publish_core_wrench(result.wrench_body)
        self.diagnostics.publish(
            result.diagnostics,
            fallback=False,
        )
        if result.snapshot is not None:
            self.snapshot_publisher.publish(result.snapshot)

        if (
            self._experiment_phase == "FORMATION"
            and self.reference_mode == "velocity"
            and self._velocity_reference is not None
        ):
            self._velocity_reference.advance(
                self._active_velocity_command(),
                self.dt,
            )
            bounds = self.runtime.workspace_reference_bounds()
            if bounds is not None:
                self._velocity_reference.project_to_box(*bounds)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LeaderControllerNode()
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
