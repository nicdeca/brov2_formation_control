"""Automatic INITIALIZE -> FORMATION experiment-phase manager.

The state source is configurable so initialization is evaluated from the same
odometry stream used by the leader/follower controllers.
"""

from __future__ import annotations

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from px4_msgs.msg import VehicleOdometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from .frame_conventions import FrameConvention
from .px4_interface import px4_qos_profile
from .state_adapter import (
    odometry_to_core_state,
    px4_vehicle_odometry_to_core_state,
)


def _topic_from_template(
    template: str,
    robot: str,
    state_source: str,
) -> str:
    template = template.strip()
    if not template:
        if state_source == "px4":
            return f"/{robot}/fmu/out/vehicle_odometry"
        return f"/mocap/{robot.lower()}/odom"

    return (
        template
        .replace("{robot}", robot)
        .replace("{robot_lower}", robot.lower())
    )


class ExperimentPhaseManager(Node):
    """Wait for fixed absolute initialization poses, then release formation.

    ``initial_positions`` contains one core-NWU position per robot. The manager
    can consume either PX4 ``VehicleOdometry`` or generic ``nav_msgs/Odometry``.

    ``state_topic_template`` may contain ``{robot}`` or ``{robot_lower}``.
    """

    def __init__(self) -> None:
        super().__init__("formation_experiment_phase_manager")

        self.declare_parameter(
            "robot_names",
            ["itrl_rov_1", "itrl_rov_2", "itrl_rov_3"],
        )
        self.declare_parameter(
            "initial_positions",
            [
                -2.20, 1.15, -95.70,
                -1.50, 2.95, -95.70,
                -2.90, 2.95, -95.70,
            ],
        )
        self.declare_parameter(
            "phase_topic",
            "/formation_control/experiment_phase",
        )
        self.declare_parameter("position_tolerance", 0.12)
        self.declare_parameter("speed_tolerance", 0.08)
        self.declare_parameter("settle_time", 1.5)
        self.declare_parameter("publish_period", 0.10)

        # State input. Empty template selects the normal default for the chosen
        # message type.
        self.declare_parameter("state_source", "px4")
        self.declare_parameter("state_topic_template", "")
        self.declare_parameter("mocap_world_frame", "core_nwu")
        self.declare_parameter("odom_twist_frame", "body")

        self.robots = [
            str(value)
            for value in self.get_parameter("robot_names").value
        ]
        if not self.robots:
            raise ValueError("robot_names must not be empty.")
        if len(set(self.robots)) != len(self.robots):
            raise ValueError("robot_names must be unique.")

        self.state_source = str(
            self.get_parameter("state_source").value
        ).strip().lower()
        if self.state_source not in ("px4", "nav_msgs"):
            raise ValueError(
                "state_source must be 'px4' or 'nav_msgs'."
            )

        self.state_topic_template = str(
            self.get_parameter("state_topic_template").value
        )
        if (
            len(self.robots) > 1
            and self.state_topic_template.strip()
            and "{robot}" not in self.state_topic_template
            and "{robot_lower}" not in self.state_topic_template
        ):
            raise ValueError(
                "state_topic_template must contain '{robot}' or "
                "'{robot_lower}' when more than one robot is used."
            )

        self.frames = FrameConvention(
            world_frame=str(
                self.get_parameter("mocap_world_frame").value
            ),
            twist_frame=str(
                self.get_parameter("odom_twist_frame").value
            ),
        )

        flat = np.asarray(
            self.get_parameter("initial_positions").value,
            dtype=float,
        ).reshape(-1)
        if flat.size != 3 * len(self.robots):
            raise ValueError(
                "initial_positions must contain three values per robot."
            )
        initial_positions = flat.reshape(len(self.robots), 3)
        if not np.all(np.isfinite(initial_positions)):
            raise ValueError("initial_positions must be finite.")
        self.targets = {
            robot: initial_positions[index].copy()
            for index, robot in enumerate(self.robots)
        }

        self.position_tolerance = float(
            self.get_parameter("position_tolerance").value
        )
        self.speed_tolerance = float(
            self.get_parameter("speed_tolerance").value
        )
        self.settle_time = float(
            self.get_parameter("settle_time").value
        )
        publish_period = float(
            self.get_parameter("publish_period").value
        )
        if (
            self.position_tolerance <= 0.0
            or self.speed_tolerance <= 0.0
            or self.settle_time <= 0.0
            or publish_period <= 0.0
        ):
            raise ValueError(
                "phase-manager tolerances/times must be positive."
            )

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        phase_topic = str(
            self.get_parameter("phase_topic").value
        )
        self.publisher = self.create_publisher(
            String,
            phase_topic,
            qos,
        )

        self.states: dict[str, np.ndarray] = {}
        self.phase = "INITIALIZE"
        self._inside_since: float | None = None

        self.state_topics = {
            robot: _topic_from_template(
                self.state_topic_template,
                robot,
                self.state_source,
            )
            for robot in self.robots
        }

        self._subscriptions = []
        for robot in self.robots:
            topic = self.state_topics[robot]
            if self.state_source == "px4":
                subscription = self.create_subscription(
                    VehicleOdometry,
                    topic,
                    self._px4_callback(robot),
                    px4_qos_profile(),
                )
            else:
                subscription = self.create_subscription(
                    Odometry,
                    topic,
                    self._odom_callback(robot),
                    10,
                )
            self._subscriptions.append(subscription)

        self.create_timer(publish_period, self._tick)

        self.get_logger().info(
            "Experiment phase manager configured:\n"
            f"  robots: {tuple(self.robots)}\n"
            f"  state source: {self.state_source}\n"
            f"  state topics: {self.state_topics}\n"
            f"  generic-odom world frame: {self.frames.world_frame}\n"
            f"  odom twist frame: {self.frames.twist_frame}\n"
            f"  phase topic: {phase_topic}\n"
            "  absolute initialization targets:\n"
            + "\n".join(
                f"    {robot}: {self.targets[robot].tolist()}"
                for robot in self.robots
            )
            + "\n"
            f"  position tolerance: {self.position_tolerance:.3f} m\n"
            f"  speed tolerance: {self.speed_tolerance:.3f} m/s\n"
            f"  settle time: {self.settle_time:.3f} s"
        )

    def _now_seconds(self) -> float:
        return 1e-9 * float(self.get_clock().now().nanoseconds)

    def _accept_state(self, robot: str, state: np.ndarray) -> None:
        self.states[robot] = state

    def _px4_callback(self, robot: str):
        def callback(message: VehicleOdometry) -> None:
            try:
                self._accept_state(
                    robot,
                    px4_vehicle_odometry_to_core_state(message),
                )
            except ValueError as error:
                self.get_logger().error(
                    f"Invalid VehicleOdometry for {robot}: {error}",
                    throttle_duration_sec=2.0,
                )
        return callback

    def _odom_callback(self, robot: str):
        def callback(message: Odometry) -> None:
            try:
                self._accept_state(
                    robot,
                    odometry_to_core_state(message, self.frames),
                )
            except ValueError as error:
                self.get_logger().error(
                    f"Invalid Odometry for {robot}: {error}",
                    throttle_duration_sec=2.0,
                )
        return callback

    def _all_settled(self) -> bool:
        if not self.targets:
            return False

        for robot in self.robots:
            state = self.states.get(robot)
            if state is None:
                return False

            position_error = float(
                np.linalg.norm(state[:3] - self.targets[robot])
            )
            speed = float(np.linalg.norm(state[7:10]))

            if (
                position_error > self.position_tolerance
                or speed > self.speed_tolerance
            ):
                return False

        return True

    def _tick(self) -> None:
        if self.phase == "INITIALIZE" and self._all_settled():
            now = self._now_seconds()
            if self._inside_since is None:
                self._inside_since = now
                self.get_logger().info(
                    "All robots are inside initialization tolerances; "
                    "starting settle dwell."
                )
            elif now - self._inside_since >= self.settle_time:
                self.phase = "FORMATION"
                self.get_logger().info(
                    "Initialization complete. Releasing FORMATION control."
                )
        else:
            if self.phase == "INITIALIZE":
                self._inside_since = None

        message = String()
        message.data = self.phase
        self.publisher.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ExperimentPhaseManager()
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
