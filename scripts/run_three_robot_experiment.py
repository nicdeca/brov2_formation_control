#!/usr/bin/env python3
"""Run reproducible three-BlueROV formation experiments.

Leader velocity commands are expressed in the current pool-aligned core NWU
frame.

The script may be started before arming. It publishes the transient-local
mission-status lifecycle expected by the split experiment recorder:

    WAITING -> RUNNING -> COMPLETE

or ABORTED on interruption/error.

Profiles:
  cautious  Moderate first wet-test validation.
  full      Large-excursion paper demonstration.
"""

from __future__ import annotations

import argparse
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import String


PHASE_TOPIC = "/formation_control/experiment_phase"
FORMATION_TOPIC = "/formation_control/desired_formation"
MISSION_STATUS_TOPIC = "/formation_control/mission_status"


class ThreeRobotExperimentRunner(Node):
    def __init__(self, leader: str, expected_followers: int = 2) -> None:
        super().__init__("three_robot_experiment_runner")
        self.expected_followers = int(expected_followers)

        transient_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        command_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.formation_pub = self.create_publisher(
            String,
            FORMATION_TOPIC,
            transient_qos,
        )
        self.mission_status_pub = self.create_publisher(
            String,
            MISSION_STATUS_TOPIC,
            transient_qos,
        )
        self.cmd_vel_topic = f"/{leader}/formation_control/cmd_vel"
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            self.cmd_vel_topic,
            command_qos,
        )

        self.phase: str | None = None
        self.create_subscription(
            String,
            PHASE_TOPIC,
            self._phase_callback,
            transient_qos,
        )

    def publish_mission_status(self, status: str) -> None:
        message = String()
        message.data = status.strip().upper()
        self.mission_status_pub.publish(message)
        self.get_logger().info(f"MISSION_STATUS {message.data}")

    def _phase_callback(self, message: String) -> None:
        self.phase = message.data.strip().upper()

    def _spin_sleep(self, duration: float) -> None:
        deadline = time.monotonic() + duration
        while rclpy.ok() and time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            rclpy.spin_once(
                self,
                timeout_sec=min(0.05, remaining),
            )

    def wait_for_phase(self, desired: str) -> None:
        desired = desired.upper()
        self.get_logger().info(
            f"Waiting for experiment phase {desired}. "
            "Arm and switch all three robots to Offboard when ready."
        )
        while rclpy.ok() and self.phase != desired:
            rclpy.spin_once(self, timeout_sec=0.2)
        if not rclpy.ok():
            raise KeyboardInterrupt
        self.get_logger().info(f"Observed experiment phase {desired}.")

    def wait_for_subscribers(self) -> None:
        self.get_logger().info(
            "Waiting for both follower formation-command subscriptions."
        )
        while (
            rclpy.ok()
            and self.formation_pub.get_subscription_count()
            < self.expected_followers
        ):
            rclpy.spin_once(self, timeout_sec=0.2)

        count = self.formation_pub.get_subscription_count()
        self.get_logger().info(
            f"Formation publisher sees {count} subscriber(s)."
        )

        self.get_logger().info(
            "Waiting for the leader cmd_vel subscription."
        )
        while rclpy.ok() and self.cmd_vel_pub.get_subscription_count() < 1:
            rclpy.spin_once(self, timeout_sec=0.2)

        self.get_logger().info(
            "Leader cmd_vel subscriber discovered."
        )

    def publish_formation(
        self,
        name: str,
        *,
        duration: float = 2.0,
        rate_hz: float = 10.0,
    ) -> None:
        count = self.formation_pub.get_subscription_count()
        if count < self.expected_followers:
            raise RuntimeError(
                "Formation command lost expected subscribers: "
                f"found {count}, expected at least {self.expected_followers}."
            )

        self.get_logger().info(
            f"COMMAND formation={name!r} "
            f"(publishing for {duration:.1f} s to {count} subscribers)"
        )

        message = String()
        message.data = name
        period = 1.0 / rate_hz
        deadline = time.monotonic() + duration
        while rclpy.ok() and time.monotonic() < deadline:
            self.formation_pub.publish(message)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(period)

    def publish_velocity(
        self,
        vx: float,
        vy: float,
        vz: float,
        *,
        duration: float,
        rate_hz: float = 20.0,
    ) -> None:
        if self.cmd_vel_pub.get_subscription_count() < 1:
            raise RuntimeError("Leader cmd_vel subscriber disappeared.")

        self.get_logger().info(
            "COMMAND leader velocity "
            f"[{vx:.3f}, {vy:.3f}, {vz:.3f}] m/s "
            f"for {duration:.1f} s"
        )

        message = Twist()
        message.linear.x = float(vx)
        message.linear.y = float(vy)
        message.linear.z = float(vz)

        period = 1.0 / rate_hz
        deadline = time.monotonic() + duration
        while rclpy.ok() and time.monotonic() < deadline:
            self.cmd_vel_pub.publish(message)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(period)

        self.stop_leader()

    def stop_leader(self) -> None:
        message = Twist()
        for _ in range(10):
            if not rclpy.ok():
                return
            self.cmd_vel_pub.publish(message)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(0.05)

    def settle(self, duration: float) -> None:
        self.get_logger().info(f"SETTLE {duration:.1f} s")
        self._spin_sleep(duration)

    def run_cautious(self) -> None:
        """Moderate pool-centered first wet-test experiment."""
        self.get_logger().info(
            "=== CAUTIOUS THREE-ROBOT EXPERIMENT START ==="
        )

        self.publish_formation("triangle_nominal")
        self.settle(8.0)

        # Small +x excursion from the pool-centered initialization.
        self.publish_velocity(0.10, 0.0, 0.0, duration=3.0)
        self.settle(6.0)

        self.publish_formation("triangle_wide")
        self.settle(10.0)

        self.publish_formation("triangle_compact")
        self.settle(10.0)

        # Traverse to the opposite side of the central operating region.
        self.publish_velocity(-0.10, 0.0, 0.0, duration=6.0)
        self.settle(6.0)

        self.publish_formation("triangle_nominal")
        self.settle(8.0)

        # Small symmetric y excursion while using the nominal triangle.
        self.publish_velocity(0.0, -0.10, 0.0, duration=1.5)
        self.settle(5.0)
        self.publish_velocity(0.0, 0.10, 0.0, duration=3.0)
        self.settle(5.0)
        self.publish_velocity(0.0, -0.10, 0.0, duration=1.5)
        self.settle(5.0)

        # Return x to the initialization value.
        self.publish_velocity(0.10, 0.0, 0.0, duration=3.0)
        self.settle(8.0)

        self.publish_formation("triangle_nominal")
        self.settle(8.0)
        self.stop_leader()
        self.get_logger().info(
            "=== CAUTIOUS THREE-ROBOT EXPERIMENT COMPLETE ==="
        )

    def run_full(self) -> None:
        """Paper-oriented experiment centered in the reliable MoCap volume."""
        self.get_logger().info(
            "=== FULL THREE-ROBOT EXPERIMENT START ==="
        )

        self.publish_formation("triangle_nominal")
        self.settle(8.0)

        # +0.50 m in pool-frame x.
        self.publish_velocity(0.10, 0.0, 0.0, duration=5.0)
        self.settle(6.0)

        # Formation expansion is performed near y=0, away from both side walls.
        self.publish_formation("triangle_wide")
        self.settle(10.0)

        self.publish_formation("triangle_compact")
        self.settle(10.0)

        # -1.00 m: cross the pool-centered operating region symmetrically.
        self.publish_velocity(-0.10, 0.0, 0.0, duration=10.0)
        self.settle(6.0)

        self.publish_formation("triangle_wide")
        self.settle(10.0)

        self.publish_formation("triangle_compact")
        self.settle(10.0)

        # Return to the central x initialization before lateral maneuvers.
        self.publish_velocity(0.10, 0.0, 0.0, duration=5.0)
        self.settle(6.0)

        self.publish_formation("triangle_nominal")
        self.settle(8.0)

        # Symmetric +/-0.25 m leader-y excursion. With the nominal triangle,
        # both followers remain well inside the central MoCap/workspace region.
        self.publish_velocity(0.0, -0.10, 0.0, duration=2.5)
        self.settle(6.0)
        self.publish_velocity(0.0, 0.10, 0.0, duration=5.0)
        self.settle(6.0)
        self.publish_velocity(0.0, -0.10, 0.0, duration=2.5)
        self.settle(8.0)

        self.publish_formation("triangle_nominal")
        self.settle(8.0)
        self.stop_leader()
        self.get_logger().info(
            "=== FULL THREE-ROBOT EXPERIMENT COMPLETE ==="
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leader", default="itrl_rov_1")
    parser.add_argument(
        "--profile",
        choices=("cautious", "full"),
        default="cautious",
    )
    args = parser.parse_args()

    rclpy.init()
    node = ThreeRobotExperimentRunner(args.leader)
    node.publish_mission_status("WAITING")

    try:
        node.wait_for_phase("FORMATION")
        node.wait_for_subscribers()

        node.publish_mission_status("RUNNING")

        # Give record_formation_experiment.py time to close initialization,
        # observe RUNNING, and open RUN/mission/bag before the first command.
        node._spin_sleep(1.0)

        if args.profile == "cautious":
            node.run_cautious()
        else:
            node.run_full()

        node.publish_mission_status("COMPLETE")

    except KeyboardInterrupt:
        node.publish_mission_status("ABORTED")
        node.get_logger().warn(
            "Experiment interrupted; commanding zero leader velocity."
        )
        node.stop_leader()
    except Exception as error:
        node.publish_mission_status("ABORTED")
        node.get_logger().error(f"Experiment aborted: {error}")
        node.stop_leader()
        raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
