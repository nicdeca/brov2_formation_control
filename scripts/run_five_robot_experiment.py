#!/usr/bin/env python3
"""Run the reproducible five-BlueROV paper simulation.

The script may be started before arming.  It waits for the global FORMATION
phase, verifies that all four follower formation-command subscriptions and the
leader cmd_vel subscription are present, then runs the paper demonstration.

Topology is a star with one leader and four independent followers.
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


class FiveRobotExperimentRunner(Node):
    def __init__(self, leader: str, expected_followers: int = 4) -> None:
        super().__init__("five_robot_experiment_runner")
        self.expected_followers = int(expected_followers)

        formation_qos = QoSProfile(
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
        phase_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.formation_pub = self.create_publisher(
            String,
            FORMATION_TOPIC,
            formation_qos,
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
            phase_qos,
        )

    def _phase_callback(self, message: String) -> None:
        self.phase = message.data.strip().upper()

    def _spin_sleep(self, duration: float) -> None:
        deadline = time.monotonic() + duration
        while rclpy.ok() and time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            rclpy.spin_once(self, timeout_sec=min(0.05, remaining))

    def wait_for_phase(self, desired: str) -> None:
        desired = desired.upper()
        self.get_logger().info(
            f"Waiting for experiment phase {desired}. "
            "Arm/Offboard all five robots when ready."
        )
        while rclpy.ok() and self.phase != desired:
            rclpy.spin_once(self, timeout_sec=0.2)
        if not rclpy.ok():
            raise KeyboardInterrupt
        self.get_logger().info(f"Observed experiment phase {desired}.")

    def wait_for_subscribers(self) -> None:
        self.get_logger().info(
            "Waiting for all four follower formation-command subscriptions."
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

        self.get_logger().info("Waiting for leader cmd_vel subscription.")
        while rclpy.ok() and self.cmd_vel_pub.get_subscription_count() < 1:
            rclpy.spin_once(self, timeout_sec=0.2)
        self.get_logger().info("Leader cmd_vel subscriber discovered.")

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

    def stop_leader(self) -> None:
        message = Twist()
        for _ in range(10):
            if not rclpy.ok():
                return
            self.cmd_vel_pub.publish(message)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(0.05)

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

    def settle(self, duration: float) -> None:
        self.get_logger().info(f"SETTLE {duration:.1f} s")
        self._spin_sleep(duration)

    def run_sequence(self) -> None:
        """Five-robot paper demonstration.

        Approximate leader-reference excursion:
            +1.2 m in y
            +0.45 m in x
            -1.2 m in y
            -0.45 m in x

        The x translation is performed while the formation is compact to keep
        all four followers comfortably away from the side walls.
        """
        self.get_logger().info("=== FIVE-ROBOT PAPER EXPERIMENT START ===")

        # Baseline row.
        self.publish_formation("five_nominal")
        self.settle(8.0)

        # Move the full nominal formation in +y.
        self.publish_velocity(0.0, 0.20, 0.0, duration=6.0)
        self.settle(8.0)

        # Expand the four-follower row.
        self.publish_formation("five_wide")
        self.settle(12.0)

        # Contract visibly before lateral translation.
        self.publish_formation("five_compact")
        self.settle(12.0)

        # About +0.45 m in x.
        self.publish_velocity(0.15, 0.0, 0.0, duration=3.0)
        self.settle(8.0)

        # Introduce an alternating depth pattern.
        self.publish_formation("five_staggered")
        self.settle(12.0)

        # Undo y while staggered.
        self.publish_velocity(0.0, -0.20, 0.0, duration=6.0)
        self.settle(8.0)

        # Recover the nominal row.
        self.publish_formation("five_nominal")
        self.settle(10.0)

        # Undo x.
        self.publish_velocity(-0.15, 0.0, 0.0, duration=3.0)
        self.settle(12.0)

        self.stop_leader()
        self.get_logger().info(
            "=== FIVE-ROBOT PAPER EXPERIMENT COMPLETE ==="
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leader", default="itrl_rov_1")
    args = parser.parse_args()

    rclpy.init()
    node = FiveRobotExperimentRunner(args.leader)

    try:
        node.wait_for_phase("FORMATION")
        node.wait_for_subscribers()
        node.run_sequence()
    except KeyboardInterrupt:
        node.get_logger().warn(
            "Experiment interrupted; commanding zero leader velocity."
        )
        node.stop_leader()
    except Exception as error:
        node.get_logger().error(f"Experiment aborted: {error}")
        node.stop_leader()
        raise
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
