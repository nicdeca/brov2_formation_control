#!/usr/bin/env python3
"""Run the five-BlueROV balanced-tree paper experiment.

Graph:
    4 -> 2 -> 1
    5 -> 3 -> 1

The experiment may be started before arming.  It waits for FORMATION, verifies
all four follower subscriptions, then executes a reproducible sequence.
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


class FiveRobotTreeExperimentRunner(Node):
    def __init__(self, leader: str) -> None:
        super().__init__("five_robot_tree_experiment_runner")

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
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            f"/{leader}/formation_control/cmd_vel",
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

    def spin_sleep(self, duration: float) -> None:
        deadline = time.monotonic() + duration
        while rclpy.ok() and time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            rclpy.spin_once(self, timeout_sec=min(0.05, remaining))

    def wait_until_ready(self) -> None:
        self.get_logger().info(
            "Waiting for FORMATION. Arm/Offboard all five robots when ready."
        )
        while rclpy.ok() and self.phase != "FORMATION":
            rclpy.spin_once(self, timeout_sec=0.2)

        self.get_logger().info("FORMATION observed.")

        while rclpy.ok() and self.formation_pub.get_subscription_count() < 4:
            rclpy.spin_once(self, timeout_sec=0.2)
        self.get_logger().info(
            f"Formation publisher sees "
            f"{self.formation_pub.get_subscription_count()} subscribers."
        )

        while rclpy.ok() and self.cmd_vel_pub.get_subscription_count() < 1:
            rclpy.spin_once(self, timeout_sec=0.2)
        self.get_logger().info("Leader cmd_vel subscriber discovered.")

    def formation(self, name: str, duration: float = 2.0) -> None:
        if self.formation_pub.get_subscription_count() < 4:
            raise RuntimeError("Expected four formation subscribers.")

        self.get_logger().info(f"COMMAND formation={name!r}")
        message = String(data=name)
        deadline = time.monotonic() + duration
        while rclpy.ok() and time.monotonic() < deadline:
            self.formation_pub.publish(message)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(0.1)

    def stop_leader(self) -> None:
        message = Twist()
        for _ in range(10):
            self.cmd_vel_pub.publish(message)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(0.05)

    def velocity(
        self,
        vx: float,
        vy: float,
        vz: float,
        duration: float,
    ) -> None:
        self.get_logger().info(
            f"COMMAND leader velocity [{vx:.2f}, {vy:.2f}, {vz:.2f}] "
            f"for {duration:.1f} s"
        )
        message = Twist()
        message.linear.x = vx
        message.linear.y = vy
        message.linear.z = vz

        deadline = time.monotonic() + duration
        while rclpy.ok() and time.monotonic() < deadline:
            self.cmd_vel_pub.publish(message)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(0.05)
        self.stop_leader()

    def settle(self, duration: float) -> None:
        self.get_logger().info(f"SETTLE {duration:.1f} s")
        self.spin_sleep(duration)

    def run(self) -> None:
        self.get_logger().info(
            "=== FIVE-ROBOT BALANCED-TREE EXPERIMENT START ==="
        )

        self.formation("tree_nominal")
        self.settle(8.0)

        # A moderate y translation tests propagation down both two-hop branches.
        self.velocity(0.0, 0.20, 0.0, 2.0)  # about +0.4 m
        self.settle(8.0)

        # Expand both levels of the tree.
        self.formation("tree_wide")
        self.settle(14.0)

        # Contract both levels.
        self.formation("tree_compact")
        self.settle(12.0)

        # Translate laterally while compact.
        self.velocity(0.15, 0.0, 0.0, 3.0)  # about +0.45 m
        self.settle(8.0)

        # Depth-staggered two-level tree.
        self.formation("tree_staggered")
        self.settle(12.0)

        # Undo y while keeping the hierarchical formation.
        self.velocity(0.0, -0.20, 0.0, 2.0)  # about -0.4 m
        self.settle(8.0)

        self.formation("tree_nominal")
        self.settle(10.0)

        # Undo x.
        self.velocity(-0.15, 0.0, 0.0, 3.0)
        self.settle(12.0)

        self.stop_leader()
        self.get_logger().info(
            "=== FIVE-ROBOT BALANCED-TREE EXPERIMENT COMPLETE ==="
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leader", default="itrl_rov_1")
    args = parser.parse_args()

    rclpy.init()
    node = FiveRobotTreeExperimentRunner(args.leader)
    try:
        node.wait_until_ready()
        node.run()
    except KeyboardInterrupt:
        node.get_logger().warn(
            "Experiment interrupted; commanding zero leader velocity."
        )
        node.stop_leader()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
