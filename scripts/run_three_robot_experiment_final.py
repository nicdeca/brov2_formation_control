#!/usr/bin/env python3
"""Run the reproducible three-BlueROV formation experiment.

The script may be started before the robots are armed.  It waits for the
global FORMATION phase, verifies that the expected subscribers are present,
then executes a timed sequence of formation switches and leader velocity
commands.

All velocity commands use the leader's standard cmd_vel interface, so the
same wall-aware reference projection and CLF-QP used by keyboard teleoperation
remain active.
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


class ThreeRobotExperimentRunner(Node):
    def __init__(self, leader: str) -> None:
        super().__init__("three_robot_experiment_runner")

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
            rclpy.spin_once(self, timeout_sec=min(0.05, deadline - time.monotonic()))

    def wait_for_phase(self, desired: str) -> None:
        desired = desired.upper()
        self.get_logger().info(
            f"Waiting for experiment phase {desired}. "
            "You can arm/Offboard the robots now."
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
        while rclpy.ok() and self.formation_pub.get_subscription_count() < 2:
            rclpy.spin_once(self, timeout_sec=0.2)

        self.get_logger().info(
            "Formation publisher sees "
            f"{self.formation_pub.get_subscription_count()} subscriber(s)."
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
        if count < 2:
            raise RuntimeError(
                "Formation command lost expected subscribers: "
                f"found {count}, expected at least 2."
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

    def run_sequence(self) -> None:
        """Execute the final, large-excursion three-robot demonstration.

        Approximate leader-reference excursion:
            +1.4 m in y
            +0.7 m in x
            -1.4 m in y
            -0.7 m in x

        The x displacement is intentionally performed only after the formation
        has contracted to triangle_compact, giving both followers substantial
        clearance from the lateral tank walls.
        """
        self.get_logger().info("=== FINAL THREE-ROBOT EXPERIMENT START ===")

        # 1) Baseline.
        self.publish_formation("triangle_nominal")
        self.settle(8.0)

        # 2) Long translation in +y: approximately +1.4 m.
        self.publish_velocity(0.0, 0.20, 0.0, duration=7.0)
        self.settle(8.0)

        # 3) Strong expansion.
        self.publish_formation("triangle_wide")
        self.settle(12.0)

        # 4) Strong contraction before the larger lateral leader maneuver.
        self.publish_formation("triangle_compact")
        self.settle(12.0)

        # 5) Larger +x translation: approximately +0.7 m.
        self.publish_velocity(0.14, 0.0, 0.0, duration=5.0)
        self.settle(8.0)

        # 6) Vertical deformation at the displaced leader location.
        self.publish_formation("triangle_high")
        self.settle(12.0)

        # 7) Undo the y displacement while holding triangle_high.
        self.publish_velocity(0.0, -0.20, 0.0, duration=7.0)
        self.settle(8.0)

        # 8) Return to nominal shape.
        self.publish_formation("triangle_nominal")
        self.settle(10.0)

        # 9) Undo the x displacement.
        self.publish_velocity(-0.14, 0.0, 0.0, duration=5.0)
        self.settle(12.0)

        self.stop_leader()
        self.get_logger().info("=== FINAL THREE-ROBOT EXPERIMENT COMPLETE ===")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leader", default="itrl_rov_1")
    args = parser.parse_args()

    rclpy.init()
    node = ThreeRobotExperimentRunner(args.leader)

    try:
        node.wait_for_phase("FORMATION")
        node.wait_for_subscribers()
        node.run_sequence()
    except KeyboardInterrupt:
        node.get_logger().warn("Experiment interrupted; commanding zero velocity.")
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
