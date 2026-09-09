#!/usr/bin/env python3
"""Run reproducible two-BlueROV formation experiments.

Leader velocity commands are expressed in the pool-aligned core NWU frame.
The absolute mission depth is set by the INITIALIZE targets in the launch
configuration.  All wet-test profiles below keep leader vertical velocity
at zero so that the mission remains around that initialized depth.

The script can be started before arming.  It waits for FORMATION and for the
leader/follower command subscriptions.

Profiles:
  cautious     First wet-test validation.
  full         Standard research experiment.
  challenging  Deliberately excites adaptive sensing limits and uses faster,
               larger leader maneuvers.  Validate cautious/full first.
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


class TwoRobotExperimentRunner(Node):
    def __init__(self, leader: str) -> None:
        super().__init__("two_robot_experiment_runner")

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
        self.mission_status_pub = self.create_publisher(
            String,
            MISSION_STATUS_TOPIC,
            phase_qos,
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
            rclpy.spin_once(self, timeout_sec=min(0.05, remaining))

    def wait_for_phase(self, desired: str) -> None:
        desired = desired.upper()
        self.get_logger().info(
            f"Waiting for experiment phase {desired}. "
            "Arm and switch both robots to Offboard when ready."
        )
        while rclpy.ok() and self.phase != desired:
            rclpy.spin_once(self, timeout_sec=0.2)
        if not rclpy.ok():
            raise KeyboardInterrupt
        self.get_logger().info(f"Observed experiment phase {desired}.")

    def wait_for_subscribers(self) -> None:
        self.get_logger().info(
            "Waiting for the follower formation-command subscription."
        )
        while rclpy.ok() and self.formation_pub.get_subscription_count() < 1:
            rclpy.spin_once(self, timeout_sec=0.2)
        self.get_logger().info(
            "Follower formation-command subscriber discovered."
        )

        self.get_logger().info("Waiting for the leader cmd_vel subscription.")
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
        if self.formation_pub.get_subscription_count() < 1:
            raise RuntimeError(
                "Follower formation-command subscriber disappeared."
            )
        self.get_logger().info(f"COMMAND formation={name!r}")

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

    def run_cautious(self) -> None:
        """First wet test, kept close to the pool/MoCap center."""
        self.get_logger().info(
            "=== CAUTIOUS TWO-ROBOT EXPERIMENT START ==="
        )

        self.publish_formation("pair_nominal")
        self.settle(10.0)

        self.publish_formation("pair_far")
        self.settle(10.0)

        self.publish_formation("pair_nominal")
        self.settle(8.0)

        # Symmetric +/-0.25 m x excursion about initialization.
        self.publish_velocity(0.10, 0.0, 0.0, duration=2.5)
        self.settle(6.0)
        self.publish_velocity(-0.10, 0.0, 0.0, duration=5.0)
        self.settle(6.0)
        self.publish_velocity(0.10, 0.0, 0.0, duration=2.5)
        self.settle(8.0)

        self.publish_formation("pair_close")
        self.settle(10.0)

        self.publish_formation("pair_nominal")
        self.settle(10.0)
        self.stop_leader()
        self.get_logger().info(
            "=== CAUTIOUS TWO-ROBOT EXPERIMENT COMPLETE ==="
        )

    def run_full(self) -> None:
        """Standard paper experiment centered in the reliable MoCap volume."""
        self.get_logger().info(
            "=== FULL TWO-ROBOT EXPERIMENT START ==="
        )

        self.publish_formation("pair_nominal")
        self.settle(8.0)

        # +0.45 m x excursion.
        self.publish_velocity(0.09, 0.0, 0.0, duration=5.0)
        self.settle(6.0)

        self.publish_formation("pair_far")
        self.settle(10.0)

        self.publish_formation("pair_close")
        self.settle(10.0)

        # Cross to -0.45 m relative to initialization.
        self.publish_velocity(-0.09, 0.0, 0.0, duration=10.0)
        self.settle(6.0)

        self.publish_formation("pair_nominal")
        self.settle(8.0)

        # Return to the central x initialization.
        self.publish_velocity(0.09, 0.0, 0.0, duration=5.0)
        self.settle(6.0)

        # Symmetric +/-0.20 m y excursion.
        self.publish_velocity(0.0, -0.08, 0.0, duration=2.5)
        self.settle(6.0)
        self.publish_velocity(0.0, 0.08, 0.0, duration=5.0)
        self.settle(6.0)
        self.publish_velocity(0.0, -0.08, 0.0, duration=2.5)
        self.settle(8.0)

        # Keep the wet experiment at the initialized depth. pair_high is
        # deliberately not used because it moves the follower toward the
        # near-surface region where MoCap tracking degrades.
        self.publish_formation("pair_nominal")
        self.settle(10.0)
        self.stop_leader()
        self.get_logger().info(
            "=== FULL TWO-ROBOT EXPERIMENT COMPLETE ==="
        )

    def run_challenging(self) -> None:
        """Stress sensing adaptation while remaining pool-centered.

        The challenging profile pushes sensing/FoV constraints through the
        relative-formation references rather than by driving the entire pair
        toward a pool wall.
        """
        self.get_logger().info(
            "=== CHALLENGING TWO-ROBOT EXPERIMENT START ==="
        )

        self.publish_formation("pair_nominal")
        self.settle(6.0)

        # Range-upper-bound challenge: ||d|| ~= 3.18 m.
        self.publish_formation("pair_range_far_edge")
        self.settle(10.0)

        self.publish_formation("pair_nominal")
        self.settle(6.0)

        # Move +0.40 m in x while remaining near the pool center.
        self.publish_velocity(0.20, 0.0, 0.0, duration=2.0)
        self.settle(5.0)

        # Horizontal FoV challenge.
        self.publish_formation("pair_fov_edge")
        self.settle(10.0)

        # Move -0.80 m in x, crossing to the opposite side of center.
        self.publish_velocity(-0.20, 0.0, 0.0, duration=4.0)
        self.settle(5.0)

        # Range-lower-bound challenge: ||d|| ~= 0.70 m.
        self.publish_formation("pair_range_close_edge")
        self.settle(10.0)

        # Small symmetric lateral excursion while the pair is compact.
        self.publish_velocity(0.0, -0.15, 0.0, duration=2.0)
        self.settle(5.0)
        self.publish_velocity(0.0, 0.15, 0.0, duration=4.0)
        self.settle(5.0)
        self.publish_velocity(0.0, -0.15, 0.0, duration=2.0)
        self.settle(5.0)

        self.publish_formation("pair_nominal")
        self.settle(8.0)

        # Return the x coordinate to initialization.
        self.publish_velocity(0.20, 0.0, 0.0, duration=2.0)
        self.settle(10.0)

        self.stop_leader()
        self.get_logger().info(
            "=== CHALLENGING TWO-ROBOT EXPERIMENT COMPLETE ==="
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leader", default="itrl_rov_1")
    parser.add_argument(
        "--profile",
        choices=("cautious", "full", "challenging"),
        default="cautious",
    )
    args = parser.parse_args()

    rclpy.init()
    node = TwoRobotExperimentRunner(args.leader)
    node.publish_mission_status("WAITING")

    try:
        node.wait_for_phase("FORMATION")
        node.wait_for_subscribers()

        node.publish_mission_status("RUNNING")
        # Give the split recorder time to open RUN/mission/bag before the
        # first formation or velocity command is sent.
        node._spin_sleep(1.0)

        if args.profile == "cautious":
            node.run_cautious()
        elif args.profile == "full":
            node.run_full()
        else:
            node.run_challenging()

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
        rclpy.shutdown()


if __name__ == "__main__":
    main()
