#!/usr/bin/env python3
"""Run reproducible five-BlueROV balanced-tree experiments.

Graph:
    4 -> 2 -> 1
    5 -> 3 -> 1

The script may be started before arming. It waits for FORMATION, verifies
all four follower formation-command subscriptions and the leader cmd_vel
subscription, then executes the requested profile.

IMPORTANT: the leader controller must be launched with
``leader_reference_mode:=velocity`` for the cmd_vel commands below to move
the leader reference.

Profiles:
  cautious     Moderate validation maneuvers.
  full         Standard paper demonstration with clearly visible leader motion.
  challenging  Faster/larger maneuvers intended to excite tracking limits and
               sensing-domain adaptation. Validate full first.
"""

from __future__ import annotations

import argparse
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.parameter import Parameter
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
    def __init__(
        self,
        leader: str,
        expected_followers: int = 4,
        *,
        gazebo_timer: bool = False,
    ) -> None:
        super().__init__(
            "five_robot_experiment_runner",
            parameter_overrides=[
                Parameter("use_sim_time", value=bool(gazebo_timer)),
            ],
        )
        self.expected_followers = int(expected_followers)
        self.gazebo_timer = bool(gazebo_timer)

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

        clock_source = "Gazebo simulation time" if self.gazebo_timer else "wall time"
        self.get_logger().info(f"Experiment mission timer uses {clock_source}.")

    def _phase_callback(self, message: String) -> None:
        self.phase = message.data.strip().upper()

    def _now_seconds(self) -> float:
        if self.gazebo_timer:
            return 1e-9 * float(self.get_clock().now().nanoseconds)
        return time.monotonic()

    def _spin_sleep(self, duration: float) -> None:
        deadline = self._now_seconds() + duration
        while rclpy.ok() and self._now_seconds() < deadline:
            remaining = deadline - self._now_seconds()
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
        deadline = self._now_seconds() + duration
        while rclpy.ok() and self._now_seconds() < deadline:
            self.formation_pub.publish(message)
            rclpy.spin_once(self, timeout_sec=0.0)
            self._spin_sleep(period)

    def stop_leader(self) -> None:
        message = Twist()
        for _ in range(10):
            if not rclpy.ok():
                return
            self.cmd_vel_pub.publish(message)
            rclpy.spin_once(self, timeout_sec=0.0)
            self._spin_sleep(0.05)

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
        deadline = self._now_seconds() + duration
        while rclpy.ok() and self._now_seconds() < deadline:
            self.cmd_vel_pub.publish(message)
            rclpy.spin_once(self, timeout_sec=0.0)
            self._spin_sleep(period)

        self.stop_leader()

    def settle(self, duration: float) -> None:
        self.get_logger().info(f"SETTLE {duration:.1f} s")
        self._spin_sleep(duration)

    def run_cautious(self) -> None:
        """Moderate validation of the balanced-tree controller."""
        self.get_logger().info(
            "=== CAUTIOUS FIVE-ROBOT TREE EXPERIMENT START ==="
        )

        self.publish_formation("tree_nominal")
        self.settle(8.0)

        self.publish_formation("tree_wide")
        self.settle(12.0)

        self.publish_formation("tree_compact")
        self.settle(10.0)

        # Small, clearly visible leader translations while compact.
        self.publish_velocity(0.0, 0.15, 0.0, duration=3.0)  # +0.45 m y
        self.settle(8.0)

        self.publish_velocity(0.10, 0.0, 0.0, duration=3.0)  # +0.30 m x
        self.settle(8.0)

        self.publish_formation("tree_staggered")
        self.settle(10.0)

        self.publish_velocity(0.0, -0.15, 0.0, duration=3.0)
        self.settle(8.0)

        self.publish_formation("tree_nominal")
        self.settle(8.0)

        self.publish_velocity(-0.10, 0.0, 0.0, duration=3.0)
        self.settle(10.0)

        self.stop_leader()
        self.get_logger().info(
            "=== CAUTIOUS FIVE-ROBOT TREE EXPERIMENT COMPLETE ==="
        )

    def run_full(self) -> None:
        """Standard paper profile with substantially larger leader motion."""
        self.get_logger().info(
            "=== FULL FIVE-ROBOT TREE EXPERIMENT START ==="
        )

        self.publish_formation("tree_nominal")
        self.settle(8.0)

        # Show the wide tree while it is centered in the tank.
        self.publish_formation("tree_wide")
        self.settle(12.0)

        # Contract before larger translations to preserve workspace margin.
        self.publish_formation("tree_compact")
        self.settle(10.0)

        # About +1.0 m in y.
        self.publish_velocity(0.0, 0.25, 0.0, duration=4.0)
        self.settle(10.0)

        # About +0.63 m in x.
        self.publish_velocity(0.18, 0.0, 0.0, duration=3.5)
        self.settle(10.0)

        # Return to the centered leader reference while the compact formation
        # still provides ample workspace margin.  Expanding to tree_staggered
        # at the translated reference would place robot 4 at approximately
        # x=0.63 m, outside the conservative reference upper bound of 0.625 m
        # (the 0.675 m conservative wall minus the 0.05 m reference margin).
        self.publish_velocity(-0.18, 0.0, 0.0, duration=3.5)
        self.settle(10.0)

        # Undo y before restoring the larger tree footprint.
        self.publish_velocity(0.0, -0.25, 0.0, duration=4.0)
        self.settle(10.0)

        self.publish_formation("tree_staggered")
        self.settle(12.0)

        self.publish_formation("tree_nominal")
        self.settle(12.0)

        self.stop_leader()
        self.get_logger().info(
            "=== FULL FIVE-ROBOT TREE EXPERIMENT COMPLETE ==="
        )

    def run_challenging(self) -> None:
        """Execute large, collision-aware three-dimensional tree changes.

        The two branches are separated in depth before robots exchange sides.
        The subsequent crossed, opposed, and parallel geometries command
        approximately 2.3--2.7 m displacements for different tree levels.
        All steady references remain inside the conservative tank workspace
        and all edge lengths remain inside the conservative sensing domain.
        """
        self.get_logger().info(
            "=== CHALLENGING FIVE-ROBOT TREE EXPERIMENT START ==="
        )

        self.publish_formation("tree_nominal")
        self.settle(6.0)

        # Establish 0.8--1.3 m depth separation before any lateral crossing.
        self.publish_formation("tree_depth_split")
        self.settle(10.0)

        # The leaves exchange sides by about 2.6 m, producing an X-shaped tree.
        self.publish_formation("tree_crossed_3d")
        self.settle(16.0)

        # The first-level followers now exchange sides by about 2.3 m while
        # the leaves remain near the tank sides, yielding two opposed columns.
        self.publish_formation("tree_opposed_3d")
        self.settle(16.0)

        # All four followers move roughly 2.6 m into two parallel, depth-
        # separated branches near the far end of the tank.
        self.publish_formation("tree_parallel_3d")
        self.settle(18.0)

        # Re-form the crossed X from the opposite side of the workspace.
        self.publish_formation("tree_crossed_3d")
        self.settle(16.0)

        # Restore the uncrossed lateral ordering while retaining depth
        # separation, then close the depth layers only after paths are clear.
        self.publish_formation("tree_depth_split")
        self.settle(14.0)

        self.publish_formation("tree_nominal")
        self.settle(14.0)

        self.stop_leader()
        self.get_logger().info(
            "=== CHALLENGING FIVE-ROBOT TREE EXPERIMENT COMPLETE ==="
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leader", default="itrl_rov_1")
    parser.add_argument(
        "--profile",
        choices=("cautious", "full", "challenging"),
        default="cautious",
    )
    parser.add_argument(
        "--gazebo-timer",
        "--gazebo_timer",
        action="store_true",
        help="time all mission phases from Gazebo /clock instead of wall time",
    )
    args = parser.parse_args()

    rclpy.init()
    node = FiveRobotExperimentRunner(
        args.leader,
        gazebo_timer=args.gazebo_timer,
    )

    try:
        node.wait_for_phase("FORMATION")
        node.wait_for_subscribers()

        if args.profile == "cautious":
            node.run_cautious()
        elif args.profile == "full":
            node.run_full()
        else:
            node.run_challenging()
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
