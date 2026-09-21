#!/usr/bin/env python3
"""Run reproducible five-BlueROV balanced-tree experiments.

Graph:
    4 -> 2 -> 1
    5 -> 3 -> 1

The script may be started before arming. It waits for FORMATION, verifies
all four follower formation-command subscriptions and the leader cmd_vel
subscription, then executes the requested profile.  By default profile timing
uses wall time; pass ``--gazebo-timer`` to make every formation hold, settle
interval, and leader-velocity manoeuvre follow Gazebo /clock instead.

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
import threading
import time

import rclpy
from geometry_msgs.msg import Twist
from rosgraph_msgs.msg import Clock
from rclpy.executors import SingleThreadedExecutor
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


class FiveRobotExperimentRunner(Node):
    def __init__(
        self,
        leader: str,
        expected_followers: int = 4,
        *,
        gazebo_timer: bool = False,
    ) -> None:
        # Keep the runner itself on wall time.  In simulation-time mode we
        # subscribe explicitly to /clock instead of making the node's ROS clock
        # depend on callbacks processed by the mission loop.  A background
        # executor services /clock continuously.
        super().__init__("five_robot_experiment_runner")
        self.expected_followers = int(expected_followers)
        self.gazebo_timer = bool(gazebo_timer)
        self._sim_time_sec: float | None = None

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

        if self.gazebo_timer:
            clock_qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
            )
            self.create_subscription(
                Clock,
                "/clock",
                self._clock_callback,
                clock_qos,
            )

    def publish_mission_status(self, status: str) -> None:
        message = String()
        message.data = status.strip().upper()
        self.mission_status_pub.publish(message)
        self.get_logger().info(f"MISSION_STATUS {message.data}")

    def _phase_callback(self, message: String) -> None:
        self.phase = message.data.strip().upper()

    def _clock_callback(self, message: Clock) -> None:
        self._sim_time_sec = (
            float(message.clock.sec)
            + 1.0e-9 * float(message.clock.nanosec)
        )

    def _time(self) -> float:
        if self.gazebo_timer:
            return 0.0 if self._sim_time_sec is None else self._sim_time_sec
        return time.monotonic()

    def _spin_sleep(self, duration: float) -> None:
        """Wait in the selected mission time base.

        ROS callbacks, including /clock, are serviced continuously by the
        background executor.  This loop therefore only observes the selected
        clock and never drives it itself.
        """
        if duration <= 0.0:
            return

        deadline = self._time() + float(duration)
        while rclpy.ok() and self._time() < deadline:
            time.sleep(0.01)

    def wait_for_clock(self) -> None:
        if not self.gazebo_timer:
            return

        self.get_logger().info("Waiting for nonzero Gazebo /clock.")
        while rclpy.ok() and self._time() <= 0.0:
            time.sleep(0.02)
        if not rclpy.ok():
            raise KeyboardInterrupt
        self.get_logger().info("Gazebo simulation clock available.")

        # One-second sanity check: if Gazebo reports RTF ~= 1, this ratio
        # should also be close to 1.  This makes timing problems immediately
        # visible in the runner log.
        sim_start = self._time()
        wall_start = time.monotonic()
        time.sleep(1.0)
        sim_elapsed = self._time() - sim_start
        wall_elapsed = time.monotonic() - wall_start
        ratio = sim_elapsed / wall_elapsed if wall_elapsed > 0.0 else float("nan")
        self.get_logger().info(
            "Observed /clock advance: "
            f"{sim_elapsed:.3f} sim s in {wall_elapsed:.3f} wall s "
            f"(ratio={ratio:.3f})."
        )

    def wait_for_phase(self, desired: str) -> None:
        desired = desired.upper()
        self.get_logger().info(
            f"Waiting for experiment phase {desired}. "
            "Arm/Offboard all five robots when ready."
        )
        while rclpy.ok() and self.phase != desired:
            time.sleep(0.02)
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
            time.sleep(0.05)

        count = self.formation_pub.get_subscription_count()
        self.get_logger().info(
            f"Formation publisher sees {count} subscriber(s)."
        )

        self.get_logger().info("Waiting for leader cmd_vel subscription.")
        while rclpy.ok() and self.cmd_vel_pub.get_subscription_count() < 1:
            time.sleep(0.05)
        self.get_logger().info("Leader cmd_vel subscriber discovered.")

    def publish_formation(self, name: str) -> None:
        """Publish a formation switch without adding artificial mission delay.

        The topic is reliable and transient-local, so a short burst is enough
        once all follower subscriptions have been discovered.  The previous
        implementation spent 2 s re-publishing every switch, which added more
        than 20 s of dead time to the challenging profile.
        """
        count = self.formation_pub.get_subscription_count()
        if count < self.expected_followers:
            raise RuntimeError(
                "Formation command lost expected subscribers: "
                f"found {count}, expected at least {self.expected_followers}."
            )

        self.get_logger().info(
            f"COMMAND formation={name!r} to {count} subscribers"
        )

        message = String()
        message.data = name
        for _ in range(3):
            self.formation_pub.publish(message)
            time.sleep(0.02)

    def stop_leader(self) -> None:
        message = Twist()
        for _ in range(10):
            if not rclpy.ok():
                return
            self.cmd_vel_pub.publish(message)
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
        deadline = self._time() + duration
        while rclpy.ok() and self._time() < deadline:
            self.cmd_vel_pub.publish(message)
            # Publish in wall time so the leader keeps receiving commands
            # even when the Gazebo real-time factor is low.  The end of the
            # manoeuvre is nevertheless determined by simulation time when
            # --gazebo-timer is enabled.
            time.sleep(period)

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
        """Standard paper profile with workspace-compatible leader motion."""
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

        # Translate the compact tree by about +0.8 m in y.  The previous
        # +1.0 m pulse placed the positive-y branch too close to the physical
        # tank boundary.
        self.publish_velocity(0.0, 0.20, 0.0, duration=4.0)
        self.settle(10.0)

        # About +0.63 m in x.
        self.publish_velocity(0.18, 0.0, 0.0, duration=3.5)
        self.settle(10.0)

        # Return the compact tree toward the tank centre before expanding to
        # the staggered geometry.  This keeps all desired robot positions
        # inside the conservative workspace.
        self.publish_velocity(0.0, -0.20, 0.0, duration=4.0)
        self.settle(10.0)

        self.publish_formation("tree_staggered")
        self.settle(12.0)

        self.publish_formation("tree_nominal")
        self.settle(10.0)

        # Undo x.
        self.publish_velocity(-0.18, 0.0, 0.0, duration=3.5)
        self.settle(12.0)

        self.stop_leader()
        self.get_logger().info(
            "=== FULL FIVE-ROBOT TREE EXPERIMENT COMPLETE ==="
        )

    def run_challenging(
        self,
        settle_scale: float = 2.2,
    ) -> None:
        """Tank-safe aggressive 3-D mission with tunable reconfiguration pace.

        ``settle_scale`` multiplies only the pauses after formation switches
        and leader manoeuvres.  It does *not* scale leader-velocity command
        durations, so the commanded whole-formation displacement and the
        workspace-safety margins are unchanged.

        The base sequence is the fast 52.5 s profile.  The default
        ``settle_scale=2.2`` gives a mission of about 108 s, which leaves the
        vehicles enough time to visibly approach each requested formation
        while preserving continuous, demanding transients.
        """
        if settle_scale <= 0.0:
            raise ValueError("settle_scale must be strictly positive.")

        def pause(seconds: float) -> None:
            self.settle(float(settle_scale) * float(seconds))

        self.get_logger().info(
            "=== CHALLENGING FIVE-ROBOT TREE EXPERIMENT START ==="
        )
        self.get_logger().info(
            f"Challenging settle-time scale: {settle_scale:.3f}"
        )

        # ------------------------------------------------------------------
        # 1) 3-D branch exchange.
        # ------------------------------------------------------------------
        self.publish_formation("tree_nominal")
        pause(2.0)

        # Separate the two branches in depth.
        self.publish_formation("tree_depth_split")
        pause(3.5)

        # Leaves exchange sides while remaining vertically separated.
        self.publish_formation("tree_crossed_3d")
        pause(4.0)

        # Complete the branch-side exchange.
        self.publish_formation("tree_opposed_3d")
        pause(4.0)

        # Reverse the exchange while keeping the manoeuvre dynamic.
        self.publish_formation("tree_crossed_3d")
        pause(3.0)

        self.publish_formation("tree_depth_split")
        pause(3.0)

        # ------------------------------------------------------------------
        # 2) Tank-safe whole-formation 3-D slalom.
        # ------------------------------------------------------------------
        # Translation is performed only after contracting the tree.  These
        # velocity pulses have zero net commanded displacement.
        self.publish_formation("tree_compact")
        pause(3.0)

        self.publish_velocity(0.45, 0.40, -0.15, duration=1.5)
        pause(2.0)

        # Cross through the centre to the opposite side.
        self.publish_velocity(-0.45, -0.40, 0.15, duration=3.0)
        pause(2.0)

        # Return to the centred reference.
        self.publish_velocity(0.45, 0.40, -0.15, duration=1.5)
        pause(4.0)

        # ------------------------------------------------------------------
        # 3) Direct full exchange as the final stress manoeuvre.
        # ------------------------------------------------------------------
        self.publish_formation("tree_depth_split")
        pause(3.0)

        # Directly exchange both branches at separated depths.
        self.publish_formation("tree_opposed_3d")
        pause(5.0)

        self.publish_formation("tree_depth_split")
        pause(3.0)

        self.publish_formation("tree_nominal")
        pause(5.0)

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
        action="store_true",
        help=(
            "Use ROS/Gazebo simulation time for all mission durations. "
            "Launch the five-robot controllers with gazebo_timer:=true as "
            "well. Without this flag, the runner keeps the previous wall-"
            "time behaviour."
        ),
    )
    parser.add_argument(
        "--challenging-settle-scale",
        type=float,
        default=2.2,
        help=(
            "Multiplier applied only to waiting/settling intervals in the "
            "challenging profile. Leader velocity command durations are not "
            "scaled. Default: 2.2. Try 2.0 for a slightly faster video "
            "or 2.5 if the vehicles need more time to settle."
        ),
    )
    args = parser.parse_args()

    rclpy.init()
    node = FiveRobotExperimentRunner(
        args.leader,
        gazebo_timer=args.gazebo_timer,
    )

    # Service phase and /clock subscriptions continuously.  In particular,
    # mission time no longer advances only when the mission code calls
    # spin_once(), which could make a simulation-time mission appear much
    # slower than Gazebo itself.
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor_thread = threading.Thread(
        target=executor.spin,
        name="five_robot_mission_executor",
        daemon=True,
    )
    executor_thread.start()

    node.publish_mission_status("WAITING")

    try:
        node.wait_for_phase("FORMATION")
        node.wait_for_subscribers()
        node.wait_for_clock()

        if args.gazebo_timer:
            node.get_logger().info(
                "Mission durations use explicit Gazebo /clock time."
            )
        else:
            node.get_logger().info("Mission durations use wall time.")

        node.publish_mission_status("RUNNING")
        # Give the split recorder time to open RUN/mission/bag before the
        # first formation or velocity command is sent.
        node._spin_sleep(1.0)

        if args.profile == "cautious":
            node.run_cautious()
        elif args.profile == "full":
            node.run_full()
        else:
            node.run_challenging(
                settle_scale=args.challenging_settle_scale,
            )

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
        executor.shutdown()
        executor_thread.join(timeout=2.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
