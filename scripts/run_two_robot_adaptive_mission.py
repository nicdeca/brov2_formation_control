#!/usr/bin/env python3
"""Run the two-BlueROV adaptive-domain SITL paper mission.

The formation-switch times reproduce Example 12 exactly:

    t =  0 s : adaptive_A
    t = 18 s : adaptive_B
    t = 38 s : adaptive_C
    t = 60 s : adaptive_D
    t = 80 s : adaptive_A
    t = 100 s: end

The pure-Python example's absolute leader trajectory cannot be copied literally
into the Marinarium world: integrating its velocity commands would move the
leader by several metres in x and leave the configured pool workspace.
Instead, this SITL runner preserves the same formation-switch schedule and
adds a tank-safe y translation.  Because the sensing constraints depend on the
relative geometry, the large A/B/C/D reconfigurations remain the principal
source of range/FoV relaxation while the moving leader still exercises the
second-order tracking controller.
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
MISSION_STATUS_TOPIC = "/formation_control/mission_status"

FORMATION_EVENTS = (
    (0.0, "adaptive_A"),
    (18.0, "adaptive_B"),
    (38.0, "adaptive_C"),
    (60.0, "adaptive_D"),
    (80.0, "adaptive_A"),
)

MISSION_DURATION = 100.0


def velocity_command(t: float) -> tuple[float, float, float]:
    """Tank-safe moving-leader counterpart of the pure-Python mission.

    The translated mission now uses the pool-length x direction. The two +x
    legs each move the leader by about 1.2 m and the following -x legs
    return it to the initial x position. This is the rigidly rotated
    counterpart of the previously validated tank-safe motion.
    """
    if t < 12.0:
        return 0.0, 0.0, 0.0
    if t < 32.0:
        return 0.060, 0.0, 0.0
    if t < 52.0:
        return -0.060, 0.0, 0.0
    if t < 72.0:
        return 0.060, 0.0, 0.0
    if t < 88.0:
        return -0.075, 0.0, 0.0
    return 0.0, 0.0, 0.0


class AdaptiveMissionRunner(Node):
    def __init__(self, leader: str, *, gazebo_timer: bool) -> None:
        super().__init__(
            "two_robot_adaptive_mission_runner",
            parameter_overrides=[
                Parameter(
                    "use_sim_time",
                    Parameter.Type.BOOL,
                    gazebo_timer,
                )
            ],
        )
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
        self.mission_status_pub = self.create_publisher(
            String,
            MISSION_STATUS_TOPIC,
            phase_qos,
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

    def publish_mission_status(self, status: str) -> None:
        message = String()
        message.data = status.strip().upper()
        self.mission_status_pub.publish(message)
        self.get_logger().info(f"MISSION_STATUS {message.data}")

    def _phase_callback(self, message: String) -> None:
        self.phase = message.data.strip().upper()

    def _time(self) -> float:
        if self.gazebo_timer:
            return 1e-9 * float(self.get_clock().now().nanoseconds)
        return time.monotonic()

    def wait_until_ready(self) -> None:
        self.get_logger().info(
            "Waiting for FORMATION. Arm/Offboard both SITL robots when ready."
        )
        while rclpy.ok() and self.phase != "FORMATION":
            rclpy.spin_once(self, timeout_sec=0.2)
        if not rclpy.ok():
            raise KeyboardInterrupt
        self.get_logger().info("FORMATION observed.")

        while rclpy.ok() and self.formation_pub.get_subscription_count() < 1:
            rclpy.spin_once(self, timeout_sec=0.2)
        self.get_logger().info("Follower formation subscriber discovered.")

        while rclpy.ok() and self.cmd_vel_pub.get_subscription_count() < 1:
            rclpy.spin_once(self, timeout_sec=0.2)
        self.get_logger().info("Leader cmd_vel subscriber discovered.")

        if self.gazebo_timer:
            self.get_logger().info("Waiting for nonzero Gazebo /clock.")
            while rclpy.ok() and self._time() <= 0.0:
                rclpy.spin_once(self, timeout_sec=0.1)
            if not rclpy.ok():
                raise KeyboardInterrupt
            self.get_logger().info("Gazebo simulation clock available.")

    def publish_formation(self, name: str) -> None:
        self.get_logger().info(f"COMMAND formation={name!r}")
        message = String()
        message.data = name

        # Reliable + transient-local is already sufficient, but a few
        # back-to-back publications make the mission robust to a callback
        # coinciding with the exact switch instant without delaying cmd_vel.
        for _ in range(3):
            self.formation_pub.publish(message)
            rclpy.spin_once(self, timeout_sec=0.0)

    def publish_velocity(self, velocity: tuple[float, float, float]) -> None:
        message = Twist()
        message.linear.x = float(velocity[0])
        message.linear.y = float(velocity[1])
        message.linear.z = float(velocity[2])
        self.cmd_vel_pub.publish(message)

    def stop_leader(self) -> None:
        for _ in range(10):
            if not rclpy.ok():
                return
            self.publish_velocity((0.0, 0.0, 0.0))
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(0.05)

    def run(self) -> None:
        self.get_logger().info(
            "=== TWO-ROBOT ADAPTIVE SITL MISSION START ==="
        )
        if self.gazebo_timer:
            self.get_logger().info("Mission timeline uses Gazebo simulation time.")
        else:
            self.get_logger().info("Mission timeline uses wall time.")

        event_index = 0
        previous_velocity: tuple[float, float, float] | None = None
        start = self._time()
        wall_period = 0.05  # publish cmd_vel at about 20 Hz wall-clock rate

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.0)
            elapsed = self._time() - start

            while (
                event_index < len(FORMATION_EVENTS)
                and elapsed >= FORMATION_EVENTS[event_index][0]
            ):
                _, name = FORMATION_EVENTS[event_index]
                self.publish_formation(name)
                event_index += 1

            velocity = velocity_command(elapsed)
            if velocity != previous_velocity:
                self.get_logger().info(
                    "COMMAND leader velocity "
                    f"[{velocity[0]:.3f}, {velocity[1]:.3f}, "
                    f"{velocity[2]:.3f}] m/s at t={elapsed:.2f} s"
                )
                previous_velocity = velocity
            self.publish_velocity(velocity)

            if elapsed >= MISSION_DURATION:
                break

            # Deliberately sleep in wall time even for a simulation-time
            # mission.  If Gazebo pauses or runs slowly, elapsed simulation
            # time stops/slows while ROS callbacks and velocity publication
            # continue.
            time.sleep(wall_period)

        self.stop_leader()
        self.get_logger().info(
            "=== TWO-ROBOT ADAPTIVE SITL MISSION COMPLETE ==="
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leader", default="itrl_rov_1")
    parser.add_argument(
        "--gazebo-timer",
        action="store_true",
        help=(
            "Use ROS/Gazebo simulation time for the 100 s mission timeline. "
            "Launch the controllers with gazebo_timer:=true as well."
        ),
    )
    args = parser.parse_args()

    rclpy.init()
    node = AdaptiveMissionRunner(
        args.leader,
        gazebo_timer=args.gazebo_timer,
    )
    node.publish_mission_status("WAITING")

    try:
        node.wait_until_ready()
        node.publish_mission_status("RUNNING")
        # Give the split recorder time to open RUN/mission/bag before the
        # first formation or velocity command is sent.
        node.spin_sleep(1.0)
        node.run()
        node.publish_mission_status("COMPLETE")
    except KeyboardInterrupt:
        node.publish_mission_status("ABORTED")
        node.get_logger().warn(
            "Mission interrupted; commanding zero leader velocity."
        )
        node.stop_leader()
    except Exception as error:
        node.publish_mission_status("ABORTED")
        node.get_logger().error(f"Mission aborted: {error}")
        node.stop_leader()
        raise
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
