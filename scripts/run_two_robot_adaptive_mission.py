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
adds a tank-safe pool-length translation.  Because the sensing constraints
depend on the relative geometry, the large A/B/C/D reconfigurations remain an
important excitation.  An optional short outward leader pulse can be enabled
during formation D to demonstrate authority-triggered range relaxation while
the desired formation itself remains well inside the conservative range limit.
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

# Deliberately simple, workspace-safe stress mission.
#
# The large B/C reconfigurations used by the old Example-12 sequence are not
# needed here: they create actuator-limited transients before the event we
# actually want to demonstrate, and formation B places the desired follower
# unnecessarily close to the upper / side workspace boundaries.
#
# A is the initial interior formation. D has a 2.20 m desired range and is
# also comfortably inside the tank. The leader stress manoeuvres are applied
# only after D has settled.
FORMATION_EVENTS = (
    (0.0, "adaptive_A"),
    (10.0, "adaptive_D"),
    (50.0, "adaptive_A"),
)

MISSION_DURATION = 60.0

# Unit vector from follower to leader for formation D.  The leader's outward
# stress manoeuvre follows this direction so that it excites the sensing range
# directly, rather than wasting authority tangentially.
FORMATION_D_DIRECTION = (-0.96484625, -0.23072410, -0.12584951)

# Horizontal direction orthogonal to the formation-D line of sight.
# It excites the horizontal image / FoV coordinate strongly while avoiding a
# large vertical excursion toward the water surface or tank bottom.
FORMATION_D_TRANSVERSE = (0.23257321, -0.97257889, 0.00000000)


def velocity_command(
    t: float,
    *,
    range_stress_speed: float | None = None,
    range_stress_start: float = 24.0,
    range_stress_outward_duration: float = 0.8,
    range_stress_hold_duration: float = 0.40,
    range_stress_return_duration: float = 0.8,
    fov_stress_speed: float | None = None,
    fov_stress_start: float = 38.0,
    fov_stress_outward_duration: float = 0.8,
    fov_stress_hold_duration: float = 0.40,
    fov_stress_return_duration: float = 0.8,
) -> tuple[float, float, float]:
    """Aggressive, tank-safe leader manoeuvres for adaptive-domain validation.

    Two distinct zero-net-displacement stress manoeuvres are used while
    formation D is active.

    1. RANGE STRESS:
       a short, fast sprint directly *away from the follower* along the
       formation-D line of sight.  This primarily challenges d_21 and the
       follower's translational control authority.

    2. FOV STRESS:
       a short, fast *horizontal* sprint orthogonal to the line of sight.
       This strongly excites alpha_h while changing range only to second
       order and avoids spending workspace margin in depth.

    Both manoeuvres return along exactly the same path, so they have zero net
    commanded leader displacement. Between the stress events the leader is
    stationary; this makes their effect visually and diagnostically clear.
    """

    def symmetric_pulse(
        *,
        start: float,
        speed: float | None,
        outward_duration: float,
        hold_duration: float,
        return_duration: float,
        direction: tuple[float, float, float],
    ) -> tuple[float, float, float] | None:
        if speed is None:
            return None
        speed = float(speed)
        if speed < 0.0:
            raise ValueError("stress speed must be non-negative")
        if outward_duration <= 0.0 or return_duration <= 0.0:
            raise ValueError("stress motion durations must be positive")
        if hold_duration < 0.0:
            raise ValueError("stress hold duration must be non-negative")

        outward_end = start + outward_duration
        hold_end = outward_end + hold_duration
        return_end = hold_end + return_duration

        ux, uy, uz = direction

        if start <= t < outward_end:
            return speed * ux, speed * uy, speed * uz

        if outward_end <= t < hold_end:
            return 0.0, 0.0, 0.0

        if hold_end <= t < return_end:
            return_speed = speed * outward_duration / return_duration
            return -return_speed * ux, -return_speed * uy, -return_speed * uz

        return None

    # First: direct line-of-sight range / authority stress.
    command = symmetric_pulse(
        start=range_stress_start,
        speed=range_stress_speed,
        outward_duration=range_stress_outward_duration,
        hold_duration=range_stress_hold_duration,
        return_duration=range_stress_return_duration,
        direction=FORMATION_D_DIRECTION,
    )
    if command is not None:
        return command

    # Second: transverse FoV stress.
    command = symmetric_pulse(
        start=fov_stress_start,
        speed=fov_stress_speed,
        outward_duration=fov_stress_outward_duration,
        hold_duration=fov_stress_hold_duration,
        return_duration=fov_stress_return_duration,
        direction=FORMATION_D_TRANSVERSE,
    )
    if command is not None:
        return command

    # Keep the leader fixed outside the deliberately aggressive stress pulses.
    return 0.0, 0.0, 0.0


class AdaptiveMissionRunner(Node):
    def __init__(
        self,
        leader: str,
        *,
        gazebo_timer: bool,
        range_stress_speed: float | None,
        range_stress_start: float,
        range_stress_outward_duration: float,
        range_stress_hold_duration: float,
        range_stress_return_duration: float,
        fov_stress_speed: float | None,
        fov_stress_start: float,
        fov_stress_outward_duration: float,
        fov_stress_hold_duration: float,
        fov_stress_return_duration: float,
    ) -> None:
        super().__init__("two_robot_adaptive_mission_runner")
        self.gazebo_timer = bool(gazebo_timer)
        self.range_stress_speed = range_stress_speed
        self.range_stress_start = float(range_stress_start)
        self.range_stress_outward_duration = float(
            range_stress_outward_duration
        )
        self.range_stress_hold_duration = float(range_stress_hold_duration)
        self.range_stress_return_duration = float(
            range_stress_return_duration
        )
        self.fov_stress_speed = fov_stress_speed
        self.fov_stress_start = float(fov_stress_start)
        self.fov_stress_outward_duration = float(
            fov_stress_outward_duration
        )
        self.fov_stress_hold_duration = float(fov_stress_hold_duration)
        self.fov_stress_return_duration = float(
            fov_stress_return_duration
        )
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

    def sleep_selected_time(self, duration: float) -> None:
        if duration <= 0.0:
            return
        deadline = self._time() + float(duration)
        while rclpy.ok() and self._time() < deadline:
            time.sleep(0.01)

    def wait_until_ready(self) -> None:
        self.get_logger().info(
            "Waiting for FORMATION. Arm/Offboard both SITL robots when ready."
        )
        while rclpy.ok() and self.phase != "FORMATION":
            time.sleep(0.02)
        if not rclpy.ok():
            raise KeyboardInterrupt
        self.get_logger().info("FORMATION observed.")

        while rclpy.ok() and self.formation_pub.get_subscription_count() < 1:
            time.sleep(0.02)
        self.get_logger().info("Follower formation subscriber discovered.")

        while rclpy.ok() and self.cmd_vel_pub.get_subscription_count() < 1:
            time.sleep(0.02)
        self.get_logger().info("Leader cmd_vel subscriber discovered.")

        if self.gazebo_timer:
            self.get_logger().info("Waiting for nonzero Gazebo /clock.")
            while rclpy.ok() and self._time() <= 0.0:
                time.sleep(0.02)
            if not rclpy.ok():
                raise KeyboardInterrupt
            self.get_logger().info("Gazebo simulation clock available.")

            sim_start = self._time()
            wall_start = time.monotonic()
            time.sleep(1.0)
            sim_elapsed = self._time() - sim_start
            wall_elapsed = time.monotonic() - wall_start
            ratio = (
                sim_elapsed / wall_elapsed
                if wall_elapsed > 0.0
                else float("nan")
            )
            self.get_logger().info(
                "Observed /clock advance: "
                f"{sim_elapsed:.3f} sim s in {wall_elapsed:.3f} wall s "
                f"(ratio={ratio:.3f})."
            )

    def publish_formation(self, name: str) -> None:
        self.get_logger().info(f"COMMAND formation={name!r}")
        message = String()
        message.data = name

        # Reliable + transient-local is already sufficient, but a few
        # back-to-back publications make the mission robust to a callback
        # coinciding with the exact switch instant without delaying cmd_vel.
        for _ in range(3):
            self.formation_pub.publish(message)

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
            elapsed = self._time() - start

            while (
                event_index < len(FORMATION_EVENTS)
                and elapsed >= FORMATION_EVENTS[event_index][0]
            ):
                _, name = FORMATION_EVENTS[event_index]
                self.publish_formation(name)
                event_index += 1

            velocity = velocity_command(
                elapsed,
                range_stress_speed=self.range_stress_speed,
                range_stress_start=self.range_stress_start,
                range_stress_outward_duration=(
                    self.range_stress_outward_duration
                ),
                range_stress_hold_duration=self.range_stress_hold_duration,
                range_stress_return_duration=self.range_stress_return_duration,
                fov_stress_speed=self.fov_stress_speed,
                fov_stress_start=self.fov_stress_start,
                fov_stress_outward_duration=(
                    self.fov_stress_outward_duration
                ),
                fov_stress_hold_duration=self.fov_stress_hold_duration,
                fov_stress_return_duration=self.fov_stress_return_duration,
            )
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
    parser.add_argument(
        "--range-stress-speed",
        type=float,
        default=None,
        help=(
            "Peak leader speed [m/s] for the radial authority-stress "
            "manoeuvre. A good Gazebo starting value is 1.2."
        ),
    )
    parser.add_argument(
        "--range-stress-start",
        type=float,
        default=24.0,
        help=(
            "Mission time [s] at which the fast outward range sprint starts. "
            "Default 24 s leaves 14 s after switching to safe formation D."
        ),
    )
    parser.add_argument(
        "--range-stress-outward-duration",
        type=float,
        default=0.8,
        help="Duration [s] of the outward radial sprint.",
    )
    parser.add_argument(
        "--range-stress-hold-duration",
        type=float,
        default=0.4,
        help="Zero-velocity pause [s] after the outward sprint.",
    )
    parser.add_argument(
        "--range-stress-return-duration",
        type=float,
        default=0.8,
        help=(
            "Duration [s] of the return sprint. The return speed is chosen "
            "so the stress manoeuvre has zero net commanded displacement."
        ),
    )
    parser.add_argument(
        "--fov-stress-speed",
        type=float,
        default=None,
        help=(
            "Peak leader speed [m/s] for the 3-D transverse FoV stress "
            "manoeuvre. Start with 1.2."
        ),
    )
    parser.add_argument(
        "--fov-stress-start",
        type=float,
        default=38.0,
        help="Mission time [s] at which the horizontal transverse FoV sprint starts.",
    )
    parser.add_argument(
        "--fov-stress-outward-duration",
        type=float,
        default=0.8,
        help="Duration [s] of the first transverse FoV sprint.",
    )
    parser.add_argument(
        "--fov-stress-hold-duration",
        type=float,
        default=0.4,
        help="Zero-velocity pause [s] at maximum transverse excursion.",
    )
    parser.add_argument(
        "--fov-stress-return-duration",
        type=float,
        default=0.8,
        help="Duration [s] of the transverse return sprint.",
    )
    parser.add_argument(
        "--pre-mission-settle",
        type=float,
        default=1.0,
        help=(
            "Short settling interval [s] after FORMATION and before "
            "MISSION_STATUS=RUNNING. Default: 1 s."
        ),
    )
    args = parser.parse_args()

    rclpy.init()
    node = AdaptiveMissionRunner(
        args.leader,
        gazebo_timer=args.gazebo_timer,
        range_stress_speed=args.range_stress_speed,
        range_stress_start=args.range_stress_start,
        range_stress_outward_duration=args.range_stress_outward_duration,
        range_stress_hold_duration=args.range_stress_hold_duration,
        range_stress_return_duration=args.range_stress_return_duration,
        fov_stress_speed=args.fov_stress_speed,
        fov_stress_start=args.fov_stress_start,
        fov_stress_outward_duration=args.fov_stress_outward_duration,
        fov_stress_hold_duration=args.fov_stress_hold_duration,
        fov_stress_return_duration=args.fov_stress_return_duration,
    )

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor_thread = threading.Thread(
        target=executor.spin,
        name="two_robot_mission_executor",
        daemon=True,
    )
    executor_thread.start()

    node.publish_mission_status("WAITING")

    try:
        node.wait_until_ready()

        if args.pre_mission_settle > 0.0:
            node.get_logger().info(
                "Pre-mission settling for "
                f"{args.pre_mission_settle:.1f} s before RUNNING. "
                "Startup alignment / sampled-data transients remain in the "
                "initialization recording."
            )
            node.sleep_selected_time(args.pre_mission_settle)

        node.publish_mission_status("RUNNING")
        # Give the split recorder time to open RUN/mission/bag before the
        # first formation or velocity command is sent.
        node.sleep_selected_time(1.0)
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
        executor.shutdown()
        executor_thread.join(timeout=2.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
