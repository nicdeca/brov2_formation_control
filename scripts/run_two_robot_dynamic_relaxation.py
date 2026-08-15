#!/usr/bin/env python3
"""Final aggressive dynamic-relaxation experiment.

Unlike the earlier preload experiment, the desired formation is comfortably
inside the conservative sensing domain.  The relaxation event, if triggered,
comes from a fast leader transient plus strongly reduced follower authority.

The stress formation is

    d_des = [0.85, -2.20, 0] m
    ||d_des|| ~= 2.358 m.

The leader pulse is approximately aligned with d_des:

    v_L = [0.45, -1.15, 0] m/s
    ||v_L|| ~= 1.235 m/s
    duration = 0.90 s.

If the follower were stationary and the leader followed the command perfectly,
the added relative displacement would be about 1.11 m, producing a peak range
around 3.47 m.  This is deliberately beyond d_max^c = 3.0 m but still below the
physical d_max = 3.6 m.
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


class DynamicRelaxationRunner(Node):
    def __init__(self, leader: str) -> None:
        super().__init__("dynamic_relaxation_runner")

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
            "Waiting for FORMATION. Arm and put both robots in Offboard."
        )
        while rclpy.ok() and self.phase != "FORMATION":
            rclpy.spin_once(self, timeout_sec=0.2)

        while rclpy.ok() and self.formation_pub.get_subscription_count() < 1:
            rclpy.spin_once(self, timeout_sec=0.2)

        while rclpy.ok() and self.cmd_vel_pub.get_subscription_count() < 1:
            rclpy.spin_once(self, timeout_sec=0.2)

        self.get_logger().info("FORMATION and command subscribers ready.")

    def formation(self, name: str, duration: float = 2.0) -> None:
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
            if not rclpy.ok():
                return
            self.cmd_vel_pub.publish(message)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(0.05)

    def velocity(
        self,
        vx: float,
        vy: float,
        vz: float,
        *,
        duration: float,
        rate_hz: float = 30.0,
    ) -> None:
        self.get_logger().info(
            f"COMMAND leader velocity [{vx:.3f}, {vy:.3f}, {vz:.3f}] "
            f"for {duration:.2f} s"
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
        self.spin_sleep(duration)

    def run(self) -> None:
        self.get_logger().info(
            "=== FINAL DYNAMIC RELAXATION EXPERIMENT START ==="
        )

        # Initial settling, including any startup workspace relaxation caused
        # by positive buoyancy before Offboard.
        self.formation("pair_nominal")
        self.settle(12.0)

        # Move to a moderately long but comfortably conservative formation.
        self.formation("pair_dynamic_stress")
        self.settle(15.0)

        # High-speed pulse away from the derated follower.
        self.velocity(
            0.45,
            -1.15,
            0.0,
            duration=0.90,
        )

        # Give the follower ample time to catch up and the relaxation state to
        # recover toward zero.
        self.settle(22.0)

        # Return to the nominal formation first.
        self.formation("pair_nominal")
        self.settle(12.0)

        # Approximately undo the leader displacement.
        self.velocity(
            -0.45,
            1.15,
            0.0,
            duration=0.90,
        )
        self.settle(12.0)

        self.stop_leader()
        self.get_logger().info(
            "=== FINAL DYNAMIC RELAXATION EXPERIMENT COMPLETE ==="
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leader", default="itrl_rov_1")
    args = parser.parse_args()

    rclpy.init()
    node = DynamicRelaxationRunner(args.leader)

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
