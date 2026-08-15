#!/usr/bin/env python3
"""Run the two-robot sensing-domain relaxation paper experiment.

Sequence
--------
1. Settle in the nominal formation.
2. Move to a formation close to the conservative maximum-distance boundary.
3. Apply a short leader pulse approximately along the desired relative vector.
4. Hold and allow the adaptive range-domain state to relax/recover.
5. Return to the nominal formation.
6. Return the leader approximately to its original position.

Expected qualitative behavior
-----------------------------
During step 3 the actual distance should transiently exceed 3.0 m while staying
below the physical 3.6 m limit.  The range-relaxation channel should become
nonzero and subsequently return toward zero.
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


class DomainRelaxationRunner(Node):
    def __init__(self, leader: str) -> None:
        super().__init__("domain_relaxation_runner")

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
            "Waiting for FORMATION. Arm and switch both robots to Offboard."
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
        rate_hz: float = 20.0,
    ) -> None:
        self.get_logger().info(
            f"COMMAND leader velocity [{vx:.3f}, {vy:.3f}, {vz:.3f}] "
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
        self.spin_sleep(duration)

    def run(self) -> None:
        self.get_logger().info(
            "=== SENSING-DOMAIN RELAXATION EXPERIMENT START ==="
        )

        # 1) Baseline.
        self.formation("pair_nominal")
        self.settle(8.0)

        # 2) Preload close to d_max^c = 3.0 m while staying strictly inside it.
        self.formation("pair_range_preload")
        self.settle(14.0)

        # 3) Pulse approximately along d_des = [1.05, -2.70, 0].
        #    This increases parent-minus-follower distance faster than the
        #    follower can initially compensate.
        #
        #    Velocity norm ~= 0.39 m/s, displacement ~= [0.30, -0.78] m.
        self.velocity(
            0.15,
            -0.39,
            0.0,
            duration=2.0,
        )

        # 4) Hold: range relaxation should recover as the follower catches up.
        self.settle(18.0)

        # 5) Return to the nominal sensing geometry.
        self.formation("pair_nominal")
        self.settle(12.0)

        # 6) Approximately undo the leader displacement, now in the spacious
        #    nominal formation.
        self.velocity(
            -0.15,
            0.39,
            0.0,
            duration=2.0,
        )
        self.settle(12.0)

        self.stop_leader()
        self.get_logger().info(
            "=== SENSING-DOMAIN RELAXATION EXPERIMENT COMPLETE ==="
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leader", default="itrl_rov_1")
    args = parser.parse_args()

    rclpy.init()
    node = DomainRelaxationRunner(args.leader)

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
