#!/usr/bin/env python3
"""Run the actuation-limited sensing-domain relaxation paper experiment.

The key event is a short, aggressive leader pulse from a formation only about
5.6 cm inside d_max^c. The leader has full thrust authority; the follower is
deliberately strongly derated.

The default pulse has speed about 0.70 m/s and lasts 0.80 s. If the follower
were stationary and the leader tracked the command perfectly, the additional
separation would be about 0.56 m. Starting from the 2.944 m preload, that gives
about 3.50 m: clearly outside the 3.0 m conservative range but still inside the
3.6 m physical range.
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


class ActuationLimitedRelaxationRunner(Node):
    def __init__(self, leader: str) -> None:
        super().__init__("actuation_limited_relaxation_runner")

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
        rate_hz: float = 20.0,
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
            "=== ACTUATION-LIMITED RELAXATION EXPERIMENT START ==="
        )

        # Let both formation and any startup workspace relaxation recover.
        self.formation("pair_nominal")
        self.settle(12.0)

        # Approach d_max^c = 3.0 m very closely while remaining strictly
        # feasible in the unrelaxed sensing domain.
        self.formation("pair_range_preload")
        self.settle(15.0)

        # Aggressive stress pulse, approximately aligned with
        # d_des = [1.05, -2.75, 0].
        #
        # Unit direction is approximately [0.357, -0.934, 0].
        # Commanded speed norm ~= 0.70 m/s.
        # Over 0.80 s, ideal leader displacement norm ~= 0.56 m.
        self.velocity(
            0.25,
            -0.65,
            0.0,
            duration=0.80,
        )

        # The follower should catch up; range relaxation should then recover.
        self.settle(20.0)

        # Restore nominal formation before undoing leader displacement.
        self.formation("pair_nominal")
        self.settle(12.0)

        self.velocity(
            -0.25,
            0.65,
            0.0,
            duration=0.80,
        )
        self.settle(12.0)

        self.stop_leader()
        self.get_logger().info(
            "=== ACTUATION-LIMITED RELAXATION EXPERIMENT COMPLETE ==="
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leader", default="itrl_rov_1")
    args = parser.parse_args()

    rclpy.init()
    node = ActuationLimitedRelaxationRunner(args.leader)

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
