#!/usr/bin/env python3
"""Keyboard teleoperation for the formation leader velocity reference.

This node publishes world-frame translational velocity commands as
``geometry_msgs/Twist``.  The leader controller's velocity-reference mode
integrates them into a smooth p/v/a reference and retains the existing CLF-QP.

Keys:
    w/s : +x / -x
    a/d : +y / -y
    r/f : +z / -z
    space: stop
    q   : quit

Hold a key (terminal key-repeat) for continuous motion.  The leader controller
has its own deadman timeout, so stale commands automatically become zero.
"""

from __future__ import annotations

import argparse
import select
import sys
import termios
import tty

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


HELP = """
Leader keyboard teleop
----------------------
w/s : +x / -x
a/d : +y / -y
r/f : +z / -z
space: stop
q   : quit
"""


class KeyboardTeleop(Node):
    def __init__(self, topic: str, xy_speed: float, z_speed: float) -> None:
        super().__init__("formation_leader_keyboard_teleop")
        self.publisher = self.create_publisher(Twist, topic, 10)
        self.xy_speed = float(xy_speed)
        self.z_speed = float(z_speed)

    def publish_key(self, key: str) -> bool:
        message = Twist()

        if key == "w":
            message.linear.x = self.xy_speed
        elif key == "s":
            message.linear.x = -self.xy_speed
        elif key == "a":
            message.linear.y = self.xy_speed
        elif key == "d":
            message.linear.y = -self.xy_speed
        elif key == "r":
            message.linear.z = self.z_speed
        elif key == "f":
            message.linear.z = -self.z_speed
        elif key == " ":
            pass
        elif key == "q":
            self.publisher.publish(message)
            return False
        else:
            return True

        self.publisher.publish(message)
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", default="itrl_rov_1")
    parser.add_argument("--xy-speed", type=float, default=0.35)
    parser.add_argument("--z-speed", type=float, default=0.20)
    args = parser.parse_args()

    if args.xy_speed <= 0.0 or args.z_speed <= 0.0:
        raise SystemExit("teleoperation speeds must be positive")

    topic = f"/{args.robot}/formation_control/cmd_vel"

    rclpy.init()
    node = KeyboardTeleop(topic, args.xy_speed, args.z_speed)

    print(HELP)
    print(f"Publishing to {topic}")
    print(
        "The leader-side deadman timeout stops reference motion when "
        "commands become stale."
    )

    settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        running = True
        while rclpy.ok() and running:
            readable, _, _ = select.select([sys.stdin], [], [], 0.05)
            if readable:
                key = sys.stdin.read(1)
                running = node.publish_key(key)
            rclpy.spin_once(node, timeout_sec=0.0)
    finally:
        stop = Twist()
        node.publisher.publish(stop)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
