"""ROS 2 PX4 offboard heartbeat for thrust-and-torque control."""

import rclpy
from px4_msgs.msg import OffboardControlMode
from rclpy.node import Node

from .px4_interface import px4_qos_profile


class OffboardHeartbeat(Node):
    def __init__(self) -> None:
        super().__init__("offboard_heartbeat_wrench")
        self.declare_parameter(
            "topic",
            "fmu/in/offboard_control_mode",
        )
        topic = str(self.get_parameter("topic").value)
        self.publisher = self.create_publisher(
            OffboardControlMode,
            topic,
            px4_qos_profile(),
        )
        self.create_timer(0.05, self._tick)

    def _tick(self) -> None:
        message = OffboardControlMode()
        message.timestamp = int(
            self.get_clock().now().nanoseconds / 1000
        )
        message.position = False
        message.velocity = False
        message.acceleration = False
        message.attitude = False
        message.body_rate = False
        message.thrust_and_torque = True
        message.direct_actuator = False
        self.publisher.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OffboardHeartbeat()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
