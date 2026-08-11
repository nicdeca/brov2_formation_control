"""Shared ROS 2 node parameter helpers."""

from __future__ import annotations

from rclpy.node import Node

from .core_runtime import CoreControllerConfig
from .frame_conventions import FrameConvention
from .px4_interface import PX4WrenchNormalization


def declare_common_parameters(node: Node) -> None:
    node.declare_parameter("robot_name", "Splash")
    node.declare_parameter("dt", 0.02)
    node.declare_parameter("measurement_timeout", 0.25)
    node.declare_parameter("dry_run", False)
    node.declare_parameter("state_source", "px4")

    # Frame interface for generic nav_msgs/Odometry sources.  These defaults reproduce how the old MoCap workspace
    # interpreted its Odometry messages.
    node.declare_parameter("mocap_world_frame", "core_nwu")
    node.declare_parameter("odom_twist_frame", "body")

    # Core controller.  Wrench space is the natural choice when PX4 performs
    # the downstream allocation.
    node.declare_parameter("control_space", "wrench")
    node.declare_parameter("thruster_voltage", 16)
    node.declare_parameter("thrust_derating", 1.0)
    node.declare_parameter("virtual_linear_speed_limit", 1.5)
    node.declare_parameter("virtual_angular_speed_limit", 2.0)
    node.declare_parameter("slack_linear_penalty", 100.0)
    node.declare_parameter("slack_quadratic_penalty", 5000.0)
    node.declare_parameter("alpha_gain", 0.8)

    # PX4 wrench normalization inherited from the previous ROS 2 workspace.
    node.declare_parameter("px4_force_max", [88.0, 88.0, 137.0])
    node.declare_parameter("px4_torque_max", [30.0, 16.5, 21.0])
    node.declare_parameter("px4_thrust_command_limit", 0.10)
    node.declare_parameter("px4_torque_command_limit", 0.10)


def frame_convention_from_parameters(node: Node) -> FrameConvention:
    return FrameConvention(
        world_frame=str(node.get_parameter("mocap_world_frame").value),
        twist_frame=str(node.get_parameter("odom_twist_frame").value),
    )


def controller_config_from_parameters(node: Node) -> CoreControllerConfig:
    return CoreControllerConfig(
        control_space=str(node.get_parameter("control_space").value),
        thruster_voltage=int(node.get_parameter("thruster_voltage").value),
        thrust_derating=float(node.get_parameter("thrust_derating").value),
        virtual_linear_speed_limit=float(
            node.get_parameter("virtual_linear_speed_limit").value
        ),
        virtual_angular_speed_limit=float(
            node.get_parameter("virtual_angular_speed_limit").value
        ),
        slack_linear_penalty=float(
            node.get_parameter("slack_linear_penalty").value
        ),
        slack_quadratic_penalty=float(
            node.get_parameter("slack_quadratic_penalty").value
        ),
        alpha_gain=float(node.get_parameter("alpha_gain").value),
    )


def px4_normalization_from_parameters(
    node: Node,
) -> PX4WrenchNormalization:
    return PX4WrenchNormalization(
        force_max=tuple(
            float(value)
            for value in node.get_parameter("px4_force_max").value
        ),
        torque_max=tuple(
            float(value)
            for value in node.get_parameter("px4_torque_max").value
        ),
        thrust_command_limit=float(
            node.get_parameter("px4_thrust_command_limit").value
        ),
        torque_command_limit=float(
            node.get_parameter("px4_torque_command_limit").value
        ),
    )
