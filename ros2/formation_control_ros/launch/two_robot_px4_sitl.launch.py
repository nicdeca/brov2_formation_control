"""ROS 2 smoke test with current BlueROV SITL PX4 odometry."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    leader = LaunchConfiguration("leader")
    follower = LaunchConfiguration("follower")
    dt = LaunchConfiguration("dt")
    dry_run = LaunchConfiguration("dry_run")

    common = {
        "dt": dt,
        "state_source": "px4",
        "control_space": "thruster",
        "dry_run": dry_run,
        "px4_thrust_command_limit": 0.10,
        "px4_torque_command_limit": 0.10,
    }

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "leader",
                default_value="itrl_rov_1",
            ),
            DeclareLaunchArgument(
                "follower",
                default_value="itrl_rov_2",
            ),
            DeclareLaunchArgument(
                "dt",
                default_value="0.02",
            ),
            DeclareLaunchArgument(
                "dry_run",
                default_value="true",
            ),
            Node(
                package="formation_control_ros",
                executable="leader_controller",
                namespace=leader,
                name="controller",
                output="screen",
                parameters=[
                    common,
                    {
                        "robot_name": leader,
                        "reference_mode": "stationary",
                    },
                ],
            ),
            Node(
                package="formation_control_ros",
                executable="offboard_heartbeat_wrench",
                namespace=leader,
                name="heartbeat",
                output="screen",
            ),
            Node(
                package="formation_control_ros",
                executable="follower_controller",
                namespace=follower,
                name="controller",
                output="screen",
                parameters=[
                    common,
                    {
                        "robot_name": follower,
                        "parent_robot_name": leader,
                        "desired_relative_position": [0.0, -1.80, 0.0],
                        "adaptive": True,
                    },
                ],
            ),
            Node(
                package="formation_control_ros",
                executable="offboard_heartbeat_wrench",
                namespace=follower,
                name="heartbeat",
                output="screen",
            ),
        ]
    )
