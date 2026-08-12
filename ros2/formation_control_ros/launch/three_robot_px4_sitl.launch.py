"""Three-BlueROV SITL formation experiment.

Robot 1 is the leader. Robots 2 and 3 each observe robot 1. Both followers
subscribe to the same absolute named-formation selector topic, so one String
message switches the complete formation.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    leader = LaunchConfiguration("leader")
    follower_left = LaunchConfiguration("follower_left")
    follower_right = LaunchConfiguration("follower_right")
    dt = LaunchConfiguration("dt")
    dry_run = LaunchConfiguration("dry_run")
    leader_reference_mode = LaunchConfiguration("leader_reference_mode")

    common = {
        "dt": dt,
        "state_source": "px4",
        "control_space": "thruster",
        "dry_run": dry_run,
        "px4_thrust_command_limit": 0.10,
        "px4_torque_command_limit": 0.10,
    }

    formation_names = [
        "triangle_nominal",
        "triangle_wide",
        "triangle_high",
    ]

    left_formations = [
        -1.20, -1.50, 0.00,
        -1.75, -1.80, 0.00,
        -1.20, -1.50, 0.60,
    ]
    right_formations = [
         1.20, -1.50, 0.00,
         1.75, -1.80, 0.00,
         1.20, -1.50, -0.60,
    ]

    return LaunchDescription(
        [
            DeclareLaunchArgument("leader", default_value="itrl_rov_1"),
            DeclareLaunchArgument(
                "follower_left",
                default_value="itrl_rov_2",
            ),
            DeclareLaunchArgument(
                "follower_right",
                default_value="itrl_rov_3",
            ),
            DeclareLaunchArgument("dt", default_value="0.02"),
            DeclareLaunchArgument("dry_run", default_value="true"),
            DeclareLaunchArgument(
                "leader_reference_mode",
                default_value="stationary",
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
                        "reference_mode": leader_reference_mode,
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
                namespace=follower_left,
                name="controller",
                output="screen",
                parameters=[
                    common,
                    {
                        "robot_name": follower_left,
                        "parent_robot_name": leader,
                        "desired_relative_position": [-1.20, -1.50, 0.0],
                        "desired_formation_topic": (
                            "/formation_control/desired_formation"
                        ),
                        "formation_names": formation_names,
                        "formation_relative_positions": left_formations,
                        "adaptive": True,
                    },
                ],
            ),
            Node(
                package="formation_control_ros",
                executable="offboard_heartbeat_wrench",
                namespace=follower_left,
                name="heartbeat",
                output="screen",
            ),

            Node(
                package="formation_control_ros",
                executable="follower_controller",
                namespace=follower_right,
                name="controller",
                output="screen",
                parameters=[
                    common,
                    {
                        "robot_name": follower_right,
                        "parent_robot_name": leader,
                        "desired_relative_position": [1.20, -1.50, 0.0],
                        "desired_formation_topic": (
                            "/formation_control/desired_formation"
                        ),
                        "formation_names": formation_names,
                        "formation_relative_positions": right_formations,
                        "adaptive": True,
                    },
                ],
            ),
            Node(
                package="formation_control_ros",
                executable="offboard_heartbeat_wrench",
                namespace=follower_right,
                name="heartbeat",
                output="screen",
            ),
        ]
    )
