"""Start the in-repo three-BlueROV simulator and formation experiment.

State-source arguments are forwarded to ``three_robot_experiment.launch.py``.
PX4 remains the default for SITL.
"""

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    package_share = FindPackageShare("formation_control_ros")

    px4_dir = LaunchConfiguration("px4_dir")
    world = LaunchConfiguration("world")
    dry_run = LaunchConfiguration("dry_run")
    initialization_x = LaunchConfiguration("initialization_x")
    initialization_y = LaunchConfiguration("initialization_y")
    initialization_z = LaunchConfiguration("initialization_z")
    leader_reference_mode = LaunchConfiguration("leader_reference_mode")
    state_source = LaunchConfiguration("state_source")
    state_topic_template = LaunchConfiguration("state_topic_template")
    mocap_world_frame = LaunchConfiguration("mocap_world_frame")
    odom_twist_frame = LaunchConfiguration("odom_twist_frame")

    simulator = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [package_share, "launch", "multi_bluerov2_sim.launch.py"]
            )
        ),
        launch_arguments={
            "px4_dir": px4_dir,
            "world": world,
            "robot_count": "3",
        }.items(),
    )

    experiment = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [package_share, "launch", "three_robot_experiment.launch.py"]
            )
        ),
        launch_arguments={
            "dry_run": dry_run,
            "initialization_x": initialization_x,
            "initialization_y": initialization_y,
            "initialization_z": initialization_z,
            "leader_reference_mode": leader_reference_mode,
            "leader_robot_configuration": "gazebo",
            "follower_left_robot_configuration": "gazebo",
            "follower_right_robot_configuration": "gazebo",
            "state_source": state_source,
            "state_topic_template": state_topic_template,
            "mocap_world_frame": mocap_world_frame,
            "odom_twist_frame": odom_twist_frame,
        }.items(),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "px4_dir",
                default_value=os.environ.get(
                    "PX4_AUTOPILOT_DIR", str(Path.home() / "PX4-Autopilot")
                ),
            ),
            DeclareLaunchArgument(
                "world",
                default_value="kth_marinarium_docking",
            ),
            DeclareLaunchArgument("dry_run", default_value="false"),
            DeclareLaunchArgument(
                "initialization_x",
                default_value="2.500",
            ),
            DeclareLaunchArgument(
                "initialization_y",
                default_value="0.000",
            ),
            DeclareLaunchArgument(
                "initialization_z",
                default_value="-1.45",
                description="Common core-NWU initialization depth [m].",
            ),
            DeclareLaunchArgument(
                "leader_reference_mode",
                default_value="stationary",
            ),
            DeclareLaunchArgument(
                "state_source",
                default_value="px4",
            ),
            DeclareLaunchArgument(
                "state_topic_template",
                default_value="",
            ),
            DeclareLaunchArgument(
                "mocap_world_frame",
                default_value="core_nwu",
            ),
            DeclareLaunchArgument(
                "odom_twist_frame",
                default_value="body",
            ),
            simulator,
            experiment,
        ]
    )
