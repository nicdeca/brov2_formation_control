"""Start the in-repo three-BlueROV simulator and formation experiment."""

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
    leader_reference_mode = LaunchConfiguration("leader_reference_mode")

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
            "leader_reference_mode": leader_reference_mode,
        }.items(),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "px4_dir",
                default_value="/home/nicola/Gits/KTH-PX4/PX4-Autopilot",
            ),
            DeclareLaunchArgument(
                "world",
                default_value="kth_marinarium_docking",
            ),
            DeclareLaunchArgument("dry_run", default_value="false"),
            DeclareLaunchArgument(
                "leader_reference_mode",
                default_value="stationary",
            ),
            simulator,
            experiment,
        ]
    )
