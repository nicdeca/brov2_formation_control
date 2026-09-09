"""Two-robot SITL through lab-like simulated MoCap and the EKF."""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def _perform(context, name: str) -> str:
    return LaunchConfiguration(name).perform(context)


def _setup(context):
    package_share = FindPackageShare("formation_control_ros")
    leader = _perform(context, "leader")
    follower = _perform(context, "follower")
    robots = f"{leader},{follower}"

    simulated_mocap = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [package_share, "launch", "simulated_mocap.launch.py"]
            )
        ),
        launch_arguments={
            "robots": robots,
            "input_topic_template": "/{robot}/fmu/out/vehicle_odometry",
            "output_pose_topic_template": "/mocap/{robot}/pose",
            "output_imu_topic_template": "/mocap/{robot}/imu",
            "pose_frame_id": "mocap_ned",
            "imu_frame_id_template": "{robot}/base_link_frd",
            "measurement_mode": _perform(
                context, "mocap_measurement_mode"
            ),
            "dropout_start_sec": _perform(
                context, "mocap_dropout_start_sec"
            ),
            "dropout_period_sec": _perform(
                context, "mocap_dropout_period_sec"
            ),
            "dropout_duration_sec": _perform(
                context, "mocap_dropout_duration_sec"
            ),
        }.items(),
    )

    experiment = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    package_share,
                    "launch",
                    "two_robot_experiment_with_ekf.launch.py",
                ]
            )
        ),
        launch_arguments={
            "leader": leader,
            "follower": follower,
            "dry_run": _perform(context, "dry_run"),
            "initialization_z": _perform(context, "initialization_z"),
            "leader_reference_mode": _perform(
                context, "leader_reference_mode"
            ),
            "leader_robot_configuration": "gazebo",
            "follower_robot_configuration": "gazebo",
            "workspace_barrier_enabled": _perform(
                context, "workspace_barrier_enabled"
            ),
            "workspace_adaptive": _perform(
                context, "workspace_adaptive"
            ),
            "pose_topic_template": "/mocap/{robot}/pose",
            "core_pose_topic_template": "/mocap/{robot}/pose_core",
            "odom_topic_template": "/mocap/{robot}/odom_ekf",
            "imu_topic_template": "/mocap/{robot}/imu",
            "input_world_frame": "ned",
            "input_body_frame": "frd",
            "world_to_core_translation": "0,0,0",
            "use_imu_gyro": _perform(context, "use_imu_gyro"),
            "imu_body_frame": "frd",
            "publish_tf": _perform(context, "publish_tf"),
            "orientation_measurement_gain": "1.0",
            "orientation_std": "0.005",
            "gyro_time_constant_sec": "0.01",
            "gyro_std": "0.01",
            "max_coast_sec": "0.0",
        }.items(),
    )

    return [simulated_mocap, experiment]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("leader", default_value="itrl_rov_1"),
            DeclareLaunchArgument("follower", default_value="itrl_rov_2"),
            DeclareLaunchArgument("dry_run", default_value="true"),
            DeclareLaunchArgument(
                "initialization_z",
                default_value="-1.45",
                description="Common core-NWU initialization depth [m].",
            ),
            DeclareLaunchArgument(
                "leader_reference_mode",
                default_value="velocity",
            ),
            DeclareLaunchArgument(
                "workspace_barrier_enabled",
                default_value="true",
            ),
            DeclareLaunchArgument(
                "workspace_adaptive",
                default_value="true",
            ),
            DeclareLaunchArgument(
                "publish_tf",
                default_value="false",
            ),
            DeclareLaunchArgument(
                "use_imu_gyro",
                default_value="true",
                description=(
                    "Use the simulated FRD gyro when fresh. Set false for "
                    "the no-IMU estimator test."
                ),
            ),
            DeclareLaunchArgument(
                "mocap_measurement_mode",
                default_value="ideal",
                description="ideal | intermittent",
            ),
            DeclareLaunchArgument(
                "mocap_dropout_start_sec", default_value="5.0"
            ),
            DeclareLaunchArgument(
                "mocap_dropout_period_sec", default_value="10.0"
            ),
            DeclareLaunchArgument(
                "mocap_dropout_duration_sec", default_value="2.0"
            ),
            OpaqueFunction(function=_setup),
        ]
    )
