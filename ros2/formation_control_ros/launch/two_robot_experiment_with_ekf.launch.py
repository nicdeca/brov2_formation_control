"""Two-robot hardware experiment using the revised MoCap estimator.

The estimator converts the real MoCap input to the fixed controller-facing
contract core-NWU / body-FLU before publishing /mocap/<robot>/odom_ekf.
"""

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


_EKF_FORWARD_ARGS = (
    "pose_topic_template",
    "core_pose_topic_template",
    "odom_topic_template",
    "imu_topic_template",
    "publish_rate_hz",
    "publish_tf",
    "input_world_frame",
    "input_body_frame",
    "world_to_core_translation",
    "world_to_core_quaternion_xyzw",
    "body_flu_to_input_quaternion_xyzw",
    "body_origin_offset_input_body",
    "use_imu_gyro",
    "imu_body_frame",
    "body_flu_to_imu_quaternion_xyzw",
    "gyro_timeout_sec",
    "gyro_time_constant_sec",
    "gyro_std",
    "max_gyro_abs_rad_s",
    "position_std",
    "linear_accel_std",
    "initial_velocity_std",
    "orientation_std",
    "orientation_measurement_gain",
    "mocap_angular_velocity_time_constant_sec",
    "max_position_innovation_m",
    "max_orientation_innovation_rad",
    "max_body_z_axis_angle_rad",
    "max_coast_sec",
    "max_rejected_samples",
    "status_period_sec",
)


def _setup(context):
    package_share = FindPackageShare("formation_control_ros")
    leader = _perform(context, "leader")
    follower = _perform(context, "follower")
    odom_topic_template = _perform(context, "odom_topic_template")

    ekf_arguments = {
        name: _perform(context, name)
        for name in _EKF_FORWARD_ARGS
    }
    ekf_arguments.update(
        {
            "robots": f"{leader},{follower}",
            "parent_frame": "core_nwu",
        }
    )

    ekf = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [package_share, "launch", "mocap_odom_ekf.launch.py"]
            )
        ),
        launch_arguments=ekf_arguments.items(),
    )

    experiment = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [package_share, "launch", "two_robot_experiment.launch.py"]
            )
        ),
        launch_arguments={
            "leader": leader,
            "follower": follower,
            "dt": _perform(context, "dt"),
            "dry_run": _perform(context, "dry_run"),
            "initialization_z": _perform(context, "initialization_z"),
            "leader_reference_mode": _perform(
                context, "leader_reference_mode"
            ),
            "leader_robot_configuration": _perform(
                context, "leader_robot_configuration"
            ),
            "follower_robot_configuration": _perform(
                context, "follower_robot_configuration"
            ),
            "position_gain": _perform(context, "position_gain"),
            "formation_gain": _perform(context, "formation_gain"),
            "virtual_linear_gain": _perform(
                context, "virtual_linear_gain"
            ),
            "virtual_angular_gain": _perform(
                context, "virtual_angular_gain"
            ),
            "command_filter_linear_bandwidth": _perform(
                context, "command_filter_linear_bandwidth"
            ),
            "command_filter_angular_bandwidth": _perform(
                context, "command_filter_angular_bandwidth"
            ),
            "alpha_gain": _perform(context, "alpha_gain"),
            "workspace_barrier_enabled": _perform(
                context, "workspace_barrier_enabled"
            ),
            "workspace_adaptive": _perform(
                context, "workspace_adaptive"
            ),
            "state_source": "nav_msgs",
            "state_topic_template": odom_topic_template,
            # EKF output is unconditionally core-NWU / body-FLU.
            "mocap_world_frame": "core_nwu",
            "odom_twist_frame": "body",
        }.items(),
    )
    return [ekf, experiment]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("leader", default_value="itrl_rov_1"),
            DeclareLaunchArgument("follower", default_value="itrl_rov_2"),
            DeclareLaunchArgument("dt", default_value="0.02"),
            DeclareLaunchArgument("dry_run", default_value="true"),
            DeclareLaunchArgument(
                "initialization_z",
                default_value="-1.45",
                description="Common core-NWU wet-test initialization depth [m].",
            ),
            DeclareLaunchArgument(
                "leader_reference_mode",
                default_value="stationary",
            ),
            DeclareLaunchArgument(
                "leader_robot_configuration",
                default_value="auto",
            ),
            DeclareLaunchArgument(
                "follower_robot_configuration",
                default_value="auto",
            ),
            DeclareLaunchArgument("position_gain", default_value="2.0"),
            DeclareLaunchArgument("formation_gain", default_value="2.0"),
            DeclareLaunchArgument(
                "virtual_linear_gain",
                default_value="0.55",
            ),
            DeclareLaunchArgument(
                "virtual_angular_gain",
                default_value="0.80",
            ),
            DeclareLaunchArgument(
                "command_filter_linear_bandwidth",
                default_value="3.0",
            ),
            DeclareLaunchArgument(
                "command_filter_angular_bandwidth",
                default_value="4.0",
            ),
            DeclareLaunchArgument("alpha_gain", default_value="1.8"),
            DeclareLaunchArgument(
                "workspace_barrier_enabled",
                default_value="true",
            ),
            DeclareLaunchArgument(
                "workspace_adaptive",
                default_value="true",
            ),
            DeclareLaunchArgument(
                "pose_topic_template",
                default_value="/mocap/{robot}/pose",
            ),
            DeclareLaunchArgument(
                "core_pose_topic_template",
                default_value="/mocap/{robot}/pose_core",
            ),
            DeclareLaunchArgument(
                "odom_topic_template",
                default_value="/mocap/{robot}/odom_ekf",
            ),
            DeclareLaunchArgument(
                "imu_topic_template",
                default_value="/{robot}/mavros/imu/data",
            ),
            DeclareLaunchArgument(
                "publish_rate_hz",
                default_value="80.0",
            ),
            DeclareLaunchArgument(
                "publish_tf",
                default_value="false",
            ),
            # Laboratory raw MoCap convention established in the pool:
            # NED world and FRD rigid body. The estimator converts this to the
            # controller-facing core NWU / FLU contract.
            DeclareLaunchArgument(
                "input_world_frame",
                default_value="ned",
            ),
            DeclareLaunchArgument(
                "input_body_frame",
                default_value="frd",
            ),
            DeclareLaunchArgument(
                "world_to_core_translation",
                default_value="0,0,0",
            ),
            DeclareLaunchArgument(
                "world_to_core_quaternion_xyzw",
                default_value="0,0,0,1",
            ),
            DeclareLaunchArgument(
                "body_flu_to_input_quaternion_xyzw",
                default_value="0,0,0,1",
            ),
            DeclareLaunchArgument(
                "body_origin_offset_input_body",
                default_value="0,0,0",
            ),
            # Real robot gyro.
            DeclareLaunchArgument(
                "use_imu_gyro",
                default_value="false",
                description=(
                    "Hardware default: no MAVROS gyro is currently available. "
                    "The estimator uses MoCap attitude finite differences."
                ),
            ),
            DeclareLaunchArgument(
                "imu_body_frame",
                default_value="flu",
            ),
            DeclareLaunchArgument(
                "body_flu_to_imu_quaternion_xyzw",
                default_value="0,0,0,1",
            ),
            DeclareLaunchArgument("gyro_timeout_sec", default_value="0.20"),
            DeclareLaunchArgument(
                "gyro_time_constant_sec",
                default_value="0.03",
            ),
            DeclareLaunchArgument("gyro_std", default_value="0.03"),
            DeclareLaunchArgument(
                "max_gyro_abs_rad_s",
                default_value="5.0",
            ),
            DeclareLaunchArgument("position_std", default_value="0.01"),
            DeclareLaunchArgument(
                "linear_accel_std",
                default_value="0.7",
            ),
            DeclareLaunchArgument(
                "initial_velocity_std",
                default_value="0.5",
            ),
            DeclareLaunchArgument(
                "orientation_std",
                default_value="0.015",
            ),
            DeclareLaunchArgument(
                "orientation_measurement_gain",
                default_value="1.0",
            ),
            DeclareLaunchArgument(
                "mocap_angular_velocity_time_constant_sec",
                default_value="0.05",
            ),
            DeclareLaunchArgument(
                "max_position_innovation_m",
                default_value="0.50",
            ),
            DeclareLaunchArgument(
                "max_orientation_innovation_rad",
                default_value="1.20",
            ),
            DeclareLaunchArgument(
                "max_body_z_axis_angle_rad",
                default_value="1.20",
            ),
            DeclareLaunchArgument(
                "max_coast_sec",
                default_value="1.0",
            ),
            DeclareLaunchArgument(
                "max_rejected_samples",
                default_value="200",
            ),
            DeclareLaunchArgument(
                "status_period_sec",
                default_value="2.0",
                description=(
                    "Health-check interval. Healthy operation is silent; "
                    "only problems are logged."
                ),
            ),
            OpaqueFunction(function=_setup),
        ]
    )
