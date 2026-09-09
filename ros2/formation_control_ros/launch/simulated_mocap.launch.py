"""Launch the PX4-SITL -> laboratory-like simulated MoCap adapter."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _setup(context):
    def arg(name: str) -> str:
        return LaunchConfiguration(name).perform(context)

    robots = [
        value.strip()
        for value in arg("robots").split(",")
        if value.strip()
    ]
    if not robots:
        raise ValueError("robots must contain at least one name")
    if len(set(robots)) != len(robots):
        raise ValueError("robots must contain unique names")

    return [
        Node(
            package="formation_control_ros",
            executable="simulated_mocap",
            name="simulated_mocap",
            output="screen",
            parameters=[
                {
                    "robots": robots,
                    "input_topic_template": arg(
                        "input_topic_template"
                    ),
                    "output_pose_topic_template": arg(
                        "output_pose_topic_template"
                    ),
                    "output_imu_topic_template": arg(
                        "output_imu_topic_template"
                    ),
                    "pose_frame_id": arg("pose_frame_id"),
                    "imu_frame_id_template": arg(
                        "imu_frame_id_template"
                    ),
                    "measurement_mode": arg("measurement_mode"),
                    "dropout_start_sec": float(
                        arg("dropout_start_sec")
                    ),
                    "dropout_period_sec": float(
                        arg("dropout_period_sec")
                    ),
                    "dropout_duration_sec": float(
                        arg("dropout_duration_sec")
                    ),
                    "status_period_sec": float(
                        arg("status_period_sec")
                    ),
                }
            ],
        )
    ]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "robots",
                default_value="itrl_rov_1",
            ),
            DeclareLaunchArgument(
                "input_topic_template",
                default_value="/{robot}/fmu/out/vehicle_odometry",
            ),
            DeclareLaunchArgument(
                "output_pose_topic_template",
                default_value="/mocap/{robot}/pose",
            ),
            DeclareLaunchArgument(
                "output_imu_topic_template",
                default_value="/mocap/{robot}/imu",
            ),
            DeclareLaunchArgument(
                "pose_frame_id",
                default_value="mocap_ned",
            ),
            DeclareLaunchArgument(
                "imu_frame_id_template",
                default_value="{robot}/base_link_frd",
            ),
            DeclareLaunchArgument(
                "measurement_mode",
                default_value="ideal",
                description="Simulated MoCap pose delivery: ideal or intermittent.",
            ),
            DeclareLaunchArgument(
                "dropout_start_sec",
                default_value="5.0",
                description=(
                    "Intermittent mode: time before the first pose dropout."
                ),
            ),
            DeclareLaunchArgument(
                "dropout_period_sec",
                default_value="10.0",
                description=(
                    "Intermittent mode: period between dropout-window starts."
                ),
            ),
            DeclareLaunchArgument(
                "dropout_duration_sec",
                default_value="2.0",
                description=(
                    "Intermittent mode: duration of each suppressed-pose window."
                ),
            ),
            DeclareLaunchArgument(
                "status_period_sec",
                default_value="2.0",
            ),
            OpaqueFunction(function=_setup),
        ]
    )
