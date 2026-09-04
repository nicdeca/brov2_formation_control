"""Launch PX4-SITL -> simulated MoCap pose + FLU gyro."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _setup(context):
    def arg(name: str) -> str:
        return LaunchConfiguration(name).perform(context)

    robots = [
        item.strip()
        for item in arg("robots").split(",")
        if item.strip()
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
                default_value="core_nwu",
            ),
            DeclareLaunchArgument(
                "imu_frame_id_template",
                default_value="{robot}/base_link",
            ),
            DeclareLaunchArgument(
                "status_period_sec",
                default_value="2.0",
                description=(
                    "Health-check interval. Healthy streams are silent; "
                    "only problems are logged."
                ),
            ),
            OpaqueFunction(function=_setup),
        ]
    )
