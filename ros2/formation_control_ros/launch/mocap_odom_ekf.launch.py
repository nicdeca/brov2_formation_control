"""Launch the core-NWU / FLU MoCap state estimator for one or more robots."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _as_bool(value: str) -> bool:
    value = value.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"invalid boolean value {value!r}")


def _csv_floats(value: str, size: int, name: str) -> list[float]:
    result = [
        float(item.strip())
        for item in value.split(",")
        if item.strip()
    ]
    if len(result) != size:
        raise ValueError(
            f"{name} must contain {size} comma-separated values"
        )
    return result


def _expand(template: str, robot: str) -> str:
    return (
        template
        .replace("{robot}", robot)
        .replace("{robot_lower}", robot.lower())
    )


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

    pose_template = arg("pose_topic_template")
    core_pose_template = arg("core_pose_topic_template")
    odom_template = arg("odom_topic_template")
    imu_template = arg("imu_topic_template")

    for name, template in (
        ("pose_topic_template", pose_template),
        ("core_pose_topic_template", core_pose_template),
        ("odom_topic_template", odom_template),
        ("imu_topic_template", imu_template),
    ):
        if (
            len(robots) > 1
            and "{robot}" not in template
            and "{robot_lower}" not in template
        ):
            raise ValueError(
                f"{name} must contain '{{robot}}' or '{{robot_lower}}' "
                "for multiple robots"
            )

    common = {
        "parent_frame": arg("parent_frame"),
        "publish_rate_hz": float(arg("publish_rate_hz")),
        "publish_tf": _as_bool(arg("publish_tf")),
        "input_world_frame": arg("input_world_frame"),
        "input_body_frame": arg("input_body_frame"),
        "world_to_core_translation": _csv_floats(
            arg("world_to_core_translation"),
            3,
            "world_to_core_translation",
        ),
        "world_to_core_quaternion_xyzw": _csv_floats(
            arg("world_to_core_quaternion_xyzw"),
            4,
            "world_to_core_quaternion_xyzw",
        ),
        "body_flu_to_input_quaternion_xyzw": _csv_floats(
            arg("body_flu_to_input_quaternion_xyzw"),
            4,
            "body_flu_to_input_quaternion_xyzw",
        ),
        "body_origin_offset_input_body": _csv_floats(
            arg("body_origin_offset_input_body"),
            3,
            "body_origin_offset_input_body",
        ),
        "use_imu_gyro": _as_bool(arg("use_imu_gyro")),
        "imu_body_frame": arg("imu_body_frame"),
        "body_flu_to_imu_quaternion_xyzw": _csv_floats(
            arg("body_flu_to_imu_quaternion_xyzw"),
            4,
            "body_flu_to_imu_quaternion_xyzw",
        ),
        "gyro_timeout_sec": float(arg("gyro_timeout_sec")),
        "gyro_time_constant_sec": float(
            arg("gyro_time_constant_sec")
        ),
        "gyro_std": float(arg("gyro_std")),
        "max_gyro_abs_rad_s": float(arg("max_gyro_abs_rad_s")),
        "position_std": float(arg("position_std")),
        "linear_accel_std": float(arg("linear_accel_std")),
        "initial_velocity_std": float(
            arg("initial_velocity_std")
        ),
        "orientation_std": float(arg("orientation_std")),
        "orientation_measurement_gain": float(
            arg("orientation_measurement_gain")
        ),
        "mocap_angular_velocity_time_constant_sec": float(
            arg("mocap_angular_velocity_time_constant_sec")
        ),
        "max_position_innovation_m": float(
            arg("max_position_innovation_m")
        ),
        "max_orientation_innovation_rad": float(
            arg("max_orientation_innovation_rad")
        ),
        "max_body_z_axis_angle_rad": float(
            arg("max_body_z_axis_angle_rad")
        ),
        "max_coast_sec": float(arg("max_coast_sec")),
        "max_rejected_samples": int(
            arg("max_rejected_samples")
        ),
        "status_period_sec": float(arg("status_period_sec")),
    }

    nodes = []
    for robot in robots:
        parameters = dict(common)
        parameters.update(
            {
                "pose_topic": _expand(pose_template, robot),
                "core_pose_topic": _expand(
                    core_pose_template,
                    robot,
                ),
                "odom_topic": _expand(odom_template, robot),
                "imu_topic": _expand(imu_template, robot),
                "child_frame": f"{robot}/base_link_ekf",
            }
        )
        nodes.append(
            Node(
                package="formation_control_ros",
                executable="mocap_odom_ekf",
                namespace=robot,
                name="mocap_odom_ekf",
                output="screen",
                parameters=[parameters],
            )
        )
    return nodes


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "robots",
                default_value="itrl_rov_1",
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
                "parent_frame",
                default_value="core_nwu",
            ),
            DeclareLaunchArgument(
                "publish_rate_hz",
                default_value="80.0",
            ),
            DeclareLaunchArgument(
                "publish_tf",
                default_value="false",
            ),
            # Incoming real MoCap convention.
            DeclareLaunchArgument(
                "input_world_frame",
                default_value="core_nwu",
                description="core_nwu | ros_enu | custom",
            ),
            DeclareLaunchArgument(
                "input_body_frame",
                default_value="flu",
                description="flu | frd | custom",
            ),
            DeclareLaunchArgument(
                "world_to_core_translation",
                default_value="0,0,0",
                description=(
                    "Input-world origin expressed in core NWU [m]. "
                    "Used for all modes; especially important for real MoCap."
                ),
            ),
            DeclareLaunchArgument(
                "world_to_core_quaternion_xyzw",
                default_value="0,0,0,1",
                description=(
                    "Custom rotation input-world -> core, ROS xyzw."
                ),
            ),
            DeclareLaunchArgument(
                "body_flu_to_input_quaternion_xyzw",
                default_value="0,0,0,1",
                description=(
                    "Custom rotation FLU -> incoming rigid-body frame."
                ),
            ),
            DeclareLaunchArgument(
                "body_origin_offset_input_body",
                default_value="0,0,0",
                description=(
                    "Vector from MoCap rigid-body origin to controller "
                    "body origin, expressed in incoming body frame [m]."
                ),
            ),
            # Gyro.
            DeclareLaunchArgument(
                "use_imu_gyro",
                default_value="true",
            ),
            DeclareLaunchArgument(
                "imu_body_frame",
                default_value="flu",
                description="flu | frd | custom",
            ),
            DeclareLaunchArgument(
                "body_flu_to_imu_quaternion_xyzw",
                default_value="0,0,0,1",
            ),
            DeclareLaunchArgument(
                "gyro_timeout_sec",
                default_value="0.20",
            ),
            DeclareLaunchArgument(
                "gyro_time_constant_sec",
                default_value="0.03",
            ),
            DeclareLaunchArgument(
                "gyro_std",
                default_value="0.03",
            ),
            DeclareLaunchArgument(
                "max_gyro_abs_rad_s",
                default_value="5.0",
            ),
            # Estimator tuning.
            DeclareLaunchArgument(
                "position_std",
                default_value="0.01",
            ),
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
