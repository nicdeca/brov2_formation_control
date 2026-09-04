"""Two-BlueROV adaptive sensing-domain mission with selectable state input.

This launch retains the dedicated Example-12 sensing/adaptation tuning while
allowing the estimator input to be selected from the command line.

Directed sensing edge:
    robot 2 -> robot 1
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


PHASE_TOPIC = "/formation_control/experiment_phase"
FORMATION_TOPIC = "/formation_control/desired_formation"

INITIAL_LEADER_POSITION = [2.675, 0.050, -0.775]
INITIAL_RELATIVE = [-1.800, -0.700, 0.000]
INITIAL_FOLLOWER_POSITION = [4.475, 0.750, -0.775]

FORMATION_NAMES = [
    "adaptive_A",
    "adaptive_B",
    "adaptive_C",
    "adaptive_D",
]

FORMATION_RELATIVE_POSITIONS = [
    -1.80, -0.70,  0.00,
    -1.65, -1.45, -0.40,
    -1.65,  1.45,  0.40,
    -2.30, -0.55, -0.30,
]

def _state_settings(context):
    state_source = LaunchConfiguration("state_source").perform(context).strip().lower()
    if state_source not in ("px4", "nav_msgs"):
        raise ValueError("state_source must be 'px4' or 'nav_msgs'.")

    template = LaunchConfiguration("state_topic_template").perform(context).strip()
    if not template:
        template = (
            "/{robot}/fmu/out/vehicle_odometry"
            if state_source == "px4"
            else "/mocap/{robot}/odom"
        )

    if "{robot}" not in template and "{robot_lower}" not in template:
        raise ValueError(
            "state_topic_template must contain '{robot}' or '{robot_lower}'."
        )

    def topic(robot):
        name = str(robot)
        return (
            template
            .replace("{robot}", name)
            .replace("{robot_lower}", name.lower())
        )

    return state_source, template, topic


def _common_parameters(context):
    state_source, _, _ = _state_settings(context)
    return {
        "dt": float(LaunchConfiguration("dt").perform(context)),
        "state_source": state_source,
        "mocap_world_frame": LaunchConfiguration(
            "mocap_world_frame"
        ).perform(context),
        "odom_twist_frame": LaunchConfiguration(
            "odom_twist_frame"
        ).perform(context),
        "control_space": "thruster",
        "dry_run": LaunchConfiguration("dry_run").perform(context).lower()
        in ("1", "true", "yes", "on"),
        "workspace_barrier_enabled": (
            LaunchConfiguration("workspace_barrier_enabled")
            .perform(context)
            .lower()
            in ("1", "true", "yes", "on")
        ),
        "workspace_adaptive": (
            LaunchConfiguration("workspace_adaptive")
            .perform(context)
            .lower()
            in ("1", "true", "yes", "on")
        ),
        "workspace_physical_lower": [0.300, -1.975, -2.155],
        "workspace_physical_upper": [7.100, 1.975, 0.225],
        "workspace_conservative_lower": [0.450, -1.825, -1.955],
        "workspace_conservative_upper": [6.950, 1.825, -0.325],
        "workspace_barrier_weight": 0.10,
        "workspace_reference_margin": 0.05,
        "workspace_relaxation_recovery_gain": 0.8,
        "workspace_relaxation_domain_margin_ratio": 0.10,
        "workspace_minimum_constraint_margin": 1e-3,
        "px4_thrust_command_limit": 0.10,
        "px4_torque_command_limit": 0.10,
        "virtual_linear_gain": float(
            LaunchConfiguration("virtual_linear_gain").perform(context)
        ),
        "virtual_angular_gain": float(
            LaunchConfiguration("virtual_angular_gain").perform(context)
        ),
        "command_filter_linear_bandwidth": float(
            LaunchConfiguration(
                "command_filter_linear_bandwidth"
            ).perform(context)
        ),
        "command_filter_angular_bandwidth": float(
            LaunchConfiguration(
                "command_filter_angular_bandwidth"
            ).perform(context)
        ),
        "alpha_gain": float(
            LaunchConfiguration("alpha_gain").perform(context)
        ),
    }


def _state_launch_arguments():
    return [
        DeclareLaunchArgument(
            "state_source",
            default_value="px4",
            description=(
                "State message type: 'px4' for px4_msgs/VehicleOdometry or "
                "'nav_msgs' for nav_msgs/Odometry."
            ),
        ),
        DeclareLaunchArgument(
            "state_topic_template",
            default_value="",
            description=(
                "Per-robot state topic template. Use {robot} or "
                "{robot_lower}. Empty selects the source default: "
                "/{robot}/fmu/out/vehicle_odometry for px4, "
                "/mocap/{robot}/odom for nav_msgs."
            ),
        ),
        DeclareLaunchArgument(
            "mocap_world_frame",
            default_value="core_nwu",
            description=(
                "World-frame convention for nav_msgs/Odometry: "
                "'core_nwu' or 'ros_enu'."
            ),
        ),
        DeclareLaunchArgument(
            "odom_twist_frame",
            default_value="body",
            description=(
                "Twist convention for nav_msgs/Odometry: 'body' or 'world'."
            ),
        ),
    ]


def _setup(context):
    leader = LaunchConfiguration("leader").perform(context)
    follower = LaunchConfiguration("follower").perform(context)
    state_source, template, topic = _state_settings(context)

    common = _common_parameters(context)
    common["robot_configuration"] = LaunchConfiguration(
        "robot_configuration"
    ).perform(context)
    common["use_sim_time"] = (
        LaunchConfiguration("gazebo_timer").perform(context).lower()
        in ("1", "true", "yes", "on")
    )

    follower_task = {
        "robot_name": follower,
        "parent_robot_name": leader,
        "self_odometry_topic": topic(follower),
        "parent_odometry_topic": topic(leader),
        "desired_relative_position": INITIAL_RELATIVE,
        "initialization_position": INITIAL_FOLLOWER_POSITION,
        "experiment_phase_topic": PHASE_TOPIC,
        "desired_formation_topic": FORMATION_TOPIC,
        "formation_names": FORMATION_NAMES,
        "formation_relative_positions": FORMATION_RELATIVE_POSITIONS,
        "formation_gain": float(
            LaunchConfiguration("formation_gain").perform(context)
        ),
        "image_horizontal_gain": 0.8,
        "image_vertical_gain": 0.8,
        "d_min": 0.5,
        "d_max": 3.6,
        "d_min_conservative": 0.8,
        "d_max_conservative": float(
            LaunchConfiguration("d_max_conservative").perform(context)
        ),
        "alpha_h_conservative": float(
            LaunchConfiguration("alpha_h_conservative").perform(context)
        ),
        "alpha_v_conservative": float(
            LaunchConfiguration("alpha_v_conservative").perform(context)
        ),
        "horizontal_half_angle_deg": 45.0,
        "vertical_half_angle_deg": 30.0,
        "collision_barrier_weight": 0.18,
        "range_barrier_weight": 0.18,
        "horizontal_fov_barrier_weight": 0.25,
        "vertical_fov_barrier_weight": 0.25,
        "adaptive": True,
        "relaxation_recovery_gain": float(
            LaunchConfiguration(
                "relaxation_recovery_gain"
            ).perform(context)
        ),
        "relaxation_barrier_gain": float(
            LaunchConfiguration(
                "relaxation_barrier_gain"
            ).perform(context)
        ),
        "relaxation_domain_margin_ratio": float(
            LaunchConfiguration(
                "relaxation_domain_margin_ratio"
            ).perform(context)
        ),
        "relaxation_activation_on_ratio": float(
            LaunchConfiguration(
                "relaxation_activation_on_ratio"
            ).perform(context)
        ),
        "relaxation_activation_off_ratio": float(
            LaunchConfiguration(
                "relaxation_activation_off_ratio"
            ).perform(context)
        ),
        "use_parent_velocity_in_clf": False,
    }

    return [
        Node(
            package="formation_control_ros",
            executable="experiment_phase_manager",
            name="experiment_phase_manager",
            output="screen",
            parameters=[
                {
                    "robot_names": [leader, follower],
                    "initial_positions": [
                        *INITIAL_LEADER_POSITION,
                        *INITIAL_FOLLOWER_POSITION,
                    ],
                    "phase_topic": PHASE_TOPIC,
                    "position_tolerance": 0.65,
                    "speed_tolerance": 0.08,
                    "settle_time": 1.5,
                    "state_source": state_source,
                    "state_topic_template": template,
                    "mocap_world_frame": LaunchConfiguration(
                        "mocap_world_frame"
                    ).perform(context),
                    "odom_twist_frame": LaunchConfiguration(
                        "odom_twist_frame"
                    ).perform(context),
                }
            ],
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
                    "odometry_topic": topic(leader),
                    "reference_mode": "velocity",
                    "experiment_phase_topic": PHASE_TOPIC,
                    "initialization_position": INITIAL_LEADER_POSITION,
                    "position_gain": float(
                        LaunchConfiguration("position_gain").perform(context)
                    ),
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
            parameters=[common, follower_task],
        ),
        Node(
            package="formation_control_ros",
            executable="offboard_heartbeat_wrench",
            namespace=follower,
            name="heartbeat",
            output="screen",
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    gazebo_timer = LaunchConfiguration("gazebo_timer")

    actions = [
        DeclareLaunchArgument("leader", default_value="itrl_rov_1"),
        DeclareLaunchArgument("follower", default_value="itrl_rov_2"),
        DeclareLaunchArgument("dt", default_value="0.02"),
        DeclareLaunchArgument("dry_run", default_value="true"),
        DeclareLaunchArgument(
            "gazebo_timer",
            default_value="false",
        ),
        DeclareLaunchArgument(
            "robot_configuration",
            default_value="gazebo",
            description=(
                "Dynamics preset. Keep 'gazebo' for the canonical SITL "
                "adaptive mission."
            ),
        ),
        DeclareLaunchArgument("position_gain", default_value="2.0"),
        DeclareLaunchArgument("formation_gain", default_value="1.4"),
        DeclareLaunchArgument("virtual_linear_gain", default_value="0.55"),
        DeclareLaunchArgument("virtual_angular_gain", default_value="0.80"),
        DeclareLaunchArgument(
            "command_filter_linear_bandwidth",
            default_value="3.0",
        ),
        DeclareLaunchArgument(
            "command_filter_angular_bandwidth",
            default_value="4.0",
        ),
        DeclareLaunchArgument("alpha_gain", default_value="0.8"),
        DeclareLaunchArgument("d_max_conservative", default_value="2.4"),
        DeclareLaunchArgument("alpha_h_conservative", default_value="0.45"),
        DeclareLaunchArgument("alpha_v_conservative", default_value="0.45"),
        DeclareLaunchArgument(
            "relaxation_recovery_gain",
            default_value="0.8",
        ),
        DeclareLaunchArgument(
            "relaxation_barrier_gain",
            default_value="0.20",
        ),
        DeclareLaunchArgument(
            "relaxation_domain_margin_ratio",
            default_value="0.02",
        ),
        DeclareLaunchArgument(
            "relaxation_activation_on_ratio",
            default_value="0.001",
        ),
        DeclareLaunchArgument(
            "relaxation_activation_off_ratio",
            default_value="0.15",
        ),
        DeclareLaunchArgument(
            "workspace_barrier_enabled",
            default_value="true",
        ),
        DeclareLaunchArgument(
            "workspace_adaptive",
            default_value="true",
        ),
        *_state_launch_arguments(),
        ExecuteProcess(
            cmd=[
                "ros2",
                "run",
                "ros_gz_bridge",
                "parameter_bridge",
                "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            ],
            output="screen",
            condition=IfCondition(gazebo_timer),
        ),
        OpaqueFunction(function=_setup),
    ]
    return LaunchDescription(actions)
