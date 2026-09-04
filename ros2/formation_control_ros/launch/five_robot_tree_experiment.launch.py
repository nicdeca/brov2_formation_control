"""Five-BlueROV balanced-tree experiment with selectable state input.

Directed sensing/control graph (follower -> parent):

    2 -> 1
    3 -> 1
    4 -> 2
    5 -> 3

PX4 VehicleOdometry remains the default. A generic nav_msgs/Odometry source
can be selected through launch arguments for estimator comparison.
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
INITIAL_ROBOT_2_POSITION = [4.325, -0.650, -0.775]
INITIAL_ROBOT_3_POSITION = [4.325, 0.750, -0.775]
INITIAL_ROBOT_4_POSITION = [5.775, -1.150, -0.775]
INITIAL_ROBOT_5_POSITION = [5.775, 1.250, -0.775]

INITIAL_D21 = [-1.650, 0.700, 0.000]
INITIAL_D31 = [-1.650, -0.700, 0.000]
INITIAL_D42 = [-1.450, 0.500, 0.000]
INITIAL_D53 = [-1.450, -0.500, 0.000]

FORMATION_NAMES = [
    "tree_nominal",
    "tree_wide",
    "tree_compact",
    "tree_staggered",
    "tree_depth_split",
    "tree_crossed_3d",
    "tree_opposed_3d",
    "tree_parallel_3d",
]

D21_FORMATIONS = [
    -1.65, 0.70, 0.00,
    -1.95, 0.80, 0.00,
    -1.30, 0.50, 0.00,
    -1.65, 0.70, 0.10,
    -1.65, 0.70, 0.55,
    -1.30, 0.95, 0.55,
    -1.15, -1.35, 0.55,
    -2.25, 1.10, 0.70,
]

D31_FORMATIONS = [
    -1.65, -0.70, 0.00,
    -1.95, -0.80, 0.00,
    -1.30, -0.50, 0.00,
    -1.65, -0.70, -0.10,
    -1.65, -0.70, -0.35,
    -1.30, -0.95, -0.35,
    -1.15, 1.35, -0.35,
    -2.25, -1.10, -0.35,
]

D42_FORMATIONS = [
    -1.45, 0.50, 0.00,
    -1.65, 0.55, 0.00,
    -1.10, 0.35, 0.00,
    -1.45, 0.50, -0.10,
    -1.45, 0.50, 0.40,
    -1.60, -2.30, 0.40,
    -2.30, -0.05, 0.35,
    -1.80, -0.00, 0.00,
]

D53_FORMATIONS = [
    -1.45, -0.50, 0.00,
    -1.65, -0.55, 0.00,
    -1.10, -0.35, 0.00,
    -1.45, -0.50, 0.10,
    -1.45, -0.50, 0.00,
    -1.60, 2.30, 0.00,
    -2.30, 0.05, 0.00,
    -1.80, -0.00, 0.00,
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


def _follower_node(
    *,
    robot,
    parent,
    initial_relative,
    initial_position,
    formations,
    common,
    topic,
    formation_gain,
):
    return Node(
        package="formation_control_ros",
        executable="follower_controller",
        namespace=robot,
        name="controller",
        output="screen",
        parameters=[
            common,
            {
                "robot_name": robot,
                "parent_robot_name": parent,
                "self_odometry_topic": topic(robot),
                "parent_odometry_topic": topic(parent),
                "desired_relative_position": initial_relative,
                "initialization_position": initial_position,
                "experiment_phase_topic": PHASE_TOPIC,
                "desired_formation_topic": FORMATION_TOPIC,
                "formation_names": FORMATION_NAMES,
                "formation_relative_positions": formations,
                "formation_gain": formation_gain,
                "adaptive": True,
            },
        ],
    )


def _setup(context):
    names = [
        LaunchConfiguration(f"robot_{index}").perform(context)
        for index in range(1, 6)
    ]
    state_source, template, topic = _state_settings(context)
    common = _common_parameters(context)

    # The canonical five-robot experiment remains a Gazebo/SITL experiment.
    common["robot_configuration"] = "gazebo"
    common["use_sim_time"] = (
        LaunchConfiguration("gazebo_timer").perform(context).lower()
        in ("1", "true", "yes", "on")
    )

    formation_gain = float(
        LaunchConfiguration("formation_gain").perform(context)
    )

    nodes = [
        Node(
            package="formation_control_ros",
            executable="experiment_phase_manager",
            name="experiment_phase_manager",
            output="screen",
            parameters=[
                {
                    "robot_names": names,
                    "initial_positions": [
                        *INITIAL_LEADER_POSITION,
                        *INITIAL_ROBOT_2_POSITION,
                        *INITIAL_ROBOT_3_POSITION,
                        *INITIAL_ROBOT_4_POSITION,
                        *INITIAL_ROBOT_5_POSITION,
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
            namespace=names[0],
            name="controller",
            output="screen",
            parameters=[
                common,
                {
                    "robot_name": names[0],
                    "odometry_topic": topic(names[0]),
                    "reference_mode": LaunchConfiguration(
                        "leader_reference_mode"
                    ).perform(context),
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
            namespace=names[0],
            name="heartbeat",
            output="screen",
        ),
    ]

    specs = [
        (
            names[1],
            names[0],
            INITIAL_D21,
            INITIAL_ROBOT_2_POSITION,
            D21_FORMATIONS,
        ),
        (
            names[2],
            names[0],
            INITIAL_D31,
            INITIAL_ROBOT_3_POSITION,
            D31_FORMATIONS,
        ),
        (
            names[3],
            names[1],
            INITIAL_D42,
            INITIAL_ROBOT_4_POSITION,
            D42_FORMATIONS,
        ),
        (
            names[4],
            names[2],
            INITIAL_D53,
            INITIAL_ROBOT_5_POSITION,
            D53_FORMATIONS,
        ),
    ]

    for robot, parent, relative, position, formations in specs:
        nodes.extend(
            [
                _follower_node(
                    robot=robot,
                    parent=parent,
                    initial_relative=relative,
                    initial_position=position,
                    formations=formations,
                    common=common,
                    topic=topic,
                    formation_gain=formation_gain,
                ),
                Node(
                    package="formation_control_ros",
                    executable="offboard_heartbeat_wrench",
                    namespace=robot,
                    name="heartbeat",
                    output="screen",
                ),
            ]
        )

    return nodes


def generate_launch_description() -> LaunchDescription:
    gazebo_timer = LaunchConfiguration("gazebo_timer")

    actions = [
        DeclareLaunchArgument("robot_1", default_value="itrl_rov_1"),
        DeclareLaunchArgument("robot_2", default_value="itrl_rov_2"),
        DeclareLaunchArgument("robot_3", default_value="itrl_rov_3"),
        DeclareLaunchArgument("robot_4", default_value="itrl_rov_4"),
        DeclareLaunchArgument("robot_5", default_value="itrl_rov_5"),
        DeclareLaunchArgument("dt", default_value="0.02"),
        DeclareLaunchArgument("dry_run", default_value="true"),
        DeclareLaunchArgument(
            "gazebo_timer",
            default_value="false",
        ),
        DeclareLaunchArgument(
            "leader_reference_mode",
            default_value="stationary",
        ),
        DeclareLaunchArgument("position_gain", default_value="2.0"),
        DeclareLaunchArgument("formation_gain", default_value="1.0"),
        DeclareLaunchArgument("virtual_linear_gain", default_value="1.0"),
        DeclareLaunchArgument("virtual_angular_gain", default_value="1.2"),
        DeclareLaunchArgument(
            "command_filter_linear_bandwidth",
            default_value="10.0",
        ),
        DeclareLaunchArgument(
            "command_filter_angular_bandwidth",
            default_value="10.0",
        ),
        DeclareLaunchArgument("alpha_gain", default_value="3.0"),
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
