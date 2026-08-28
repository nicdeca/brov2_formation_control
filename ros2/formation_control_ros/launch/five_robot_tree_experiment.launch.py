"""Five-BlueROV balanced-tree paper simulation.

Directed sensing/control graph (follower -> parent):

    itrl_rov_4 -> itrl_rov_2 -> itrl_rov_1
    itrl_rov_5 -> itrl_rov_3 -> itrl_rov_1

Equivalently:
    2 -> 1
    3 -> 1
    4 -> 2
    5 -> 3

This is a depth-two rooted tree rather than a star.  Each robot still has at
most one parent, so the existing decentralized one-parent follower controller
is reused without changing the core control architecture.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

PHASE_TOPIC = "/formation_control/experiment_phase"
FORMATION_TOPIC = "/formation_control/desired_formation"

# ---------------------------------------------------------------------------
# Initial absolute positions, pool-aligned core NWU.
#
#                 1
#              /     \
#             2       3
#             |       |
#             4       5
#
# The two branches are symmetric about the leader.
# ---------------------------------------------------------------------------
INITIAL_LEADER_POSITION = [2.675, 0.050, -0.775]
INITIAL_ROBOT_2_POSITION = [4.325, -0.650, -0.775]
INITIAL_ROBOT_3_POSITION = [4.325, 0.750, -0.775]
INITIAL_ROBOT_4_POSITION = [5.775, -1.150, -0.775]
INITIAL_ROBOT_5_POSITION = [5.775, 1.250, -0.775]

# Desired parent-minus-follower vectors for the nominal tree.
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

# Per-edge parent-minus-follower vectors in pool-aligned core NWU.
#
# tree_nominal:
#   symmetric depth-two tree.
#
# tree_wide:
#   both levels expand primarily longitudinally. Lateral expansion is limited
#   so the leaf robots retain a useful margin from the x workspace boundaries.
#
# tree_compact:
#   contracts both tree levels; useful before larger leader translations.
#
# tree_staggered:
#   nominal horizontal geometry with alternating depth offsets. The second
#   level offsets oppose the first level, so z displacement does not accumulate
#   at the leaf robots.
#
# The four formations below are reserved for the challenging profile.
# All vectors have been rigidly rotated into the pool-aligned core NWU frame;
# their distances and relative 3-D geometry are unchanged.
#
# ``tree_depth_split`` establishes vertical separation before either pair of
# robots exchanges lateral sides. This keeps even the unconnected robot pairs
# separated during the large crossed transitions.
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


def _phase_manager_setup(context):
    names = [
        LaunchConfiguration(f"robot_{index}").perform(context) for index in range(1, 6)
    ]

    return [
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
                }
            ],
        )
    ]


def _follower_actions(
    *,
    robot,
    parent,
    initial_relative,
    initial_position,
    formations,
    common,
    formation_gain,
):
    return [
        Node(
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
                    "desired_relative_position": initial_relative,
                    "initialization_position": initial_position,
                    "experiment_phase_topic": PHASE_TOPIC,
                    "desired_formation_topic": FORMATION_TOPIC,
                    "formation_names": FORMATION_NAMES,
                    "formation_relative_positions": formations,
                    "formation_gain": ParameterValue(
                        formation_gain,
                        value_type=float,
                    ),
                    "adaptive": True,
                },
            ],
        ),
        Node(
            package="formation_control_ros",
            executable="offboard_heartbeat_wrench",
            namespace=robot,
            name="heartbeat",
            output="screen",
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    robot_1 = LaunchConfiguration("robot_1")
    robot_2 = LaunchConfiguration("robot_2")
    robot_3 = LaunchConfiguration("robot_3")
    robot_4 = LaunchConfiguration("robot_4")
    robot_5 = LaunchConfiguration("robot_5")

    dt = LaunchConfiguration("dt")
    dry_run = LaunchConfiguration("dry_run")
    gazebo_timer = LaunchConfiguration("gazebo_timer")
    leader_reference_mode = LaunchConfiguration("leader_reference_mode")

    position_gain = LaunchConfiguration("position_gain")
    formation_gain = LaunchConfiguration("formation_gain")
    virtual_linear_gain = LaunchConfiguration("virtual_linear_gain")
    virtual_angular_gain = LaunchConfiguration("virtual_angular_gain")
    command_filter_linear_bandwidth = LaunchConfiguration(
        "command_filter_linear_bandwidth"
    )
    command_filter_angular_bandwidth = LaunchConfiguration(
        "command_filter_angular_bandwidth"
    )
    alpha_gain = LaunchConfiguration("alpha_gain")

    workspace_barrier_enabled = LaunchConfiguration("workspace_barrier_enabled")
    workspace_adaptive = LaunchConfiguration("workspace_adaptive")

    common = {
        "dt": dt,
        "use_sim_time": ParameterValue(gazebo_timer, value_type=bool),
        "state_source": "px4",
        "control_space": "thruster",
        "robot_configuration": "gazebo",
        "dry_run": dry_run,
        "virtual_linear_gain": ParameterValue(
            virtual_linear_gain,
            value_type=float,
        ),
        "virtual_angular_gain": ParameterValue(
            virtual_angular_gain,
            value_type=float,
        ),
        "command_filter_linear_bandwidth": ParameterValue(
            command_filter_linear_bandwidth,
            value_type=float,
        ),
        "command_filter_angular_bandwidth": ParameterValue(
            command_filter_angular_bandwidth,
            value_type=float,
        ),
        "alpha_gain": ParameterValue(alpha_gain, value_type=float),
        "workspace_barrier_enabled": ParameterValue(
            workspace_barrier_enabled,
            value_type=bool,
        ),
        "workspace_adaptive": ParameterValue(
            workspace_adaptive,
            value_type=bool,
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
    }

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
            description=(
                "Run leader and follower controller timers from Gazebo "
                "/clock instead of wall time."
            ),
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
        OpaqueFunction(function=_phase_manager_setup),
        Node(
            package="formation_control_ros",
            executable="leader_controller",
            namespace=robot_1,
            name="controller",
            output="screen",
            parameters=[
                common,
                {
                    "robot_name": robot_1,
                    "reference_mode": leader_reference_mode,
                    "experiment_phase_topic": PHASE_TOPIC,
                    "initialization_position": INITIAL_LEADER_POSITION,
                    "position_gain": ParameterValue(
                        position_gain,
                        value_type=float,
                    ),
                },
            ],
        ),
        Node(
            package="formation_control_ros",
            executable="offboard_heartbeat_wrench",
            namespace=robot_1,
            name="heartbeat",
            output="screen",
        ),
    ]

    # Balanced rooted tree:
    # 2 -> 1, 3 -> 1, 4 -> 2, 5 -> 3.
    specs = [
        (
            robot_2,
            robot_1,
            INITIAL_D21,
            INITIAL_ROBOT_2_POSITION,
            D21_FORMATIONS,
        ),
        (
            robot_3,
            robot_1,
            INITIAL_D31,
            INITIAL_ROBOT_3_POSITION,
            D31_FORMATIONS,
        ),
        (
            robot_4,
            robot_2,
            INITIAL_D42,
            INITIAL_ROBOT_4_POSITION,
            D42_FORMATIONS,
        ),
        (
            robot_5,
            robot_3,
            INITIAL_D53,
            INITIAL_ROBOT_5_POSITION,
            D53_FORMATIONS,
        ),
    ]

    for robot, parent, relative, position, formations in specs:
        actions.extend(
            _follower_actions(
                robot=robot,
                parent=parent,
                initial_relative=relative,
                initial_position=position,
                formations=formations,
                common=common,
                formation_gain=formation_gain,
            )
        )

    return LaunchDescription(actions)
