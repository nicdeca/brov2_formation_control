"""Three-BlueROV formation experiment.

Directed sensing/control graph (follower -> parent):

    robot 2 -> robot 1
    robot 3 -> robot 1

The default robot names are the generic experiment names
``itrl_rov_1``, ``itrl_rov_2``, and ``itrl_rov_3``.  Physical robot names
(e.g. ``splash``, ``glub``, and ``bubble``) can be supplied as launch
arguments without changing the experiment topology.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


PHASE_TOPIC = "/formation_control/experiment_phase"
FORMATION_TOPIC = "/formation_control/desired_formation"

# Pool-aligned core-NWU initialization geometry.
INITIAL_LEADER_POSITION = [2.675, 0.050, -0.775]
INITIAL_LEFT_POSITION = [4.475, -0.650, -0.775]
INITIAL_RIGHT_POSITION = [4.475, 0.750, -0.775]

# Desired parent-minus-follower vectors in pool-aligned core NWU.
INITIAL_LEFT = [-1.800, 0.700, 0.000]
INITIAL_RIGHT = [-1.800, -0.700, 0.000]

FORMATION_NAMES = [
    "triangle_nominal",
    "triangle_wide",
    "triangle_compact",
    "triangle_high",
]

# Each row is the desired parent-minus-follower vector in pool-aligned
# core NWU.
LEFT_FORMATIONS = [
    -1.80,  0.70,  0.00,  # triangle_nominal
    -2.30,  1.15,  0.00,  # triangle_wide
    -1.25,  0.35,  0.00,  # triangle_compact
    -1.80,  0.70, -0.30,  # triangle_high
]

RIGHT_FORMATIONS = [
    -1.80, -0.70,  0.00,  # triangle_nominal
    -2.30, -1.15,  0.00,  # triangle_wide
    -1.25, -0.35,  0.00,  # triangle_compact
    -1.80, -0.70,  0.30,  # triangle_high
]


def _phase_manager_setup(context):
    """Resolve robot names before constructing STRING_ARRAY parameters."""
    leader = LaunchConfiguration("leader").perform(context)
    follower_left = LaunchConfiguration("follower_left").perform(context)
    follower_right = LaunchConfiguration("follower_right").perform(context)

    return [
        Node(
            package="formation_control_ros",
            executable="experiment_phase_manager",
            name="experiment_phase_manager",
            output="screen",
            parameters=[
                {
                    "robot_names": [
                        leader,
                        follower_left,
                        follower_right,
                    ],
                    "initial_positions": [
                        *INITIAL_LEADER_POSITION,
                        *INITIAL_LEFT_POSITION,
                        *INITIAL_RIGHT_POSITION,
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
    robot_configuration,
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
                    "robot_configuration": robot_configuration,
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
    leader = LaunchConfiguration("leader")
    follower_left = LaunchConfiguration("follower_left")
    follower_right = LaunchConfiguration("follower_right")

    leader_robot_configuration = LaunchConfiguration(
        "leader_robot_configuration"
    )
    follower_left_robot_configuration = LaunchConfiguration(
        "follower_left_robot_configuration"
    )
    follower_right_robot_configuration = LaunchConfiguration(
        "follower_right_robot_configuration"
    )

    dt = LaunchConfiguration("dt")
    dry_run = LaunchConfiguration("dry_run")
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

    workspace_barrier_enabled = LaunchConfiguration(
        "workspace_barrier_enabled"
    )
    workspace_adaptive = LaunchConfiguration("workspace_adaptive")

    common = {
        "dt": dt,
        "state_source": "px4",
        "control_space": "thruster",
        "dry_run": dry_run,
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
        "alpha_gain": ParameterValue(
            alpha_gain,
            value_type=float,
        ),
    }

    actions = [
        DeclareLaunchArgument("leader", default_value="itrl_rov_1"),
        DeclareLaunchArgument("follower_left", default_value="itrl_rov_2"),
        DeclareLaunchArgument("follower_right", default_value="itrl_rov_3"),
        DeclareLaunchArgument("dt", default_value="0.02"),
        DeclareLaunchArgument("dry_run", default_value="true"),
        DeclareLaunchArgument(
            "leader_reference_mode",
            default_value="stationary",
        ),
        DeclareLaunchArgument(
            "leader_robot_configuration",
            default_value="auto",
            description=(
                "Dynamics preset for the leader: "
                "'auto', 'gazebo', 'standard', or 'heavy_tube'."
            ),
        ),
        DeclareLaunchArgument(
            "follower_left_robot_configuration",
            default_value="auto",
            description=(
                "Dynamics preset for follower 2: "
                "'auto', 'gazebo', 'standard', or 'heavy_tube'."
            ),
        ),
        DeclareLaunchArgument(
            "follower_right_robot_configuration",
            default_value="auto",
            description=(
                "Dynamics preset for follower 3: "
                "'auto', 'gazebo', 'standard', or 'heavy_tube'."
            ),
        ),
        DeclareLaunchArgument("position_gain", default_value="2.0"),
        DeclareLaunchArgument("formation_gain", default_value="2.0"),
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
        DeclareLaunchArgument("alpha_gain", default_value="1.8"),
        DeclareLaunchArgument(
            "workspace_barrier_enabled",
            default_value="true",
        ),
        DeclareLaunchArgument(
            "workspace_adaptive",
            default_value="true",
        ),
        OpaqueFunction(function=_phase_manager_setup),
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
                    "robot_configuration": leader_robot_configuration,
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
            namespace=leader,
            name="heartbeat",
            output="screen",
        ),
    ]

    actions.extend(
        _follower_actions(
            robot=follower_left,
            parent=leader,
            robot_configuration=follower_left_robot_configuration,
            initial_relative=INITIAL_LEFT,
            initial_position=INITIAL_LEFT_POSITION,
            formations=LEFT_FORMATIONS,
            common=common,
            formation_gain=formation_gain,
        )
    )
    actions.extend(
        _follower_actions(
            robot=follower_right,
            parent=leader,
            robot_configuration=follower_right_robot_configuration,
            initial_relative=INITIAL_RIGHT,
            initial_position=INITIAL_RIGHT_POSITION,
            formations=RIGHT_FORMATIONS,
            common=common,
            formation_gain=formation_gain,
        )
    )

    return LaunchDescription(actions)
