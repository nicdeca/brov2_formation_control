"""Five-BlueROV paper simulation with automatic initialization.

Topology:
    itrl_rov_2 -> itrl_rov_1
    itrl_rov_3 -> itrl_rov_1
    itrl_rov_4 -> itrl_rov_1
    itrl_rov_5 -> itrl_rov_1

All four followers use the same decentralized one-parent follower controller.
The phase manager is the only centralized supervisory component and releases
FORMATION only once all five robots satisfy the initialization handoff.

The geometry is designed for the current Marinarium SITL workspace and should
be treated as a paper-simulation configuration, not a hardware configuration.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


PHASE_TOPIC = "/formation_control/experiment_phase"
FORMATION_TOPIC = "/formation_control/desired_formation"

# ---------------------------------------------------------------------------
# Initialization geometry, core NWU
# ---------------------------------------------------------------------------
# Leader centered horizontally.  Four followers form a transverse row at
# y = 2.95 m.  Adjacent follower spacing is 0.8 m.
INITIAL_LEADER_POSITION = [-1.20, 1.15, -95.20]

INITIAL_FOLLOWER_2_POSITION = [ 0.00, 2.95, -95.20]
INITIAL_FOLLOWER_3_POSITION = [-0.80, 2.95, -95.20]
INITIAL_FOLLOWER_4_POSITION = [-1.60, 2.95, -95.20]
INITIAL_FOLLOWER_5_POSITION = [-2.40, 2.95, -95.20]

# Parent-minus-follower vectors.
INITIAL_D2 = [-1.20, -1.80, 0.00]
INITIAL_D3 = [-0.40, -1.80, 0.00]
INITIAL_D4 = [ 0.40, -1.80, 0.00]
INITIAL_D5 = [ 1.20, -1.80, 0.00]

FORMATION_NAMES = [
    "five_nominal",
    "five_wide",
    "five_compact",
    "five_staggered",
]

# Each flat array stores one parent-minus-follower vector for each named
# formation, in the same order as FORMATION_NAMES.
#
# five_nominal:
#   four followers in a transverse row, 0.8 m apart.
#
# five_wide:
#   larger lateral spread and larger leader-follower distance.
#
# five_compact:
#   smaller longitudinal distance and 0.7 m adjacent follower spacing.
#
# five_staggered:
#   nominal horizontal footprint with alternating depth offsets.
D2_FORMATIONS = [
    -1.20, -1.80,  0.00,
    -1.40, -2.20,  0.00,
    -1.05, -1.40,  0.00,
    -1.20, -1.80,  0.25,
]

D3_FORMATIONS = [
    -0.40, -1.80,  0.00,
    -0.50, -2.20,  0.00,
    -0.35, -1.40,  0.00,
    -0.40, -1.80,  0.10,
]

D4_FORMATIONS = [
     0.40, -1.80,  0.00,
     0.50, -2.20,  0.00,
     0.35, -1.40,  0.00,
     0.40, -1.80, -0.10,
]

D5_FORMATIONS = [
     1.20, -1.80,  0.00,
     1.40, -2.20,  0.00,
     1.05, -1.40,  0.00,
     1.20, -1.80, -0.25,
]


def _phase_manager_setup(context):
    leader = LaunchConfiguration("leader").perform(context)
    followers = [
        LaunchConfiguration(f"follower_{index}").perform(context)
        for index in range(2, 6)
    ]

    return [
        Node(
            package="formation_control_ros",
            executable="experiment_phase_manager",
            name="experiment_phase_manager",
            output="screen",
            parameters=[
                {
                    "robot_names": [leader, *followers],
                    "initial_positions": [
                        *INITIAL_LEADER_POSITION,
                        *INITIAL_FOLLOWER_2_POSITION,
                        *INITIAL_FOLLOWER_3_POSITION,
                        *INITIAL_FOLLOWER_4_POSITION,
                        *INITIAL_FOLLOWER_5_POSITION,
                    ],
                    "phase_topic": PHASE_TOPIC,
                    "position_tolerance": 0.65,
                    "speed_tolerance": 0.08,
                    "settle_time": 1.5,
                }
            ],
        )
    ]


def _follower_node(
    *,
    robot,
    leader,
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
                    "parent_robot_name": leader,
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
    follower_2 = LaunchConfiguration("follower_2")
    follower_3 = LaunchConfiguration("follower_3")
    follower_4 = LaunchConfiguration("follower_4")
    follower_5 = LaunchConfiguration("follower_5")

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

        # Dynamic controller tuning.
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

        # Stage-B workspace safety.
        "workspace_barrier_enabled": ParameterValue(
            workspace_barrier_enabled,
            value_type=bool,
        ),
        "workspace_adaptive": ParameterValue(
            workspace_adaptive,
            value_type=bool,
        ),
        "workspace_physical_lower": [-3.125, -1.225, -96.58],
        "workspace_physical_upper": [0.825, 5.575, -94.20],
        "workspace_conservative_lower": [-2.975, -1.075, -96.38],
        "workspace_conservative_upper": [0.675, 5.425, -94.75],
        "workspace_barrier_weight": 0.10,
        "workspace_reference_margin": 0.05,
        "workspace_relaxation_recovery_gain": 0.8,
        "workspace_relaxation_domain_margin_ratio": 0.10,
        "workspace_minimum_constraint_margin": 1e-3,

        "px4_thrust_command_limit": 0.10,
        "px4_torque_command_limit": 0.10,
    }

    actions = [
        DeclareLaunchArgument("leader", default_value="itrl_rov_1"),
        DeclareLaunchArgument("follower_2", default_value="itrl_rov_2"),
        DeclareLaunchArgument("follower_3", default_value="itrl_rov_3"),
        DeclareLaunchArgument("follower_4", default_value="itrl_rov_4"),
        DeclareLaunchArgument("follower_5", default_value="itrl_rov_5"),
        DeclareLaunchArgument("dt", default_value="0.02"),
        DeclareLaunchArgument("dry_run", default_value="true"),
        DeclareLaunchArgument(
            "leader_reference_mode",
            default_value="stationary",
        ),

        # Geometric gains validated in the smaller-team simulations.
        DeclareLaunchArgument("position_gain", default_value="3.0"),
        DeclareLaunchArgument("formation_gain", default_value="3.0"),

        # Explicit dynamic-layer defaults preserving the original controller.
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

    follower_specs = [
        (
            follower_2,
            INITIAL_D2,
            INITIAL_FOLLOWER_2_POSITION,
            D2_FORMATIONS,
        ),
        (
            follower_3,
            INITIAL_D3,
            INITIAL_FOLLOWER_3_POSITION,
            D3_FORMATIONS,
        ),
        (
            follower_4,
            INITIAL_D4,
            INITIAL_FOLLOWER_4_POSITION,
            D4_FORMATIONS,
        ),
        (
            follower_5,
            INITIAL_D5,
            INITIAL_FOLLOWER_5_POSITION,
            D5_FORMATIONS,
        ),
    ]

    for robot, relative, position, formations in follower_specs:
        actions.extend(
            _follower_node(
                robot=robot,
                leader=leader,
                initial_relative=relative,
                initial_position=position,
                formations=formations,
                common=common,
                formation_gain=formation_gain,
            )
        )

    return LaunchDescription(actions)
