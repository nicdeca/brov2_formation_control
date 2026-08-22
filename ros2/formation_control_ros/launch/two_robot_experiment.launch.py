"""Two-BlueROV formation experiment with automatic initialization.

This launch reuses the same leader/follower controllers as the validated
three-robot setup.  It exposes progressively more demanding experiment
profiles through the formation library used by run_two_robot_experiment.py.

IMPORTANT FOR HARDWARE:
Before wet testing, verify the initialization positions, workspace bounds,
camera extrinsics/FoV, state frames, and PX4 command normalization.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

PHASE_TOPIC = "/formation_control/experiment_phase"
FORMATION_TOPIC = "/formation_control/desired_formation"

# Core-NWU initialization geometry.
INITIAL_LEADER_POSITION = [-1.20, 1.15, -95.20]
INITIAL_FOLLOWER_POSITION = [-1.90, 2.95, -95.20]

# Parent-minus-follower vector.
INITIAL_RELATIVE = [0.70, -1.80, 0.00]

FORMATION_NAMES = [
    # Normal experiment library.
    "pair_nominal",
    "pair_far",
    "pair_close",
    "pair_high",
    # Challenging/adaptive-domain demonstrations.
    "pair_range_far_edge",
    "pair_range_close_edge",
    "pair_fov_edge",
]

# Parent-minus-follower vectors in core NWU.
#
# Current sensing domains:
#   physical range:       0.5 <= d <= 3.6 m
#   conservative range:   0.8 <= d <= 3.0 m
#   conservative FoV:     alpha_h = alpha_v = 0.72
#
# pair_range_far_edge:
#   ||d|| ~= 3.18 m -> outside conservative d_max=3.0 but inside physical 3.6.
#
# pair_range_close_edge:
#   ||d|| ~= 0.70 m -> outside conservative d_min=0.8 but inside physical 0.5.
#
# pair_fov_edge:
#   intentionally large horizontal bearing.  The exact normalized image
#   coordinate depends on vehicle attitude/camera extrinsics, so this target
#   is meant to excite horizontal FoV relaxation rather than encode a fixed
#   image-coordinate value.
FORMATION_RELATIVE_POSITIONS = [
    0.70,
    -1.80,
    0.00,  # pair_nominal
    1.00,
    -2.25,
    0.00,  # pair_far
    0.40,
    -1.30,
    0.00,  # pair_close
    0.70,
    -1.80,
    -0.20,  # pair_high
    0.90,
    -3.05,
    0.00,  # pair_range_far_edge, norm ~= 3.18
    0.15,
    -0.68,
    0.00,  # pair_range_close_edge, norm ~= 0.70
    1.30,
    -1.55,
    0.00,  # pair_fov_edge
]


def _phase_manager_setup(context):
    leader = LaunchConfiguration("leader").perform(context)
    follower = LaunchConfiguration("follower").perform(context)

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
                }
            ],
        )
    ]


def generate_launch_description() -> LaunchDescription:
    leader = LaunchConfiguration("leader")
    follower = LaunchConfiguration("follower")
    dt = LaunchConfiguration("dt")
    dry_run = LaunchConfiguration("dry_run")
    leader_reference_mode = LaunchConfiguration("leader_reference_mode")
    leader_robot_configuration = LaunchConfiguration(
        "leader_robot_configuration"
    )
    follower_robot_configuration = LaunchConfiguration(
        "follower_robot_configuration"
    )
    workspace_barrier_enabled = LaunchConfiguration("workspace_barrier_enabled")
    workspace_adaptive = LaunchConfiguration("workspace_adaptive")
    position_gain = LaunchConfiguration("position_gain")
    formation_gain = LaunchConfiguration("formation_gain")
    # Command filter gain
    virtual_linear_gain = LaunchConfiguration("virtual_linear_gain")
    virtual_angular_gain = LaunchConfiguration("virtual_angular_gain")

    command_filter_linear_bandwidth = LaunchConfiguration(
        "command_filter_linear_bandwidth"
    )
    command_filter_angular_bandwidth = LaunchConfiguration(
        "command_filter_angular_bandwidth"
    )

    alpha_gain = LaunchConfiguration("alpha_gain")

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

    return LaunchDescription(
        [
            DeclareLaunchArgument("leader", default_value="itrl_rov_1"),
            # Keep default robot IDs ordered in two-robot experiments.
            DeclareLaunchArgument("follower", default_value="itrl_rov_2"),
            DeclareLaunchArgument("dt", default_value="0.02"),
            DeclareLaunchArgument("dry_run", default_value="true"),
            DeclareLaunchArgument(
                "leader_reference_mode",
                default_value="stationary",
            ),
            DeclareLaunchArgument(
                "leader_robot_configuration",
                default_value="standard",
                description=(
                    "Dynamics preset for the leader: "
                    "'gazebo', 'standard', or 'heavy_tube'."
                ),
            ),
            DeclareLaunchArgument(
                "follower_robot_configuration",
                default_value="standard",
                description=(
                    "Dynamics preset for the follower: "
                    "'gazebo', 'standard', or 'heavy_tube'."
                ),
            ),
            DeclareLaunchArgument(
                "position_gain",
                default_value="2.0",
                description="Leader position-potential gain.",
            ),
            DeclareLaunchArgument(
                "formation_gain",
                default_value="2.0",
                description="Follower relative-position-potential gain.",
            ),
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
            DeclareLaunchArgument(
                "alpha_gain",
                default_value="1.8",
            ),
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
            Node(
                package="formation_control_ros",
                executable="follower_controller",
                namespace=follower,
                name="controller",
                output="screen",
                parameters=[
                    common,
                    {
                        "robot_name": follower,
                        "robot_configuration": follower_robot_configuration,
                        "parent_robot_name": leader,
                        "desired_relative_position": INITIAL_RELATIVE,
                        "initialization_position": INITIAL_FOLLOWER_POSITION,
                        "experiment_phase_topic": PHASE_TOPIC,
                        "desired_formation_topic": FORMATION_TOPIC,
                        "formation_names": FORMATION_NAMES,
                        "formation_relative_positions": (FORMATION_RELATIVE_POSITIONS),
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
                namespace=follower,
                name="heartbeat",
                output="screen",
            ),
        ]
    )
