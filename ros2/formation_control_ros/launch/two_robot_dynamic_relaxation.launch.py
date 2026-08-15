"""Two-BlueROV dynamic sensing-domain relaxation paper experiment.

Purpose
-------
Trigger sensing-domain relaxation through a fast transient and limited follower
actuation, without placing the desired formation close to or in conflict with
the conservative sensing boundary.

The desired parent-minus-follower vector has norm about 2.36 m, comfortably
inside the conservative upper range d_max^c = 3.0 m.  The leader then executes
a short, high-speed maneuver approximately along the separation direction.
The follower is deliberately strongly derated.

Expected mechanism:

    aggressive leader motion
        + reduced follower actuation
        -> follower temporarily cannot preserve conservative range
        -> actual distance crosses d_max^c = 3.0 m
        -> range relaxation becomes nonzero
        -> physical range d_max = 3.6 m remains satisfied
        -> follower catches up and relaxation recovers

The desired formation itself remains conservative throughout.
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
INITIAL_RELATIVE = [0.70, -1.80, 0.00]

FORMATION_NAMES = [
    "pair_nominal",
    "pair_dynamic_stress",
]

# ||[0.85, -2.20, 0]|| ~= 2.358 m.
# This is intentionally far from d_max^c = 3.0 m, leaving about 0.64 m of
# conservative range margin in steady state.
FORMATION_RELATIVE_POSITIONS = [
     0.70, -1.80, 0.00,
     0.85, -2.20, 0.00,
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

    leader_velocity_command_bandwidth = LaunchConfiguration(
        "leader_velocity_command_bandwidth"
    )

    leader_thrust_derating = LaunchConfiguration("leader_thrust_derating")
    follower_thrust_derating = LaunchConfiguration("follower_thrust_derating")

    workspace_barrier_enabled = LaunchConfiguration(
        "workspace_barrier_enabled"
    )
    workspace_adaptive = LaunchConfiguration("workspace_adaptive")

    common = {
        "dt": dt,
        "state_source": "px4",
        "control_space": "thruster",
        "dry_run": dry_run,

        # Damped controller tuning from the validated paper simulations.
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

        # Workspace adaptation remains active; startup workspace relaxation is
        # acceptable and unrelated to the sensing-domain event of interest.
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

    return LaunchDescription(
        [
            DeclareLaunchArgument("leader", default_value="itrl_rov_1"),
            DeclareLaunchArgument("follower", default_value="itrl_rov_2"),
            DeclareLaunchArgument("dt", default_value="0.02"),
            DeclareLaunchArgument("dry_run", default_value="true"),
            DeclareLaunchArgument(
                "leader_reference_mode",
                default_value="stationary",
            ),

            DeclareLaunchArgument("position_gain", default_value="2.0"),
            DeclareLaunchArgument("formation_gain", default_value="2.0"),
            DeclareLaunchArgument("virtual_linear_gain", default_value="1.0"),
            DeclareLaunchArgument("virtual_angular_gain", default_value="1.2"),
            DeclareLaunchArgument(
                "command_filter_linear_bandwidth",
                default_value="3.0",
            ),
            DeclareLaunchArgument(
                "command_filter_angular_bandwidth",
                default_value="4.0",
            ),
            DeclareLaunchArgument("alpha_gain", default_value="1.5"),

            # Make the user cmd_vel reference generator respond rapidly enough
            # that a short pulse actually reaches the leader controller.
            DeclareLaunchArgument(
                "leader_velocity_command_bandwidth",
                default_value="2.0",
                description=(
                    "Bandwidth of the leader manual velocity-reference filter."
                ),
            ),

            # Explicit asymmetry in physical authority.
            DeclareLaunchArgument(
                "leader_thrust_derating",
                default_value="1.0",
            ),
            DeclareLaunchArgument(
                "follower_thrust_derating",
                default_value="0.20",
                description=(
                    "Strongly reduced follower thruster authority for the "
                    "dynamic relaxation stress test."
                ),
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
                        "reference_mode": leader_reference_mode,
                        "experiment_phase_topic": PHASE_TOPIC,
                        "initialization_position": INITIAL_LEADER_POSITION,
                        "position_gain": ParameterValue(
                            position_gain,
                            value_type=float,
                        ),
                        "velocity_command_bandwidth": ParameterValue(
                            leader_velocity_command_bandwidth,
                            value_type=float,
                        ),
                        "thrust_derating": ParameterValue(
                            leader_thrust_derating,
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
                        "parent_robot_name": leader,
                        "desired_relative_position": INITIAL_RELATIVE,
                        "initialization_position": INITIAL_FOLLOWER_POSITION,
                        "experiment_phase_topic": PHASE_TOPIC,
                        "desired_formation_topic": FORMATION_TOPIC,
                        "formation_names": FORMATION_NAMES,
                        "formation_relative_positions": (
                            FORMATION_RELATIVE_POSITIONS
                        ),
                        "formation_gain": ParameterValue(
                            formation_gain,
                            value_type=float,
                        ),
                        "thrust_derating": ParameterValue(
                            follower_thrust_derating,
                            value_type=float,
                        ),

                        "adaptive": True,
                        "d_min": 0.5,
                        "d_max": 3.6,
                        "d_min_conservative": 0.8,
                        "d_max_conservative": 3.0,
                        "alpha_h_conservative": 0.72,
                        "alpha_v_conservative": 0.72,
                        "relaxation_recovery_gain": 0.8,
                        "relaxation_domain_margin_ratio": 0.10,
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
