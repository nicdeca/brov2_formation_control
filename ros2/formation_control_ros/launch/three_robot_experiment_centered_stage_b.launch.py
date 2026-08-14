"""Three-BlueROV formation experiment with automatic initialization."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

PHASE_TOPIC = "/formation_control/experiment_phase"
FORMATION_TOPIC = "/formation_control/desired_formation"

# Shifted slightly toward the tank center while preserving the nominal
# parent-minus-follower vectors.
INITIAL_LEADER_POSITION = [-1.20, 1.15, -95.20]
INITIAL_LEFT_POSITION = [-0.50, 2.95, -95.20]
INITIAL_RIGHT_POSITION = [-1.90, 2.95, -95.20]

INITIAL_LEFT = [-0.70, -1.80, 0.00]
INITIAL_RIGHT = [0.70, -1.80, 0.00]

FORMATION_NAMES = [
    "triangle_nominal",
    "triangle_wide",
    "triangle_high",
]
LEFT_FORMATIONS = [
    -0.70,
    -1.80,
    0.00,
    -0.90,
    -2.10,
    0.00,
    -0.70,
    -1.80,
    0.25,
]
RIGHT_FORMATIONS = [
    0.70,
    -1.80,
    0.00,
    0.90,
    -2.10,
    0.00,
    0.70,
    -1.80,
    -0.25,
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


def generate_launch_description() -> LaunchDescription:
    leader = LaunchConfiguration("leader")
    follower_left = LaunchConfiguration("follower_left")
    follower_right = LaunchConfiguration("follower_right")
    dt = LaunchConfiguration("dt")
    dry_run = LaunchConfiguration("dry_run")
    leader_reference_mode = LaunchConfiguration("leader_reference_mode")
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
            DeclareLaunchArgument(
                "follower_left",
                default_value="itrl_rov_2",
            ),
            DeclareLaunchArgument(
                "follower_right",
                default_value="itrl_rov_3",
            ),
            DeclareLaunchArgument("dt", default_value="0.02"),
            DeclareLaunchArgument("dry_run", default_value="true"),
            DeclareLaunchArgument(
                "leader_reference_mode",
                default_value="stationary",
            ),
            DeclareLaunchArgument(
                "workspace_barrier_enabled",
                default_value="true",
                description=(
                    "Enable the pool-workspace barrier on leader and followers."
                ),
            ),
            DeclareLaunchArgument(
                "workspace_adaptive",
                default_value="true",
                description=(
                    "Allow the conservative workspace to relax toward the "
                    "physical safe bounds."
                ),
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
                namespace=follower_left,
                name="controller",
                output="screen",
                parameters=[
                    common,
                    {
                        "robot_name": follower_left,
                        "parent_robot_name": leader,
                        "desired_relative_position": INITIAL_LEFT,
                        "initialization_position": INITIAL_LEFT_POSITION,
                        "experiment_phase_topic": PHASE_TOPIC,
                        "desired_formation_topic": FORMATION_TOPIC,
                        "formation_names": FORMATION_NAMES,
                        "formation_relative_positions": LEFT_FORMATIONS,
                        "adaptive": True,
                    },
                ],
            ),
            Node(
                package="formation_control_ros",
                executable="offboard_heartbeat_wrench",
                namespace=follower_left,
                name="heartbeat",
                output="screen",
            ),
            Node(
                package="formation_control_ros",
                executable="follower_controller",
                namespace=follower_right,
                name="controller",
                output="screen",
                parameters=[
                    common,
                    {
                        "robot_name": follower_right,
                        "parent_robot_name": leader,
                        "desired_relative_position": INITIAL_RIGHT,
                        "initialization_position": INITIAL_RIGHT_POSITION,
                        "experiment_phase_topic": PHASE_TOPIC,
                        "desired_formation_topic": FORMATION_TOPIC,
                        "formation_names": FORMATION_NAMES,
                        "formation_relative_positions": RIGHT_FORMATIONS,
                        "adaptive": True,
                    },
                ],
            ),
            Node(
                package="formation_control_ros",
                executable="offboard_heartbeat_wrench",
                namespace=follower_right,
                name="heartbeat",
                output="screen",
            ),
        ]
    )
