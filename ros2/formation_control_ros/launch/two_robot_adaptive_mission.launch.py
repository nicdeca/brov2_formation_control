"""Two-BlueROV SITL mission for the adaptive sensing-domain paper demo.

This launch is intentionally separate from ``two_robot_experiment.launch.py``.
It reproduces the tight sensing domains and controller tuning used by the
pure-Python Example 12 without changing the normal hardware experiment.

Directed sensing edge:
    itrl_rov_2 -> itrl_rov_1

The formation references are parent-minus-follower vectors in the pool-aligned core NWU frame.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


PHASE_TOPIC = "/formation_control/experiment_phase"
FORMATION_TOPIC = "/formation_control/desired_formation"

# Same two-robot SITL initialization geometry used by the existing experiment.
INITIAL_LEADER_POSITION = [2.675, 0.050, -0.775]

# Example-12 mission formation A after rigid rotation into the pool frame:
#     d_21 = p_1 - p_2 = [-1.80, -0.70, 0.00].
INITIAL_RELATIVE = [-1.800, -0.700, 0.000]
INITIAL_FOLLOWER_POSITION = [4.475, 0.750, -0.775]

# Exact parent-minus-follower formation sequence from Example 12.
FORMATION_NAMES = [
    "adaptive_A",
    "adaptive_B",
    "adaptive_C",
    "adaptive_D",
]

FORMATION_RELATIVE_POSITIONS = [
    -1.80, -0.70,  0.00,  # A, t = 0 s (and recovery after t = 80 s)
    -1.65, -1.45, -0.40,  # B, t = 18 s
    -1.65,  1.45,  0.40,  # C, t = 38 s
    -2.30, -0.55, -0.30,  # D, t = 60 s
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
    gazebo_timer = LaunchConfiguration("gazebo_timer")

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

    d_max_conservative = LaunchConfiguration("d_max_conservative")
    alpha_h_conservative = LaunchConfiguration("alpha_h_conservative")
    alpha_v_conservative = LaunchConfiguration("alpha_v_conservative")

    relaxation_recovery_gain = LaunchConfiguration(
        "relaxation_recovery_gain"
    )
    relaxation_barrier_gain = LaunchConfiguration(
        "relaxation_barrier_gain"
    )
    relaxation_domain_margin_ratio = LaunchConfiguration(
        "relaxation_domain_margin_ratio"
    )
    relaxation_activation_on_ratio = LaunchConfiguration(
        "relaxation_activation_on_ratio"
    )
    relaxation_activation_off_ratio = LaunchConfiguration(
        "relaxation_activation_off_ratio"
    )

    workspace_barrier_enabled = LaunchConfiguration(
        "workspace_barrier_enabled"
    )
    workspace_adaptive = LaunchConfiguration("workspace_adaptive")

    common = {
        "dt": dt,
        "use_sim_time": ParameterValue(gazebo_timer, value_type=bool),
        "state_source": "px4",
        "control_space": "thruster",
        # This is a dedicated SITL launch: never use auto/real-robot presets.
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

        # Keep the normal pool constraints active.  The mission runner uses a
        # tank-safe common translation so these should remain nonactive and the
        # paper demonstration is driven by the sensing-domain relaxation.
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

    follower_task = {
        "robot_name": follower,
        "parent_robot_name": leader,
        "desired_relative_position": INITIAL_RELATIVE,
        "initialization_position": INITIAL_FOLLOWER_POSITION,
        "experiment_phase_topic": PHASE_TOPIC,
        "desired_formation_topic": FORMATION_TOPIC,
        "formation_names": FORMATION_NAMES,
        "formation_relative_positions": FORMATION_RELATIVE_POSITIONS,

        # Example-12 formation/sensing tuning.
        "formation_gain": ParameterValue(
            formation_gain,
            value_type=float,
        ),
        "image_horizontal_gain": 0.8,
        "image_vertical_gain": 0.8,

        "d_min": 0.5,
        "d_max": 3.6,
        "d_min_conservative": 0.8,
        "d_max_conservative": ParameterValue(
            d_max_conservative,
            value_type=float,
        ),
        "alpha_h_conservative": ParameterValue(
            alpha_h_conservative,
            value_type=float,
        ),
        "alpha_v_conservative": ParameterValue(
            alpha_v_conservative,
            value_type=float,
        ),
        "horizontal_half_angle_deg": 45.0,
        "vertical_half_angle_deg": 30.0,

        "collision_barrier_weight": 0.18,
        "range_barrier_weight": 0.18,
        "horizontal_fov_barrier_weight": 0.25,
        "vertical_fov_barrier_weight": 0.25,

        "adaptive": True,
        "relaxation_recovery_gain": ParameterValue(
            relaxation_recovery_gain,
            value_type=float,
        ),
        "relaxation_barrier_gain": ParameterValue(
            relaxation_barrier_gain,
            value_type=float,
        ),
        "relaxation_domain_margin_ratio": ParameterValue(
            relaxation_domain_margin_ratio,
            value_type=float,
        ),
        "relaxation_activation_on_ratio": ParameterValue(
            relaxation_activation_on_ratio,
            value_type=float,
        ),
        "relaxation_activation_off_ratio": ParameterValue(
            relaxation_activation_off_ratio,
            value_type=float,
        ),
        "use_parent_velocity_in_clf": False,
    }

    actions = [
        DeclareLaunchArgument("leader", default_value="itrl_rov_1"),
        DeclareLaunchArgument("follower", default_value="itrl_rov_2"),
        DeclareLaunchArgument("dt", default_value="0.02"),
        DeclareLaunchArgument("dry_run", default_value="true"),
        DeclareLaunchArgument(
            "gazebo_timer",
            default_value="false",
            description=(
                "Use Gazebo /clock for the leader/follower controller timers."
            ),
        ),

        # The pure-Python mission uses the canonical BlueROV2 controller
        # defaults: virtual gains 0.55/0.80, filter bandwidths 3/4, alpha 0.8.
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

        # Tight sensing domains from Example 12.
        DeclareLaunchArgument("d_max_conservative", default_value="2.4"),
        DeclareLaunchArgument("alpha_h_conservative", default_value="0.45"),
        DeclareLaunchArgument("alpha_v_conservative", default_value="0.45"),

        # Final smooth-adaptation tuning used for the mission.
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
            namespace=leader,
            name="controller",
            output="screen",
            parameters=[
                common,
                {
                    "robot_name": leader,
                    "reference_mode": "velocity",
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

    return LaunchDescription(actions)
