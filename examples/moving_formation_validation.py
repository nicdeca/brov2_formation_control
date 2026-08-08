"""Long-horizon validation of a moving seven-BlueROV formation.

All seven vehicles use the same command-filtered, actuator-constrained
backstepping CLF-QP dynamics controller.

The leader differs only in the configuration potential and feedforward input:
it tracks either

1. a full translational trajectory supplying position, velocity, acceleration;
2. an inertial velocity command converted to a smooth full reference by a
   first-order velocity command filter plus position integration.

Followers retain the camera-constrained adaptive formation controller.
The sensing convention is ``observer -> parent``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from formation_control.actuation import BlueROV2HeavyThrusterAllocation
from formation_control.constraints import (
    DistanceDomain,
    FieldOfViewDomain,
    HorizontalFieldOfViewConstraint,
    MaximumDistanceConstraint,
    MinimumDistanceConstraint,
    NormalizedImagePoint,
    VerticalFieldOfViewConstraint,
    evaluate_sensing_constraint_kinematics,
)
from formation_control.control import (
    BlueROV2ControlSpace,
    BlueROV2ControllerDesign,
    FunnelRelaxationInfeasibleError,
    FunnelRelaxationPolicy,
    build_bluerov2_controller_design,
)
from formation_control.control.bluerov2_leader import (
    BlueROV2LeaderController,
    LeaderTrajectorySample,
)
from formation_control.geometry import (
    PinholeCamera,
    quaternion_from_roll_pitch_yaw,
    rotation_matrix_from_quaternion,
)
from formation_control.graphs import DirectedSensingGraph
from formation_control.models import BlueROV2Model
from formation_control.simulation.realism import (
    RealismConfig,
    build_perturbed_plant_model,
    delay_steps,
    delayed_noisy_parent_position,
    noisy_twist_state,
)
from formation_control.potentials import (
    AdaptiveConstraintBarrierPotential,
    ConstraintBarrierPotential,
    EdgePotential,
    ImageCenteringPotential,
    RelativePositionPotential,
)
from formation_control.simulation import (
    FormationReference,
    FormationScenario,
    FormationTrajectory,
    RK4Integrator,
)
from formation_control.simulation.leader_references import (
    SmoothSpatialTrajectoryReference,
    VelocityCommandReferenceFilter,
)
from formation_control.visualization import (
    BlueROV2HeavyVisualGeometry,
    animate_formation_3d,
    apply_visualization_style,
    save_animation,
    save_figure,
)

MotionMode = Literal["velocity_command", "trajectory"]


@dataclass(frozen=True)
class EdgeBarrierTemplates:
    collision: AdaptiveConstraintBarrierPotential[np.ndarray]
    sensing_range: AdaptiveConstraintBarrierPotential[np.ndarray]
    horizontal_fov: AdaptiveConstraintBarrierPotential[NormalizedImagePoint]
    vertical_fov: AdaptiveConstraintBarrierPotential[NormalizedImagePoint]


@dataclass(frozen=True)
class MovingFormationResult:
    scenario: FormationScenario
    trajectory: FormationTrajectory
    camera: PinholeCamera
    distance_domain: DistanceDomain
    fov_domain: FieldOfViewDomain
    leader_reference_position: np.ndarray
    leader_reference_velocity: np.ndarray
    leader_reference_acceleration: np.ndarray
    formation_error_norm: np.ndarray
    normalized_relaxation: np.ndarray
    image_history: np.ndarray
    required_slack: np.ndarray
    actuation_margin: np.ndarray
    controller_times: np.ndarray
    fallback_times: np.ndarray
    allocation: BlueROV2HeavyThrusterAllocation
    realism_enabled: bool
    random_seed: int


def look_at_quaternion(
    relative_position_inertial: np.ndarray,
    *,
    yaw_offset: float = 0.0,
    pitch_offset: float = 0.0,
) -> np.ndarray:
    relative = np.asarray(relative_position_inertial, dtype=float)
    horizontal_norm = float(np.hypot(relative[0], relative[1]))
    yaw = float(np.arctan2(relative[1], relative[0]) + yaw_offset)
    pitch = float(
        -np.arctan2(relative[2], horizontal_norm) + pitch_offset
    )
    return quaternion_from_roll_pitch_yaw(
        roll=0.0,
        pitch=pitch,
        yaw=yaw,
    )


def build_balanced_tree_scenario() -> FormationScenario:
    graph = DirectedSensingGraph.from_edges(
        7,
        (
            (1, 0),
            (2, 0),
            (3, 1),
            (4, 1),
            (5, 2),
            (6, 2),
        ),
        root=0,
    ).require_rooted_tree()

    offsets = np.array(
        [
            [0.00, 0.00, 0.00],
            [-1.90, -0.75, -0.15],
            [-1.90, 0.75, 0.15],
            [-3.75, -1.45, -0.30],
            [-3.75, -0.35, -0.05],
            [-3.75, 0.35, 0.05],
            [-3.75, 1.45, 0.30],
        ],
        dtype=float,
    )
    reference = FormationReference(offsets=offsets)

    root_position = np.array([0.0, 0.0, -1.5])
    perturbation = np.array(
        [
            [0.00, 0.00, 0.00],
            [-0.12, 0.10, 0.05],
            [0.10, -0.08, -0.04],
            [-0.08, 0.08, 0.04],
            [0.09, -0.07, 0.02],
            [-0.07, 0.06, -0.03],
            [0.08, -0.08, 0.03],
        ],
        dtype=float,
    )
    return FormationScenario(
        graph=graph,
        reference=reference,
        initial_positions=root_position + offsets + perturbation,
    )



def build_readme_demo_scenario() -> FormationScenario:
    """Seven-robot convergence demo with visibly displaced initial positions.

    The desired geometry uses the same balanced directed tree as the long
    validation examples.  The initial positions are deliberately asymmetric
    and substantially displaced from the desired formation, while remaining
    inside the physical range limits so that the animation emphasizes
    formation acquisition rather than reference tracking.
    """
    graph = DirectedSensingGraph.from_edges(
        7,
        (
            (1, 0),
            (2, 0),
            (3, 1),
            (4, 1),
            (5, 2),
            (6, 2),
        ),
        root=0,
    ).require_rooted_tree()

    desired_offsets = np.array(
        [
            [0.00, 0.00, 0.00],
            [-1.75, -0.70, -0.10],
            [-1.75, 0.70, 0.10],
            [-3.45, -1.35, -0.20],
            [-3.45, -0.35, -0.05],
            [-3.45, 0.35, 0.05],
            [-3.45, 1.35, 0.20],
        ],
        dtype=float,
    )
    reference = FormationReference(offsets=desired_offsets)

    root_position = np.array([0.0, 0.0, -1.5])

    # Deliberately nonuniform displacements.  They are large enough to make
    # convergence visually obvious but keep each follower close enough to its
    # parent for the forward-looking camera model to start inside the physical
    # sensing domain.
    initial_offsets = np.array(
        [
            [0.00, 0.00, 0.00],
            [-2.35, -0.20, 0.18],
            [-1.30, 1.20, -0.18],
            [-4.30, -0.80, 0.30],
            [-2.80, -1.05, -0.28],
            [-4.05, 0.05, 0.24],
            [-2.95, 1.85, -0.22],
        ],
        dtype=float,
    )

    return FormationScenario(
        graph=graph,
        reference=reference,
        initial_positions=root_position + initial_offsets,
    )

def build_initial_states(scenario: FormationScenario) -> np.ndarray:
    states = np.zeros((scenario.n_agents, 13), dtype=float)

    leader = scenario.graph.root
    states[leader, :3] = scenario.initial_positions[leader]
    states[leader, 3:7] = quaternion_from_roll_pitch_yaw(
        roll=0.0,
        pitch=0.0,
        yaw=0.0,
    )

    for edge in scenario.graph:
        observer = edge.observer
        target = edge.target
        relative = (
            scenario.initial_positions[target]
            - scenario.initial_positions[observer]
        )
        sign = -1.0 if observer % 2 else 1.0
        depth = scenario.graph.depth(observer)
        states[observer, :3] = scenario.initial_positions[observer]
        states[observer, 3:7] = look_at_quaternion(
            relative,
            yaw_offset=sign * np.deg2rad(3.0 + depth),
            pitch_offset=-sign * np.deg2rad(1.5),
        )

    return states


def build_domains() -> tuple[DistanceDomain, FieldOfViewDomain]:
    return (
        DistanceDomain(
            d_min=0.5,
            d_max=3.6,
            d_min_conservative=0.8,
            d_max_conservative=3.0,
        ),
        FieldOfViewDomain(
            alpha_h_conservative=0.72,
            alpha_v_conservative=0.72,
        ),
    )


def build_camera() -> PinholeCamera:
    return PinholeCamera.from_degrees(
        horizontal_half_angle=45.0,
        vertical_half_angle=30.0,
    )


def build_controller_design(
    model: BlueROV2Model,
    allocation: BlueROV2HeavyThrusterAllocation,
    *,
    control_space: BlueROV2ControlSpace,
    slack_linear_penalty: float,
    slack_quadratic_penalty: float,
    virtual_linear_speed_limit: float,
    virtual_angular_speed_limit: float,
) -> BlueROV2ControllerDesign:
    return build_bluerov2_controller_design(
        model,
        allocation,
        control_space=control_space,
        slack_linear_penalty=slack_linear_penalty,
        slack_penalty=slack_quadratic_penalty,
        virtual_velocity_norm_limits=np.array(
            [
                virtual_linear_speed_limit,
                virtual_angular_speed_limit,
            ],
            dtype=float,
        ),
    )


def build_relaxation_policy(
    distance_domain: DistanceDomain,
    fov_domain: FieldOfViewDomain,
    *,
    recovery_gain: float,
    domain_margin_ratio: float,
) -> FunnelRelaxationPolicy:
    return FunnelRelaxationPolicy(
        maximum_enlargement=np.array(
            [
                distance_domain.collision_enlargement_max,
                distance_domain.range_enlargement_max,
                fov_domain.horizontal_enlargement_max,
                fov_domain.vertical_enlargement_max,
            ],
            dtype=float,
        ),
        recovery_gain=recovery_gain,
        domain_margin_ratio=domain_margin_ratio,
        minimum_constraint_margin=1e-5,
    )


def build_barrier_templates(
    scenario: FormationScenario,
    distance_domain: DistanceDomain,
    fov_domain: FieldOfViewDomain,
    observer: int,
    target: int,
) -> EdgeBarrierTemplates:
    desired_relative = scenario.desired_relative_position(observer, target)
    desired_image = NormalizedImagePoint(0.0, 0.0)

    return EdgeBarrierTemplates(
        collision=AdaptiveConstraintBarrierPotential(
            constraint=MinimumDistanceConstraint(
                distance_domain.d_min_conservative
            ),
            reference_state=desired_relative,
            weight=0.18,
        ),
        sensing_range=AdaptiveConstraintBarrierPotential(
            constraint=MaximumDistanceConstraint(
                distance_domain.d_max_conservative
            ),
            reference_state=desired_relative,
            weight=0.18,
        ),
        horizontal_fov=AdaptiveConstraintBarrierPotential(
            constraint=HorizontalFieldOfViewConstraint(
                fov_domain.alpha_h_conservative
            ),
            reference_state=desired_image,
            weight=0.25,
        ),
        vertical_fov=AdaptiveConstraintBarrierPotential(
            constraint=VerticalFieldOfViewConstraint(
                fov_domain.alpha_v_conservative
            ),
            reference_state=desired_image,
            weight=0.25,
        ),
    )


def build_edge_potential(
    scenario: FormationScenario,
    camera: PinholeCamera,
    distance_domain: DistanceDomain,
    fov_domain: FieldOfViewDomain,
    observer: int,
    target: int,
    templates: EdgeBarrierTemplates,
    enlargement: np.ndarray,
    *,
    adaptive: bool,
) -> EdgePotential:
    desired_relative = scenario.desired_relative_position(observer, target)
    desired_image = NormalizedImagePoint(0.0, 0.0)

    if adaptive:
        collision = templates.collision.bind(float(enlargement[0]))
        sensing_range = templates.sensing_range.bind(float(enlargement[1]))
        horizontal = templates.horizontal_fov.bind(float(enlargement[2]))
        vertical = templates.vertical_fov.bind(float(enlargement[3]))
    else:
        collision = ConstraintBarrierPotential.from_reference(
            MinimumDistanceConstraint(distance_domain.d_min_conservative),
            desired_relative,
            weight=0.18,
        )
        sensing_range = ConstraintBarrierPotential.from_reference(
            MaximumDistanceConstraint(distance_domain.d_max_conservative),
            desired_relative,
            weight=0.18,
        )
        horizontal = ConstraintBarrierPotential.from_reference(
            HorizontalFieldOfViewConstraint(
                fov_domain.alpha_h_conservative
            ),
            desired_image,
            weight=0.25,
        )
        vertical = ConstraintBarrierPotential.from_reference(
            VerticalFieldOfViewConstraint(
                fov_domain.alpha_v_conservative
            ),
            desired_image,
            weight=0.25,
        )

    return EdgePotential(
        formation=RelativePositionPotential.isotropic(
            desired_relative,
            gain=1.0,
        ),
        image_centering=ImageCenteringPotential(
            horizontal_gain=0.7,
            vertical_gain=0.7,
        ),
        collision_barrier=collision,
        range_barrier=sensing_range,
        horizontal_fov_barrier=horizontal,
        vertical_fov_barrier=vertical,
        camera=camera,
    )


def inertial_linear_velocity(
    model: BlueROV2Model,
    state: np.ndarray,
) -> np.ndarray:
    _, quaternion, generalized_velocity = model.split_state(state)
    return (
        rotation_matrix_from_quaternion(quaternion)
        @ generalized_velocity[:3]
    )


def stationkeeping_wrench(
    model: BlueROV2Model,
    state: np.ndarray,
    hold_position: np.ndarray,
) -> np.ndarray:
    position, quaternion, velocity = model.split_state(state)
    rotation = rotation_matrix_from_quaternion(quaternion)
    velocity_inertial = rotation @ velocity[:3]

    force_inertial = (
        -5.0 * (position - hold_position)
        - 4.0 * velocity_inertial
    )
    force_body = rotation.T @ force_inertial
    torque_body = -1.5 * velocity[3:]
    return model.drift_wrench(state) + np.concatenate(
        (force_body, torque_body)
    )


def _record_images(
    image_history: np.ndarray,
    sample: int,
    scenario: FormationScenario,
    camera: PinholeCamera,
    states: np.ndarray,
) -> None:
    for edge in scenario.graph:
        try:
            observation = camera.observe(
                states[edge.observer, :3],
                rotation_matrix_from_quaternion(
                    states[edge.observer, 3:7]
                ),
                states[edge.target, :3],
            )
        except ValueError:
            continue
        image_history[sample, edge.observer] = np.array(
            [
                observation.image_point.alpha_h,
                observation.image_point.alpha_v,
            ]
        )


def _record_formation_error(
    error_history: np.ndarray,
    sample: int,
    scenario: FormationScenario,
    states: np.ndarray,
) -> None:
    for edge in scenario.graph:
        relative = (
            states[edge.target, :3]
            - states[edge.observer, :3]
        )
        desired = scenario.desired_relative_position(
            edge.observer,
            edge.target,
        )
        error_history[sample, edge.observer] = np.linalg.norm(
            relative - desired
        )


def simulate(
    *,
    motion: MotionMode,
    duration: float,
    dt: float,
    velocity_command: np.ndarray,
    velocity_command_bandwidth: float,
    thruster_voltage: int,
    thrust_derating: float,
    control_space: BlueROV2ControlSpace,
    adaptive: bool,
    use_parent_velocity_in_clf: bool,
    virtual_linear_speed_limit: float,
    virtual_angular_speed_limit: float,
    relaxation_recovery_gain: float,
    relaxation_domain_margin_ratio: float,
    slack_linear_penalty: float,
    slack_quadratic_penalty: float,
    realism: RealismConfig | None = None,
    random_seed: int = 7,
    scenario: FormationScenario | None = None,
) -> MovingFormationResult:
    if scenario is None:
        scenario = build_balanced_tree_scenario()
    camera = build_camera()
    distance_domain, fov_domain = build_domains()
    model = BlueROV2Model()
    rng = np.random.default_rng(random_seed)
    if realism is None:
        plant_model = model
        relative_delay_steps = 0
    else:
        plant_model = build_perturbed_plant_model(
            model,
            config=realism,
            rng=rng,
        )
        relative_delay_steps = delay_steps(
            realism.relative_measurement_delay,
            dt,
        )

    allocation = BlueROV2HeavyThrusterAllocation.default_45deg(
        voltage=thruster_voltage,
        derating=thrust_derating,
    )
    controller_design = build_controller_design(
        model,
        allocation,
        control_space=control_space,
        slack_linear_penalty=slack_linear_penalty,
        slack_quadratic_penalty=slack_quadratic_penalty,
        virtual_linear_speed_limit=virtual_linear_speed_limit,
        virtual_angular_speed_limit=virtual_angular_speed_limit,
    )
    follower_controller = controller_design.agent_controller

    states = build_initial_states(scenario)
    state_history: list[np.ndarray] = [states.copy()]
    leader = scenario.graph.root
    leader_controller = BlueROV2LeaderController.from_initial_state(
        follower_controller.dynamics_controller,
        model,
        states[leader],
        position_gain=np.diag([1.0, 1.0, 1.0]),
        attitude_gain=1.0,
    )

    analytic_trajectory = None
    velocity_reference_model = None
    velocity_reference_state = None

    if motion == "trajectory":
        analytic_trajectory = SmoothSpatialTrajectoryReference(
            initial_position=states[leader, :3],
            forward_speed=0.30,
            lateral_amplitude=1.20,
            vertical_amplitude=0.35,
            lateral_frequency=0.045,
            vertical_frequency=0.030,
            time_constant=6.0,
        )
        initial_leader_reference = analytic_trajectory.evaluate(0.0)
    else:
        velocity_reference_model = VelocityCommandReferenceFilter(
            bandwidth=velocity_command_bandwidth,
        )
        velocity_reference_state = velocity_reference_model.initialize(
            states[leader, :3],
            velocity=np.zeros(3),
        )
        initial_leader_reference = velocity_reference_model.evaluate(
            velocity_reference_state,
            velocity_command,
        )

    leader_filter = leader_controller.initialize_filter(
        state=states[leader],
        reference=initial_leader_reference,
    )

    relaxation = build_relaxation_policy(
        distance_domain,
        fov_domain,
        recovery_gain=relaxation_recovery_gain,
        domain_margin_ratio=relaxation_domain_margin_ratio,
    )
    enabled_relaxation = np.ones(4, dtype=bool)
    relaxation_state = np.zeros((scenario.n_agents, 4), dtype=float)
    fallback_mode = np.zeros(scenario.n_agents, dtype=bool)
    fallback_positions = states[:, :3].copy()
    fallback_times = np.full(scenario.n_agents, np.nan)

    templates = {
        edge.observer: build_barrier_templates(
            scenario,
            distance_domain,
            fov_domain,
            edge.observer,
            edge.target,
        )
        for edge in scenario.graph
    }

    if adaptive:
        for edge in scenario.graph:
            kinematics = evaluate_sensing_constraint_kinematics(
                model,
                camera,
                distance_domain,
                fov_domain,
                states[edge.observer],
                states[edge.target],
            )
            projected, _ = relaxation.project_to_current_domain(
                relaxation_state[edge.observer],
                kinematics.values,
                enabled=enabled_relaxation,
            )
            relaxation_state[edge.observer] = projected

    follower_filters: dict[int, np.ndarray] = {}
    for edge in scenario.graph:
        enlargement = relaxation.enlargement(
            relaxation_state[edge.observer]
        )
        potential = build_edge_potential(
            scenario,
            camera,
            distance_domain,
            fov_domain,
            edge.observer,
            edge.target,
            templates[edge.observer],
            enlargement,
            adaptive=adaptive,
        )
        follower_filters[edge.observer] = (
            follower_controller.initialize_filter(
                follower_state=states[edge.observer],
                parent_position=states[edge.target, :3],
                edge_potential=potential,
            )
        )

    plant_integrator = RK4Integrator()
    filter_integrator = RK4Integrator()
    reference_integrator = RK4Integrator()

    steps = int(np.ceil(duration / dt))
    times = np.arange(steps + 1, dtype=float) * dt

    positions = np.empty((steps + 1, scenario.n_agents, 3))
    quaternions = np.empty((steps + 1, scenario.n_agents, 4))
    inertial_velocities = np.empty((steps + 1, scenario.n_agents, 3))
    controls = np.zeros((steps, scenario.n_agents, 8))
    reference_positions = np.empty((steps + 1, 3))
    reference_velocities = np.empty((steps + 1, 3))
    reference_accelerations = np.empty((steps + 1, 3))
    formation_error_norm = np.full(
        (steps + 1, scenario.n_agents),
        np.nan,
    )
    normalized_relaxation = np.zeros(
        (steps + 1, scenario.n_agents, 4),
        dtype=float,
    )
    image_history = np.full(
        (steps + 1, scenario.n_agents, 2),
        np.nan,
    )
    required_slack = np.full(
        (steps, scenario.n_agents),
        np.nan,
    )
    actuation_margin = np.full(
        (steps, scenario.n_agents),
        np.nan,
    )
    controller_times = np.full(
        (steps, scenario.n_agents),
        np.nan,
    )

    def current_leader_reference(
        sample_time: float,
    ) -> LeaderTrajectorySample:
        if analytic_trajectory is not None:
            return analytic_trajectory.evaluate(sample_time)
        assert velocity_reference_model is not None
        assert velocity_reference_state is not None
        return velocity_reference_model.evaluate(
            velocity_reference_state,
            velocity_command,
        )

    def record(sample: int) -> None:
        for agent in range(scenario.n_agents):
            position, quaternion, _ = model.split_state(states[agent])
            positions[sample, agent] = position
            quaternions[sample, agent] = quaternion
            inertial_velocities[sample, agent] = inertial_linear_velocity(
                model,
                states[agent],
            )

        reference = current_leader_reference(times[sample])
        reference_positions[sample] = reference.position
        reference_velocities[sample] = reference.velocity
        reference_accelerations[sample] = reference.acceleration
        normalized_relaxation[sample] = relaxation_state
        _record_images(
            image_history,
            sample,
            scenario,
            camera,
            states,
        )
        _record_formation_error(
            formation_error_norm,
            sample,
            scenario,
            states,
        )

    record(0)

    for step in range(steps):
        next_states = states.copy()
        next_relaxation_state = relaxation_state.copy()

        # Leader: same saturated virtual-twist / filter / CLF-QP chain.
        leader_reference = current_leader_reference(times[step])
        measured_leader_state = states[leader]
        if realism is not None:
            measured_leader_state = noisy_twist_state(
                states[leader],
                rng=rng,
                linear_std=realism.linear_velocity_noise_std,
                angular_std=realism.angular_velocity_noise_std,
            )

        start_time = perf_counter()
        leader_evaluation = leader_controller.evaluate(
            state=measured_leader_state,
            reference=leader_reference,
            filter_state=leader_filter,
        )
        controller_times[step, leader] = perf_counter() - start_time
        controls[step, leader] = (
            controller_design.representative_thruster_forces(
                leader_evaluation
            )
        )
        if leader_evaluation.required_slack is not None:
            required_slack[step, leader] = (
                leader_evaluation.required_slack
            )
        if leader_evaluation.actuation_margin is not None:
            actuation_margin[step, leader] = (
                leader_evaluation.actuation_margin
            )

        leader_filter = filter_integrator.step(
            leader_controller.dynamics_controller.command_filter,
            leader_filter,
            leader_evaluation.controller.filter_command,
            dt,
        )
        next_states[leader] = plant_integrator.step(
            plant_model,
            states[leader],
            leader_evaluation.wrench_body,
            dt,
        )

        # Followers all evaluate the same network snapshot.
        for edge in scenario.graph:
            observer = edge.observer
            target = edge.target

            if fallback_mode[observer]:
                hold_wrench = stationkeeping_wrench(
                    model,
                    states[observer],
                    fallback_positions[observer],
                )
                hold_allocation = allocation.bounded_least_squares(
                    hold_wrench
                )
                controls[step, observer] = hold_allocation.forces
                next_states[observer] = plant_integrator.step(
                    plant_model,
                    states[observer],
                    hold_allocation.achieved_wrench,
                    dt,
                )
                continue

            measured_follower_state = states[observer]
            measured_parent_position = states[target, :3]
            delayed_parent_state = states[target]

            if realism is not None:
                history_index = max(
                    0,
                    len(state_history) - 1 - relative_delay_steps,
                )
                delayed_snapshot = state_history[history_index]
                delayed_observer_state = delayed_snapshot[observer]
                delayed_parent_state = delayed_snapshot[target]

                measured_follower_state = noisy_twist_state(
                    states[observer],
                    rng=rng,
                    linear_std=realism.linear_velocity_noise_std,
                    angular_std=realism.angular_velocity_noise_std,
                )
                measured_parent_position = delayed_noisy_parent_position(
                    states[observer, :3],
                    delayed_observer_state[:3],
                    delayed_parent_state[:3],
                    rng=rng,
                    noise_std=realism.relative_position_noise_std,
                )

            measured_parent_state = delayed_parent_state.copy()
            measured_parent_state[:3] = measured_parent_position

            start_time = perf_counter()
            try:
                kinematics = evaluate_sensing_constraint_kinematics(
                    model,
                    camera,
                    distance_domain,
                    fov_domain,
                    measured_follower_state,
                    measured_parent_state,
                )

                if adaptive:
                    projected, _ = relaxation.project_to_current_domain(
                        relaxation_state[observer],
                        kinematics.values,
                        enabled=enabled_relaxation,
                    )
                    relaxation_state[observer] = projected

                enlargement = relaxation.enlargement(
                    relaxation_state[observer]
                )
                potential = build_edge_potential(
                    scenario,
                    camera,
                    distance_domain,
                    fov_domain,
                    observer,
                    target,
                    templates[observer],
                    enlargement,
                    adaptive=adaptive,
                )

                parent_velocity = None
                if use_parent_velocity_in_clf:
                    parent_velocity = inertial_linear_velocity(
                        model,
                        delayed_parent_state,
                    )

                evaluation = follower_controller.evaluate(
                    follower_state=measured_follower_state,
                    parent_position=measured_parent_position,
                    parent_linear_velocity_inertial=parent_velocity,
                    edge_potential=potential,
                    filter_state=follower_filters[observer],
                )

                relaxation_evaluation = None
                if adaptive:
                    relaxation_evaluation = relaxation.evaluate(
                        relaxation_state[observer],
                        conservative_values=kinematics.values,
                        conservative_rates=kinematics.rates,
                        enabled=enabled_relaxation,
                        sample_time=dt,
                    )
            except (FunnelRelaxationInfeasibleError, ValueError) as error:
                fallback_mode[observer] = True
                fallback_positions[observer] = states[observer, :3]
                fallback_times[observer] = times[step]
                print(
                    f"Fallback hold activated for agent {observer} at "
                    f"t={times[step]:.3f} s: {error}"
                )
                hold_wrench = stationkeeping_wrench(
                    model,
                    states[observer],
                    fallback_positions[observer],
                )
                hold_allocation = allocation.bounded_least_squares(
                    hold_wrench
                )
                controls[step, observer] = hold_allocation.forces
                next_states[observer] = plant_integrator.step(
                    plant_model,
                    states[observer],
                    hold_allocation.achieved_wrench,
                    dt,
                )
                controller_times[step, observer] = (
                    perf_counter() - start_time
                )
                continue

            controller_times[step, observer] = perf_counter() - start_time
            controls[step, observer] = (
                controller_design.representative_thruster_forces(
                    evaluation
                )
            )
            if evaluation.required_slack is not None:
                required_slack[
                    step,
                    observer,
                ] = evaluation.required_slack
            if evaluation.actuation_margin is not None:
                actuation_margin[
                    step,
                    observer,
                ] = evaluation.actuation_margin

            if adaptive:
                assert relaxation_evaluation is not None
                next_relaxation_state[observer] = np.clip(
                    relaxation_state[observer]
                    + dt * relaxation_evaluation.selected_rate,
                    0.0,
                    1.0,
                )

            follower_filters[observer] = filter_integrator.step(
                follower_controller.dynamics_controller.command_filter,
                follower_filters[observer],
                evaluation.controller.filter_command,
                dt,
            )
            next_states[observer] = plant_integrator.step(
                model,
                states[observer],
                evaluation.wrench_body,
                dt,
            )

        if velocity_reference_model is not None:
            assert velocity_reference_state is not None
            velocity_reference_state = reference_integrator.step(
                velocity_reference_model,
                velocity_reference_state,
                velocity_command,
                dt,
            )

        states = next_states
        relaxation_state = next_relaxation_state
        state_history.append(states.copy())
        record(step + 1)

    return MovingFormationResult(
        scenario=scenario,
        trajectory=FormationTrajectory(
            times=times,
            positions=positions,
            velocities=inertial_velocities,
            controls=controls,
            quaternions=quaternions,
        ),
        camera=camera,
        distance_domain=distance_domain,
        fov_domain=fov_domain,
        leader_reference_position=reference_positions,
        leader_reference_velocity=reference_velocities,
        leader_reference_acceleration=reference_accelerations,
        formation_error_norm=formation_error_norm,
        normalized_relaxation=normalized_relaxation,
        image_history=image_history,
        required_slack=required_slack,
        actuation_margin=actuation_margin,
        controller_times=controller_times,
        fallback_times=fallback_times,
        allocation=allocation,
        realism_enabled=(realism is not None),
        random_seed=random_seed,
    )


def sensing_margin_histories(
    result: MovingFormationResult,
) -> tuple[np.ndarray, np.ndarray]:
    conservative = np.full(result.trajectory.n_samples, np.inf)
    physical = np.full(result.trajectory.n_samples, np.inf)

    for edge in result.scenario.graph:
        relative = (
            result.trajectory.positions[:, edge.target]
            - result.trajectory.positions[:, edge.observer]
        )
        distance = np.linalg.norm(relative, axis=1)
        alpha_h = np.abs(result.image_history[:, edge.observer, 0])
        alpha_v = np.abs(result.image_history[:, edge.observer, 1])

        edge_conservative = np.minimum.reduce(
            (
                distance - result.distance_domain.d_min_conservative,
                result.distance_domain.d_max_conservative - distance,
                result.fov_domain.alpha_h_conservative - alpha_h,
                result.fov_domain.alpha_v_conservative - alpha_v,
            )
        )
        edge_physical = np.minimum.reduce(
            (
                distance - result.distance_domain.d_min,
                result.distance_domain.d_max - distance,
                1.0 - alpha_h,
                1.0 - alpha_v,
            )
        )
        conservative = np.minimum(conservative, edge_conservative)
        physical = np.minimum(physical, edge_physical)

    return conservative, physical


def connection_quality_history(
    result: MovingFormationResult,
) -> np.ndarray:
    quality = np.ones(
        (
            result.trajectory.n_samples,
            len(result.scenario.graph.edges),
        )
    )

    for edge_index, edge in enumerate(result.scenario.graph.edges):
        relative = (
            result.trajectory.positions[:, edge.target]
            - result.trajectory.positions[:, edge.observer]
        )
        distance = np.linalg.norm(relative, axis=1)
        range_quality = np.clip(
            (result.distance_domain.d_max - distance)
            / (
                result.distance_domain.d_max
                - result.distance_domain.d_max_conservative
            ),
            0.0,
            1.0,
        )

        alpha_h = np.abs(result.image_history[:, edge.observer, 0])
        alpha_v = np.abs(result.image_history[:, edge.observer, 1])
        horizontal_quality = np.clip(
            (1.0 - alpha_h)
            / (1.0 - result.fov_domain.alpha_h_conservative),
            0.0,
            1.0,
        )
        vertical_quality = np.clip(
            (1.0 - alpha_v)
            / (1.0 - result.fov_domain.alpha_v_conservative),
            0.0,
            1.0,
        )
        quality[:, edge_index] = np.nan_to_num(
            np.minimum.reduce(
                (
                    range_quality,
                    horizontal_quality,
                    vertical_quality,
                )
            ),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    return quality


def thruster_utilization_history(
    result: MovingFormationResult,
) -> np.ndarray:
    controls = result.trajectory.controls
    assert controls is not None
    utilization = np.zeros(
        (controls.shape[0], result.trajectory.n_agents),
        dtype=float,
    )
    for step in range(controls.shape[0]):
        for agent in range(result.trajectory.n_agents):
            utilization[step, agent] = float(
                np.max(
                    result.allocation.utilization(
                        controls[step, agent]
                    )
                )
            )
    return utilization


def depth_maximum_history(
    scenario: FormationScenario,
    values: np.ndarray,
) -> dict[int, np.ndarray]:
    histories: dict[int, np.ndarray] = {}
    maximum_depth = max(
        scenario.graph.depth(agent)
        for agent in range(scenario.n_agents)
    )
    for depth in range(1, maximum_depth + 1):
        agents = [
            agent
            for agent in range(scenario.n_agents)
            if scenario.graph.depth(agent) == depth
        ]
        histories[depth] = np.nanmax(values[:, agents], axis=1)
    return histories


def print_summary(result: MovingFormationResult, *, motion: MotionMode) -> None:
    trajectory = result.trajectory
    leader = result.scenario.graph.root
    assert trajectory.velocities is not None

    position_error = np.linalg.norm(
        trajectory.positions[:, leader]
        - result.leader_reference_position,
        axis=1,
    )
    velocity_error = np.linalg.norm(
        trajectory.velocities[:, leader]
        - result.leader_reference_velocity,
        axis=1,
    )
    conservative_margin, physical_margin = sensing_margin_histories(result)
    utilization = thruster_utilization_history(result)

    print(f"Moving-formation scenario: {motion}")
    print(
        "Realism perturbations: "
        + ("enabled" if result.realism_enabled else "disabled")
        + f", seed {result.random_seed}"
    )
    print(
        "Leader CLF-QP tracking: "
        f"RMS position error {np.sqrt(np.mean(position_error**2)):.4f} m, "
        f"max {np.max(position_error):.4f} m; "
        f"RMS velocity error {np.sqrt(np.mean(velocity_error**2)):.4f} m/s"
    )

    for depth, history in depth_maximum_history(
        result.scenario,
        result.formation_error_norm,
    ).items():
        print(
            f"Formation depth {depth}: "
            f"RMS max-edge error {np.sqrt(np.nanmean(history**2)):.4f} m, "
            f"peak {np.nanmax(history):.4f} m"
        )

    print(
        "Sensing margins: "
        f"minimum conservative {np.nanmin(conservative_margin):.4g}, "
        f"minimum physical {np.nanmin(physical_margin):.4g}"
    )
    print(
        "Adaptive enlargement: "
        f"maximum normalized s {np.nanmax(result.normalized_relaxation):.3f}"
    )
    print(
        "Thruster utilization: "
        f"maximum {np.nanmax(utilization):.3f}"
    )

    finite_required = result.required_slack[
        np.isfinite(result.required_slack)
    ]
    finite_margin = result.actuation_margin[
        np.isfinite(result.actuation_margin)
    ]
    if finite_required.size:
        print(
            "Actuation feasibility: "
            f"max required slack {np.max(finite_required):.4g}, "
            f"infeasible samples "
            f"{100.0 * np.mean(finite_required > 1e-10):.2f}%"
        )
    if finite_margin.size:
        print(
            "Minimum zero-slack actuation margin: "
            f"{np.min(finite_margin):.4g}"
        )

    timing = result.controller_times[
        np.isfinite(result.controller_times)
    ]
    if timing.size:
        print(
            "Controller timing (all agents): "
            f"mean {1e3 * np.mean(timing):.3f} ms, "
            f"max {1e3 * np.max(timing):.3f} ms"
        )

    finite_fallback = result.fallback_times[
        np.isfinite(result.fallback_times)
    ]
    if finite_fallback.size:
        print(
            "Fallback: activated, first occurrence at "
            f"{np.min(finite_fallback):.3f} s"
        )
    else:
        print("Fallback: no activations")


def plot_spatial_paths(
    result: MovingFormationResult,
    *,
    motion: MotionMode,
) -> plt.Figure:
    figure = plt.figure()
    axes = figure.add_subplot(111, projection="3d")

    for agent in range(result.trajectory.n_agents):
        axes.plot(
            result.trajectory.positions[:, agent, 0],
            result.trajectory.positions[:, agent, 1],
            result.trajectory.positions[:, agent, 2],
            label=f"agent {agent}",
        )

    axes.plot(
        result.leader_reference_position[:, 0],
        result.leader_reference_position[:, 1],
        result.leader_reference_position[:, 2],
        linestyle="--",
        linewidth=1.5,
        label="leader reference",
    )
    axes.set_xlabel("x [m]")
    axes.set_ylabel("y [m]")
    axes.set_zlabel("z [m]")
    axes.set_title(
        "Seven-BlueROV moving formation — "
        + motion.replace("_", " ")
    )
    axes.legend(ncol=2)
    figure.tight_layout()
    return figure


def plot_leader_tracking(result: MovingFormationResult) -> plt.Figure:
    leader = result.scenario.graph.root
    assert result.trajectory.velocities is not None

    position_error = np.linalg.norm(
        result.trajectory.positions[:, leader]
        - result.leader_reference_position,
        axis=1,
    )
    velocity_error = np.linalg.norm(
        result.trajectory.velocities[:, leader]
        - result.leader_reference_velocity,
        axis=1,
    )

    figure, axes = plt.subplots()
    axes.plot(
        result.trajectory.times,
        position_error,
        label=r"$\|p_1-p_{1,r}\|$",
    )
    axes.plot(
        result.trajectory.times,
        velocity_error,
        linestyle="--",
        label=r"$\|v_1-v_{1,r}\|$",
    )
    axes.set_xlabel(r"$t$ [s]")
    axes.set_ylabel("tracking-error norm")
    axes.set_title("Leader tracking with the common CLF-QP")
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    return figure


def plot_formation_error_by_depth(
    result: MovingFormationResult,
) -> plt.Figure:
    figure, axes = plt.subplots()
    histories = depth_maximum_history(
        result.scenario,
        result.formation_error_norm,
    )
    for depth, history in histories.items():
        axes.plot(
            result.trajectory.times,
            history,
            label=f"depth {depth}: max edge error",
        )
    axes.set_xlabel(r"$t$ [s]")
    axes.set_ylabel(r"$\|\tilde p_{ij}\|$ [m]")
    axes.set_title("Formation error propagation through the sensing tree")
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    return figure


def plot_adaptive_enlargement_by_depth(
    result: MovingFormationResult,
) -> plt.Figure:
    per_agent = np.max(result.normalized_relaxation, axis=2)
    histories = depth_maximum_history(result.scenario, per_agent)

    figure, axes = plt.subplots()
    for depth, history in histories.items():
        axes.plot(
            result.trajectory.times,
            history,
            label=f"depth {depth}",
        )
    axes.axhline(1.0, linestyle=":", label="physical-domain limit")
    axes.set_xlabel(r"$t$ [s]")
    axes.set_ylabel(r"$\max_\ell s_{\ell,i}$")
    axes.set_title("Maximum adaptive-domain enlargement by tree depth")
    axes.set_ylim(bottom=-0.02)
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    return figure


def plot_sensing_margins(result: MovingFormationResult) -> plt.Figure:
    conservative, physical = sensing_margin_histories(result)
    figure, axes = plt.subplots()
    axes.plot(
        result.trajectory.times,
        conservative,
        label="minimum conservative margin",
    )
    axes.plot(
        result.trajectory.times,
        physical,
        linestyle="--",
        label="minimum physical margin",
    )
    axes.axhline(0.0, linestyle=":")
    axes.set_xlabel(r"$t$ [s]")
    axes.set_ylabel("minimum normalized/distance margin")
    axes.set_title("Worst sensing margin across the directed tree")
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    return figure


def plot_thruster_utilization(result: MovingFormationResult) -> plt.Figure:
    utilization = thruster_utilization_history(result)
    figure, axes = plt.subplots()
    control_times = result.trajectory.times[:-1]

    maximum_depth = max(
        result.scenario.graph.depth(agent)
        for agent in range(result.scenario.n_agents)
    )
    for depth in range(maximum_depth + 1):
        agents = [
            agent
            for agent in range(result.scenario.n_agents)
            if result.scenario.graph.depth(agent) == depth
        ]
        maximum = np.max(utilization[:, agents], axis=1)
        axes.plot(
            control_times,
            maximum,
            label=f"depth {depth}",
        )

    axes.axhline(1.0, linestyle=":", label="thruster limit")
    axes.set_xlabel(r"$t$ [s]")
    axes.set_ylabel("maximum normalized thruster utilization")
    axes.set_title("Control effort by tree depth")
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    return figure


def save_histories(
    result: MovingFormationResult,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    conservative_margin, physical_margin = sensing_margin_histories(result)
    utilization = thruster_utilization_history(result)
    depth_histories = depth_maximum_history(
        result.scenario,
        result.formation_error_norm,
    )

    leader = result.scenario.graph.root
    assert result.trajectory.velocities is not None
    leader_position_error = np.linalg.norm(
        result.trajectory.positions[:, leader]
        - result.leader_reference_position,
        axis=1,
    )
    leader_velocity_error = np.linalg.norm(
        result.trajectory.velocities[:, leader]
        - result.leader_reference_velocity,
        axis=1,
    )

    with (output_dir / "validation_metrics.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        header = [
            "time",
            "leader_position_error",
            "leader_velocity_error",
            "minimum_conservative_sensing_margin",
            "minimum_physical_sensing_margin",
            "maximum_normalized_enlargement",
            "maximum_thruster_utilization",
        ]
        for depth in sorted(depth_histories):
            header.append(f"depth_{depth}_maximum_formation_error")
        writer.writerow(header)

        for sample, time in enumerate(result.trajectory.times):
            control_sample = min(
                sample,
                utilization.shape[0] - 1,
            )
            row = [
                time,
                leader_position_error[sample],
                leader_velocity_error[sample],
                conservative_margin[sample],
                physical_margin[sample],
                np.max(result.normalized_relaxation[sample]),
                np.max(utilization[control_sample]),
            ]
            row.extend(
                depth_histories[depth][sample]
                for depth in sorted(depth_histories)
            )
            writer.writerow(row)

    np.savez_compressed(
        output_dir / "moving_formation_histories.npz",
        times=result.trajectory.times,
        positions=result.trajectory.positions,
        velocities=result.trajectory.velocities,
        quaternions=result.trajectory.quaternions,
        controls=result.trajectory.controls,
        leader_reference_position=result.leader_reference_position,
        leader_reference_velocity=result.leader_reference_velocity,
        leader_reference_acceleration=(
            result.leader_reference_acceleration
        ),
        formation_error_norm=result.formation_error_norm,
        normalized_relaxation=result.normalized_relaxation,
        image_history=result.image_history,
        required_slack=result.required_slack,
        actuation_margin=result.actuation_margin,
        controller_times=result.controller_times,
        fallback_times=result.fallback_times,
        realism_enabled=np.array(result.realism_enabled),
        random_seed=np.array(result.random_seed),
    )


def parser_for_motion(
    default_motion: MotionMode,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--motion",
        choices=("velocity_command", "trajectory"),
        default=default_motion,
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="defaults to 150 s or 200 s according to the motion mode",
    )
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--thruster-voltage", type=int, default=16)
    parser.add_argument("--thrust-derating", type=float, default=1.0)
    parser.add_argument(
        "--control-space",
        choices=("thruster", "wrench"),
        default="thruster",
    )
    parser.add_argument(
        "--adaptive",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--use-parent-velocity-in-clf",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--leader-velocity",
        nargs=3,
        type=float,
        default=(0.35, 0.08, 0.0),
        metavar=("VX", "VY", "VZ"),
        help="inertial velocity command [m/s]",
    )
    parser.add_argument(
        "--leader-velocity-bandwidth",
        type=float,
        default=0.2,
        help=(
            "bandwidth [1/s] of the velocity-command reference filter; "
            "its output derivative supplies leader reference acceleration"
        ),
    )
    parser.add_argument(
        "--realistic",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--model-parameter-variation",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--water-current",
        nargs=3,
        type=float,
        default=(0.05, -0.02, 0.0),
        metavar=("VX", "VY", "VZ"),
    )
    parser.add_argument(
        "--relative-position-noise-std",
        type=float,
        default=0.015,
    )
    parser.add_argument(
        "--linear-velocity-noise-std",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--angular-velocity-noise-std",
        type=float,
        default=0.005,
    )
    parser.add_argument(
        "--relative-measurement-delay",
        type=float,
        default=0.04,
    )
    parser.add_argument(
        "--virtual-linear-speed-limit",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--virtual-angular-speed-limit",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--relaxation-recovery-gain",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--relaxation-domain-margin-ratio",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--slack-linear-penalty",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--slack-quadratic-penalty",
        type=float,
        default=5e3,
    )
    parser.add_argument("--paper-quality", action="store_true")
    parser.add_argument(
        "--figure-format",
        choices=("png", "pdf", "svg"),
        default=None,
    )
    parser.add_argument("--animate", action="store_true")
    parser.add_argument("--frame-stride", type=int, default=20)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--save-animation", action="store_true")
    parser.add_argument(
        "--animation-format",
        choices=("mp4", "gif"),
        default="mp4",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--no-show", action="store_true")
    return parser


def main(default_motion: MotionMode) -> None:
    parser = parser_for_motion(default_motion)
    args = parser.parse_args()

    motion: MotionMode = args.motion
    duration = args.duration
    if duration is None:
        duration = 150.0 if motion == "velocity_command" else 200.0
    if duration <= 0.0:
        parser.error("--duration must be positive")
    if args.dt <= 0.0:
        parser.error("--dt must be positive")
    if args.leader_velocity_bandwidth <= 0.0:
        parser.error("--leader-velocity-bandwidth must be positive")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            Path("outputs")
            / "bluerov2_moving_formation"
            / motion
        )

    apply_visualization_style(paper_quality=args.paper_quality)

    realism = None
    if args.realistic:
        realism = RealismConfig(
            parameter_variation=args.model_parameter_variation,
            water_current_inertial=tuple(args.water_current),
            relative_position_noise_std=(
                args.relative_position_noise_std
            ),
            linear_velocity_noise_std=(
                args.linear_velocity_noise_std
            ),
            angular_velocity_noise_std=(
                args.angular_velocity_noise_std
            ),
            relative_measurement_delay=(
                args.relative_measurement_delay
            ),
        )

    result = simulate(
        motion=motion,
        duration=duration,
        dt=args.dt,
        velocity_command=np.asarray(args.leader_velocity, dtype=float),
        velocity_command_bandwidth=args.leader_velocity_bandwidth,
        thruster_voltage=args.thruster_voltage,
        thrust_derating=args.thrust_derating,
        control_space=args.control_space,
        adaptive=args.adaptive,
        use_parent_velocity_in_clf=args.use_parent_velocity_in_clf,
        virtual_linear_speed_limit=args.virtual_linear_speed_limit,
        virtual_angular_speed_limit=args.virtual_angular_speed_limit,
        relaxation_recovery_gain=args.relaxation_recovery_gain,
        relaxation_domain_margin_ratio=(
            args.relaxation_domain_margin_ratio
        ),
        slack_linear_penalty=args.slack_linear_penalty,
        slack_quadratic_penalty=args.slack_quadratic_penalty,
        realism=realism,
        random_seed=args.seed,
    )
    print_summary(result, motion=motion)

    figures = {
        "spatial_paths": plot_spatial_paths(result, motion=motion),
        "leader_tracking": plot_leader_tracking(result),
        "formation_error_by_depth": plot_formation_error_by_depth(result),
        "adaptive_enlargement_by_depth": (
            plot_adaptive_enlargement_by_depth(result)
        ),
        "sensing_margins": plot_sensing_margins(result),
        "thruster_utilization": plot_thruster_utilization(result),
    }

    animation = None
    if args.animate or args.save_animation:
        followers = tuple(
            agent
            for agent in range(result.scenario.n_agents)
            if agent != result.scenario.graph.root
        )
        vehicle_geometry = BlueROV2HeavyVisualGeometry(
            thruster_configuration=result.allocation.configuration
        ).wireframe()
        animation = animate_formation_3d(
            result.trajectory,
            result.scenario.graph,
            camera=result.camera,
            camera_agents=followers,
            camera_depth=0.75,
            vehicle_geometry=vehicle_geometry,
            edge_quality=connection_quality_history(result),
            paper_quality=args.paper_quality,
            show_body_forward=False,
            trail_length=500,
            frame_stride=args.frame_stride,
            title=(
                "Seven-BlueROV moving formation — "
                + motion.replace("_", " ")
            ),
        )

    if args.save or args.save_animation:
        output_dir.mkdir(parents=True, exist_ok=True)
        figure_format = (
            args.figure_format
            if args.figure_format is not None
            else "pdf" if args.paper_quality else "png"
        )
        if args.save:
            for name, figure in figures.items():
                save_figure(
                    figure,
                    output_dir / f"{name}.{figure_format}",
                    paper_quality=args.paper_quality,
                )
            save_histories(result, output_dir)

        if args.save_animation:
            assert animation is not None
            save_animation(
                animation.animation,
                output_dir / f"formation_animation.{args.animation_format}",
                paper_quality=args.paper_quality,
                fps=30,
            )

    if not args.no_show:
        plt.show()

    _ = animation
