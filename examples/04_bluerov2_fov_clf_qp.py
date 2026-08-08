"""3-D BlueROV2 formation control with camera/FoV barriers and a CLF-QP.

The leader is held at a fixed 6-DoF equilibrium by applying its exact model
drift wrench.  Two BlueROV2 followers observe the leader through forward
cameras and regulate desired relative positions while keeping the target
centered in the image.

The example supports conservative distance/FoV barriers and optional
domain-preserving normalized funnel relaxation toward the physical admissible
domain. The physical CLF-QP treats the current funnel state as frozen and
optimizes only the physical actuator input. The auxiliary funnel rate is
computed independently from a sampled-data barrier-domain condition.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

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
    BlueROV2ControllerDesign,
    BlueROV2ControlSpace,
    FunnelRelaxationInfeasibleError,
    FunnelRelaxationPolicy,
    build_bluerov2_controller_design,
)
from formation_control.geometry import (
    PinholeCamera,
    quaternion_from_roll_pitch_yaw,
    rotation_matrix_from_quaternion,
)
from formation_control.graphs import DirectedSensingGraph
from formation_control.models import BlueROV2Model
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
from formation_control.simulation.stress_cases import (
    set_outward_linear_velocity,
)
from formation_control.visualization import (
    BlueROV2HeavyVisualGeometry,
    animate_formation_3d,
    apply_visualization_style,
    plot_formation_3d,
    save_animation,
    save_figure,
)


@dataclass(frozen=True)
class EdgeBarrierTemplates:
    collision: AdaptiveConstraintBarrierPotential[np.ndarray] | None
    sensing_range: AdaptiveConstraintBarrierPotential[np.ndarray] | None
    horizontal_fov: AdaptiveConstraintBarrierPotential[NormalizedImagePoint] | None
    vertical_fov: AdaptiveConstraintBarrierPotential[NormalizedImagePoint] | None


@dataclass(frozen=True)
class SimulationStatus:
    """Failure information retained after graceful fallback."""

    fallback_times: np.ndarray
    fallback_reasons: tuple[str | None, ...]

    def __post_init__(self) -> None:
        times = np.asarray(self.fallback_times, dtype=float)
        if times.ndim != 1:
            raise ValueError("fallback_times must be one-dimensional.")
        if len(self.fallback_reasons) != times.size:
            raise ValueError("fallback_reasons must match fallback_times length.")
        object.__setattr__(self, "fallback_times", times.copy())

    @property
    def first_fallback_time(self) -> float | None:
        finite = self.fallback_times[np.isfinite(self.fallback_times)]
        if finite.size == 0:
            return None
        return float(np.min(finite))

    @property
    def fallback_occurred(self) -> bool:
        return self.first_fallback_time is not None


@dataclass(frozen=True)
class CLFDiagnosticHistory:
    """Time histories of the terms entering the physical hard-CLF condition.

    The zero-slack condition is

        a + b.T f + alpha(W) <= 0.

    ``best_actuator_contribution`` is the exact box-constrained minimum of
    ``b.T f``.  Hence ``hard_clf_residual`` is

        a + alpha(W) + min_f b.T f,

    and its positive part equals the actuator-required slack.
    """

    value: np.ndarray
    decay: np.ndarray
    drift: np.ndarray
    configuration_local_rate: np.ndarray
    parent_rate: np.ndarray
    velocity_backstepping_rate: np.ndarray
    dynamics_bias_rate: np.ndarray
    dynamics_bias_linear_rate: np.ndarray
    dynamics_bias_angular_rate: np.ndarray
    command_acceleration_rate: np.ndarray
    command_acceleration_linear_rate: np.ndarray
    command_acceleration_angular_rate: np.ndarray
    velocity_error_norm: np.ndarray
    velocity_error_linear_norm: np.ndarray
    velocity_error_angular_norm: np.ndarray
    filtered_velocity_derivative_norm: np.ndarray
    filtered_linear_acceleration_norm: np.ndarray
    filtered_angular_acceleration_norm: np.ndarray
    generalized_velocity: np.ndarray
    filtered_velocity: np.ndarray
    desired_velocity: np.ndarray
    unlimited_desired_velocity: np.ndarray
    filtered_velocity_derivative: np.ndarray
    dynamics_bias: np.ndarray
    best_actuator_contribution: np.ndarray
    minimum_modeled_derivative: np.ndarray
    hard_clf_residual: np.ndarray


@dataclass(frozen=True)
class ThrustAuthoritySweepRow:
    derating: float
    fallback: bool
    fallback_time: float | None
    maximum_normalized_relaxation: float
    minimum_physical_margin_before_fallback: float
    maximum_thruster_utilization: float
    actuator_infeasible_fraction: float


def look_at_quaternion(
    relative_position_inertial: np.ndarray,
    *,
    yaw_offset: float = 0.0,
    pitch_offset: float = 0.0,
) -> np.ndarray:
    """Point body ``+x`` approximately toward an inertial relative vector."""
    relative = np.asarray(relative_position_inertial, dtype=float)
    horizontal_norm = float(np.hypot(relative[0], relative[1]))

    yaw = float(np.arctan2(relative[1], relative[0]) + yaw_offset)
    pitch = float(-np.arctan2(relative[2], horizontal_norm) + pitch_offset)

    return quaternion_from_roll_pitch_yaw(
        roll=0.0,
        pitch=pitch,
        yaw=yaw,
    )


def build_scenario(
    *,
    stress_test: bool = False,
) -> FormationScenario:
    graph = DirectedSensingGraph.rooted_star(3)
    reference = FormationReference(
        offsets=np.array(
            [
                [0.0, 0.0, 0.0],
                [-1.8, -0.7, -0.4],
                [-2.0, 0.8, 0.5],
            ],
            dtype=float,
        )
    )

    if stress_test:
        # Both followers start strictly inside the conservative sensing domain
        # but close to its outer range boundary.  Their initial velocities and
        # attitudes (set below) deliberately push them toward loss of sensing.
        initial_positions = np.array(
            [
                [0.0, 0.0, -1.0],
                [-2.78, -0.55, -0.85],
                [-2.55, 1.15, -1.35],
            ],
            dtype=float,
        )
    else:
        initial_positions = np.array(
            [
                [0.0, 0.0, -1.0],
                [-2.25, -0.25, -0.8],
                [-1.55, 1.25, -1.7],
            ],
            dtype=float,
        )

    return FormationScenario(
        graph=graph,
        reference=reference,
        initial_positions=initial_positions,
    )


def build_initial_states(
    scenario: FormationScenario,
    model: BlueROV2Model,
    *,
    stress_test: bool = False,
    stress_scale: float = 1.0,
) -> np.ndarray:
    if not np.isfinite(stress_scale) or stress_scale < 0.0:
        raise ValueError("stress_scale must be finite and nonnegative.")

    states = np.zeros((scenario.n_agents, 13), dtype=float)

    states[0, :3] = scenario.initial_positions[0]
    states[0, 3:7] = quaternion_from_roll_pitch_yaw(0.0, 0.0, 0.0)

    if stress_test:
        orientation_offsets = {
            1: (np.deg2rad(30.0), np.deg2rad(10.0)),
            2: (np.deg2rad(-30.0), np.deg2rad(-12.0)),
        }
        outward_speeds = {
            1: 1.10 * stress_scale,
            2: 0.75 * stress_scale,
        }
        yaw_rates = {
            1: 0.35 * stress_scale,
            2: -0.45 * stress_scale,
        }
    else:
        orientation_offsets = {
            1: (np.deg2rad(10.0), np.deg2rad(-5.0)),
            2: (np.deg2rad(-9.0), np.deg2rad(5.0)),
        }
        outward_speeds = {1: 0.0, 2: 0.0}
        yaw_rates = {1: 0.0, 2: 0.0}

    for edge in scenario.graph:
        relative = (
            scenario.initial_positions[edge.target] - scenario.initial_positions[edge.observer]
        )
        yaw_offset, pitch_offset = orientation_offsets[edge.observer]

        states[edge.observer, :3] = scenario.initial_positions[edge.observer]
        states[edge.observer, 3:7] = look_at_quaternion(
            relative,
            yaw_offset=yaw_offset,
            pitch_offset=pitch_offset,
        )

        if stress_test:
            states[edge.observer] = set_outward_linear_velocity(
                model,
                states[edge.observer],
                states[edge.target, :3],
                speed=outward_speeds[edge.observer],
            )
            states[edge.observer, 12] = yaw_rates[edge.observer]

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


def build_agent_controller(
    model: BlueROV2Model,
    allocation: BlueROV2HeavyThrusterAllocation,
    *,
    control_space: BlueROV2ControlSpace,
    slack_linear_penalty: float,
    slack_quadratic_penalty: float,
    virtual_linear_speed_limit: float,
    virtual_angular_speed_limit: float,
) -> BlueROV2ControllerDesign:
    """Build the physical CLF-QP controller.

    Adaptive funnel relaxation does not augment the CLF-QP decision vector.
    The current funnel state changes the potential, but its auxiliary dynamics
    are handled independently.
    """
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
    distance_constraints: bool,
    fov_constraints: bool,
    recovery_gain: float,
    domain_margin_ratio: float,
) -> FunnelRelaxationPolicy:
    maximum = np.array(
        [
            distance_domain.collision_enlargement_max if distance_constraints else 0.0,
            distance_domain.range_enlargement_max if distance_constraints else 0.0,
            fov_domain.horizontal_enlargement_max if fov_constraints else 0.0,
            fov_domain.vertical_enlargement_max if fov_constraints else 0.0,
        ],
        dtype=float,
    )
    return FunnelRelaxationPolicy(
        maximum_enlargement=maximum,
        recovery_gain=recovery_gain,
        domain_margin_ratio=domain_margin_ratio,
        minimum_constraint_margin=1e-5,
    )


def relaxation_enabled_mask(
    *,
    distance_constraints: bool,
    fov_constraints: bool,
) -> np.ndarray:
    return np.array(
        [
            distance_constraints,
            distance_constraints,
            fov_constraints,
            fov_constraints,
        ],
        dtype=bool,
    )


def build_barrier_templates(
    scenario: FormationScenario,
    distance_domain: DistanceDomain,
    fov_domain: FieldOfViewDomain,
    observer: int,
    target: int,
    *,
    distance_constraints: bool,
    fov_constraints: bool,
) -> EdgeBarrierTemplates:
    desired_relative = scenario.desired_relative_position(observer, target)
    desired_image = NormalizedImagePoint(0.0, 0.0)

    collision = None
    sensing_range = None
    horizontal_fov = None
    vertical_fov = None

    if distance_constraints:
        collision = AdaptiveConstraintBarrierPotential(
            constraint=MinimumDistanceConstraint(distance_domain.d_min_conservative),
            reference_state=desired_relative,
            weight=0.18,
        )
        sensing_range = AdaptiveConstraintBarrierPotential(
            constraint=MaximumDistanceConstraint(distance_domain.d_max_conservative),
            reference_state=desired_relative,
            weight=0.18,
        )

    if fov_constraints:
        horizontal_fov = AdaptiveConstraintBarrierPotential(
            constraint=HorizontalFieldOfViewConstraint(fov_domain.alpha_h_conservative),
            reference_state=desired_image,
            weight=0.25,
        )
        vertical_fov = AdaptiveConstraintBarrierPotential(
            constraint=VerticalFieldOfViewConstraint(fov_domain.alpha_v_conservative),
            reference_state=desired_image,
            weight=0.25,
        )

    return EdgeBarrierTemplates(
        collision=collision,
        sensing_range=sensing_range,
        horizontal_fov=horizontal_fov,
        vertical_fov=vertical_fov,
    )


def build_edge_potential(
    scenario: FormationScenario,
    camera: PinholeCamera,
    distance_domain: DistanceDomain,
    fov_domain: FieldOfViewDomain,
    observer: int,
    target: int,
    templates: EdgeBarrierTemplates,
    rho: np.ndarray,
    *,
    adaptive: bool,
) -> EdgePotential:
    desired_relative = scenario.desired_relative_position(observer, target)
    desired_image = NormalizedImagePoint(0.0, 0.0)

    collision_barrier = None
    range_barrier = None
    horizontal_barrier = None
    vertical_barrier = None

    if templates.collision is not None:
        if adaptive:
            collision_barrier = templates.collision.bind(float(rho[0]))
        else:
            collision_barrier = ConstraintBarrierPotential.from_reference(
                MinimumDistanceConstraint(distance_domain.d_min_conservative),
                desired_relative,
                weight=0.18,
            )

    if templates.sensing_range is not None:
        if adaptive:
            range_barrier = templates.sensing_range.bind(float(rho[1]))
        else:
            range_barrier = ConstraintBarrierPotential.from_reference(
                MaximumDistanceConstraint(distance_domain.d_max_conservative),
                desired_relative,
                weight=0.18,
            )

    if templates.horizontal_fov is not None:
        if adaptive:
            horizontal_barrier = templates.horizontal_fov.bind(float(rho[2]))
        else:
            horizontal_barrier = ConstraintBarrierPotential.from_reference(
                HorizontalFieldOfViewConstraint(fov_domain.alpha_h_conservative),
                desired_image,
                weight=0.25,
            )

    if templates.vertical_fov is not None:
        if adaptive:
            vertical_barrier = templates.vertical_fov.bind(float(rho[3]))
        else:
            vertical_barrier = ConstraintBarrierPotential.from_reference(
                VerticalFieldOfViewConstraint(fov_domain.alpha_v_conservative),
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
        collision_barrier=collision_barrier,
        range_barrier=range_barrier,
        horizontal_fov_barrier=horizontal_barrier,
        vertical_fov_barrier=vertical_barrier,
        camera=camera,
    )


def inertial_linear_velocity(
    model: BlueROV2Model,
    state: np.ndarray,
) -> np.ndarray:
    _, quaternion, generalized_velocity = model.split_state(state)
    rotation = rotation_matrix_from_quaternion(quaternion)
    return rotation @ generalized_velocity[:3]


SENSING_CHANNELS = (
    "collision",
    "range",
    "horizontal_fov",
    "vertical_fov",
)


def conservative_domain_failure_reason(
    constraint_values: np.ndarray,
    *,
    enabled: np.ndarray,
    minimum_margin: float,
) -> str | None:
    """Return a clean failure reason before a log barrier becomes undefined."""
    values = np.asarray(constraint_values, dtype=float)
    active = np.asarray(enabled, dtype=bool)
    if values.shape != (4,) or active.shape != (4,):
        raise ValueError("constraint_values and enabled must both have shape (4,).")

    for index, channel in enumerate(SENSING_CHANNELS):
        if active[index] and values[index] <= minimum_margin:
            return (
                f"conservative {channel} domain lost: "
                f"h_c={values[index]:.6g}, "
                f"required h_c>{minimum_margin:.6g}"
            )
    return None


def stationkeeping_wrench(
    model: BlueROV2Model,
    state: np.ndarray,
    hold_position: np.ndarray,
) -> np.ndarray:
    """Simple sensing-independent fallback that brakes and holds position."""
    position, quaternion, velocity = model.split_state(state)
    rotation = rotation_matrix_from_quaternion(quaternion)
    velocity_inertial = rotation @ velocity[:3]

    force_inertial = -5.0 * (position - hold_position) - 4.0 * velocity_inertial
    force_body = rotation.T @ force_inertial
    torque_body = -1.5 * velocity[3:]

    return model.drift_wrench(state) + np.concatenate((force_body, torque_body))


def simulate(
    *,
    duration: float = 10.0,
    dt: float = 0.01,
    thruster_voltage: int = 16,
    thrust_derating: float = 1.0,
    control_space: BlueROV2ControlSpace = "thruster",
    distance_constraints: bool = True,
    fov_constraints: bool = True,
    adaptive: bool = False,
    stress_test: bool = False,
    stress_scale: float = 1.0,
    slack_linear_penalty: float = 100.0,
    slack_quadratic_penalty: float = 5e3,
    virtual_linear_speed_limit: float = 1.5,
    virtual_angular_speed_limit: float = 2.0,
    relaxation_recovery_gain: float = 0.8,
    relaxation_domain_margin_ratio: float = 0.1,
) -> tuple[
    FormationScenario,
    FormationTrajectory,
    PinholeCamera,
    DistanceDomain,
    FieldOfViewDomain,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    CLFDiagnosticHistory,
    BlueROV2HeavyThrusterAllocation,
    SimulationStatus,
]:
    scenario = build_scenario(stress_test=stress_test)
    camera = build_camera()
    distance_domain, fov_domain = build_domains()

    model = BlueROV2Model()
    allocation = BlueROV2HeavyThrusterAllocation.default_45deg(
        voltage=thruster_voltage,
        derating=thrust_derating,
    )
    controller_design = build_agent_controller(
        model,
        allocation,
        control_space=control_space,
        slack_linear_penalty=slack_linear_penalty,
        slack_quadratic_penalty=slack_quadratic_penalty,
        virtual_linear_speed_limit=virtual_linear_speed_limit,
        virtual_angular_speed_limit=virtual_angular_speed_limit,
    )
    controller = controller_design.agent_controller

    relaxation = build_relaxation_policy(
        distance_domain,
        fov_domain,
        distance_constraints=distance_constraints,
        fov_constraints=fov_constraints,
        recovery_gain=relaxation_recovery_gain,
        domain_margin_ratio=relaxation_domain_margin_ratio,
    )
    enabled_relaxation = relaxation_enabled_mask(
        distance_constraints=distance_constraints,
        fov_constraints=fov_constraints,
    )

    plant_integrator = RK4Integrator()
    filter_integrator = RK4Integrator()

    states = build_initial_states(
        scenario,
        model,
        stress_test=stress_test,
        stress_scale=stress_scale,
    )
    relaxation_state = np.zeros((scenario.n_agents, 4), dtype=float)
    guard_activation_count = 0
    guard_maximum_correction = 0.0
    fallback_mode = np.zeros(scenario.n_agents, dtype=bool)
    fallback_positions = states[:, :3].copy()
    fallback_times = np.full(scenario.n_agents, np.nan)
    fallback_reasons: list[str | None] = [None for _ in range(scenario.n_agents)]

    templates = {
        edge.observer: build_barrier_templates(
            scenario,
            distance_domain,
            fov_domain,
            edge.observer,
            edge.target,
            distance_constraints=distance_constraints,
            fov_constraints=fov_constraints,
        )
        for edge in scenario.graph
    }
    if adaptive:
        for edge in scenario.graph:
            try:
                kinematics = evaluate_sensing_constraint_kinematics(
                    model,
                    camera,
                    distance_domain,
                    fov_domain,
                    states[edge.observer],
                    states[edge.target],
                )
                projected, correction = relaxation.project_to_current_domain(
                    relaxation_state[edge.observer],
                    kinematics.values,
                    enabled=enabled_relaxation,
                )
                relaxation_state[edge.observer] = projected
                if np.any(correction > 0.0):
                    guard_activation_count += 1
                    guard_maximum_correction = max(
                        guard_maximum_correction,
                        float(np.max(correction)),
                    )
            except (FunnelRelaxationInfeasibleError, ValueError) as error:
                fallback_mode[edge.observer] = True
                fallback_positions[edge.observer] = states[edge.observer, :3]
                fallback_times[edge.observer] = 0.0
                fallback_reasons[edge.observer] = str(error)
                print(f"Fallback hold activated for agent {edge.observer} at t=0.000 s: {error}")

    filters: dict[int, np.ndarray] = {}
    for edge in scenario.graph:
        if fallback_mode[edge.observer]:
            filters[edge.observer] = controller.dynamics_controller.command_filter.initialize(
                np.zeros(6)
            )
            continue

        rho = relaxation.enlargement(relaxation_state[edge.observer])
        potential = build_edge_potential(
            scenario,
            camera,
            distance_domain,
            fov_domain,
            edge.observer,
            edge.target,
            templates[edge.observer],
            rho,
            adaptive=adaptive,
        )
        filters[edge.observer] = controller.initialize_filter(
            follower_state=states[edge.observer],
            parent_position=states[edge.target, :3],
            edge_potential=potential,
        )

    steps = int(np.ceil(duration / dt))
    times = np.arange(steps + 1, dtype=float) * dt

    positions = np.empty((steps + 1, scenario.n_agents, 3))
    quaternions = np.empty((steps + 1, scenario.n_agents, 4))
    inertial_velocities = np.empty((steps + 1, scenario.n_agents, 3))
    controls = np.zeros((steps, scenario.n_agents, 8))
    slacks = np.zeros((steps, scenario.n_agents))
    required_slacks = np.full((steps, scenario.n_agents), np.nan)
    actuation_margins = np.full((steps, scenario.n_agents), np.nan)
    clf_values = np.full((steps, scenario.n_agents), np.nan)
    clf_decays = np.full((steps, scenario.n_agents), np.nan)
    clf_drifts = np.full((steps, scenario.n_agents), np.nan)
    clf_configuration_local_rates = np.full(
        (steps, scenario.n_agents),
        np.nan,
    )
    clf_parent_rates = np.full((steps, scenario.n_agents), np.nan)
    clf_velocity_backstepping_rates = np.full(
        (steps, scenario.n_agents),
        np.nan,
    )
    clf_dynamics_bias_rates = np.full(
        (steps, scenario.n_agents),
        np.nan,
    )
    clf_dynamics_bias_linear_rates = np.full(
        (steps, scenario.n_agents),
        np.nan,
    )
    clf_dynamics_bias_angular_rates = np.full(
        (steps, scenario.n_agents),
        np.nan,
    )
    clf_command_acceleration_rates = np.full(
        (steps, scenario.n_agents),
        np.nan,
    )
    clf_command_acceleration_linear_rates = np.full(
        (steps, scenario.n_agents),
        np.nan,
    )
    clf_command_acceleration_angular_rates = np.full(
        (steps, scenario.n_agents),
        np.nan,
    )
    clf_velocity_error_norms = np.full(
        (steps, scenario.n_agents),
        np.nan,
    )
    clf_velocity_error_linear_norms = np.full(
        (steps, scenario.n_agents),
        np.nan,
    )
    clf_velocity_error_angular_norms = np.full(
        (steps, scenario.n_agents),
        np.nan,
    )
    clf_filtered_velocity_derivative_norms = np.full(
        (steps, scenario.n_agents),
        np.nan,
    )
    clf_filtered_linear_acceleration_norms = np.full(
        (steps, scenario.n_agents),
        np.nan,
    )
    clf_filtered_angular_acceleration_norms = np.full(
        (steps, scenario.n_agents),
        np.nan,
    )
    clf_generalized_velocities = np.full(
        (steps, scenario.n_agents, 6),
        np.nan,
    )
    clf_filtered_velocities = np.full(
        (steps, scenario.n_agents, 6),
        np.nan,
    )
    clf_desired_velocities = np.full(
        (steps, scenario.n_agents, 6),
        np.nan,
    )
    clf_unlimited_desired_velocities = np.full(
        (steps, scenario.n_agents, 6),
        np.nan,
    )
    clf_filtered_velocity_derivatives = np.full(
        (steps, scenario.n_agents, 6),
        np.nan,
    )
    clf_dynamics_biases = np.full(
        (steps, scenario.n_agents, 6),
        np.nan,
    )
    clf_best_actuator_contributions = np.full(
        (steps, scenario.n_agents),
        np.nan,
    )
    clf_minimum_modeled_derivatives = np.full(
        (steps, scenario.n_agents),
        np.nan,
    )
    clf_hard_residuals = np.full((steps, scenario.n_agents), np.nan)
    controller_times = np.zeros((steps, scenario.n_agents))
    rho_history = np.zeros((steps + 1, scenario.n_agents, 4))
    relaxation_rate_history = np.zeros((steps, scenario.n_agents, 4))
    image_history = np.full(
        (steps + 1, scenario.n_agents, 2),
        np.nan,
    )

    def record(sample: int) -> None:
        positions[sample] = states[:, :3]
        quaternions[sample] = states[:, 3:7]
        for agent in range(scenario.n_agents):
            rho_history[sample, agent] = relaxation.enlargement(relaxation_state[agent])
            inertial_velocities[sample, agent] = inertial_linear_velocity(
                model,
                states[agent],
            )

        for edge in scenario.graph:
            try:
                observation = camera.observe(
                    states[edge.observer, :3],
                    rotation_matrix_from_quaternion(states[edge.observer, 3:7]),
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

    def activate_fallback(
        observer: int,
        *,
        time: float,
        reason: str,
    ) -> None:
        if fallback_mode[observer]:
            return
        fallback_mode[observer] = True
        fallback_positions[observer] = states[observer, :3]
        fallback_times[observer] = time
        fallback_reasons[observer] = reason
        print(f"Fallback hold activated for agent {observer} at t={time:.3f} s: {reason}")

    record(0)

    for step in range(steps):
        next_states = states.copy()
        next_relaxation_state = relaxation_state.copy()

        leader = scenario.graph.root
        leader_requested_wrench = model.drift_wrench(states[leader])
        leader_allocation = allocation.bounded_least_squares(leader_requested_wrench)
        controls[step, leader] = leader_allocation.forces
        next_states[leader] = plant_integrator.step(
            model,
            states[leader],
            leader_allocation.achieved_wrench,
            dt,
        )

        for edge in scenario.graph:
            observer = edge.observer
            target = edge.target

            if fallback_mode[observer]:
                requested_wrench = stationkeeping_wrench(
                    model,
                    states[observer],
                    fallback_positions[observer],
                )
                hold_allocation = allocation.bounded_least_squares(requested_wrench)
                controls[step, observer] = hold_allocation.forces
                next_states[observer] = plant_integrator.step(
                    model,
                    states[observer],
                    hold_allocation.achieved_wrench,
                    dt,
                )
                continue

            start_time = perf_counter()

            relaxation_evaluation = None

            # Evaluate the sensing constraints before evaluating any logarithmic
            # potential.  This also gives the non-adaptive baseline a graceful
            # failure mode instead of allowing log(h_c) to raise ValueError.
            try:
                kinematics = evaluate_sensing_constraint_kinematics(
                    model,
                    camera,
                    distance_domain,
                    fov_domain,
                    states[observer],
                    states[target],
                )
            except ValueError as error:
                activate_fallback(
                    observer,
                    time=times[step],
                    reason=f"invalid sensing geometry: {error}",
                )
                requested_wrench = stationkeeping_wrench(
                    model,
                    states[observer],
                    fallback_positions[observer],
                )
                hold_allocation = allocation.bounded_least_squares(requested_wrench)
                controls[step, observer] = hold_allocation.forces
                next_states[observer] = plant_integrator.step(
                    model,
                    states[observer],
                    hold_allocation.achieved_wrench,
                    dt,
                )
                continue

            if adaptive:
                try:
                    projected, correction = relaxation.project_to_current_domain(
                        relaxation_state[observer],
                        kinematics.values,
                        enabled=enabled_relaxation,
                    )
                    relaxation_state[observer] = projected
                    if np.any(correction > 0.0):
                        guard_activation_count += 1
                        guard_maximum_correction = max(
                            guard_maximum_correction,
                            float(np.max(correction)),
                        )
                except FunnelRelaxationInfeasibleError as error:
                    activate_fallback(
                        observer,
                        time=times[step],
                        reason=str(error),
                    )
                    requested_wrench = stationkeeping_wrench(
                        model,
                        states[observer],
                        fallback_positions[observer],
                    )
                    hold_allocation = allocation.bounded_least_squares(requested_wrench)
                    controls[step, observer] = hold_allocation.forces
                    next_states[observer] = plant_integrator.step(
                        model,
                        states[observer],
                        hold_allocation.achieved_wrench,
                        dt,
                    )
                    continue
            else:
                failure_reason = conservative_domain_failure_reason(
                    kinematics.values,
                    enabled=enabled_relaxation,
                    minimum_margin=relaxation.minimum_constraint_margin,
                )
                if failure_reason is not None:
                    activate_fallback(
                        observer,
                        time=times[step],
                        reason=failure_reason,
                    )
                    requested_wrench = stationkeeping_wrench(
                        model,
                        states[observer],
                        fallback_positions[observer],
                    )
                    hold_allocation = allocation.bounded_least_squares(requested_wrench)
                    controls[step, observer] = hold_allocation.forces
                    next_states[observer] = plant_integrator.step(
                        model,
                        states[observer],
                        hold_allocation.achieved_wrench,
                        dt,
                    )
                    continue

            rho = relaxation.enlargement(relaxation_state[observer])
            potential = build_edge_potential(
                scenario,
                camera,
                distance_domain,
                fov_domain,
                observer,
                target,
                templates[observer],
                rho,
                adaptive=adaptive,
            )

            parent_velocity = inertial_linear_velocity(
                model,
                states[target],
            )

            # Primary layer: physical CLF-QP with the current funnel state
            # frozen. Neither V_s nor (partial V / partial s) v enters the
            # CLF inequality.
            try:
                evaluation = controller.evaluate(
                    follower_state=states[observer],
                    parent_position=states[target, :3],
                    parent_linear_velocity_inertial=parent_velocity,
                    edge_potential=potential,
                    filter_state=filters[observer],
                )
            except ValueError as error:
                activate_fallback(
                    observer,
                    time=times[step],
                    reason=f"CLF/potential domain evaluation failed: {error}",
                )
                requested_wrench = stationkeeping_wrench(
                    model,
                    states[observer],
                    fallback_positions[observer],
                )
                hold_allocation = allocation.bounded_least_squares(requested_wrench)
                controls[step, observer] = hold_allocation.forces
                next_states[observer] = plant_integrator.step(
                    model,
                    states[observer],
                    hold_allocation.achieved_wrench,
                    dt,
                )
                controller_times[step, observer] = perf_counter() - start_time
                continue

            # Secondary layer: independently choose v = s_dot. Its only role
            # is to preserve the logarithmic-barrier domain while otherwise
            # recovering s -> 0.
            if adaptive:
                try:
                    relaxation_evaluation = relaxation.evaluate(
                        relaxation_state[observer],
                        conservative_values=kinematics.values,
                        conservative_rates=kinematics.rates,
                        enabled=enabled_relaxation,
                        sample_time=dt,
                    )
                except FunnelRelaxationInfeasibleError as error:
                    activate_fallback(
                        observer,
                        time=times[step],
                        reason=str(error),
                    )
                    requested_wrench = stationkeeping_wrench(
                        model,
                        states[observer],
                        fallback_positions[observer],
                    )
                    hold_allocation = allocation.bounded_least_squares(requested_wrench)
                    controls[step, observer] = hold_allocation.forces
                    next_states[observer] = plant_integrator.step(
                        model,
                        states[observer],
                        hold_allocation.achieved_wrench,
                        dt,
                    )
                    controller_times[step, observer] = perf_counter() - start_time
                    continue

            controller_times[step, observer] = perf_counter() - start_time

            controls[step, observer] = controller_design.representative_thruster_forces(evaluation)
            slacks[step, observer] = evaluation.slack
            if evaluation.required_slack is not None:
                required_slacks[step, observer] = evaluation.required_slack
            if evaluation.actuation_margin is not None:
                actuation_margins[step, observer] = evaluation.actuation_margin

            # Detailed physical CLF decomposition.  This is diagnostic only;
            # it does not modify the control law.
            clf = evaluation.controller.clf
            qp_result = evaluation.controller.qp
            generalized_velocity = model.split_state(states[observer])[2]
            zeta = evaluation.generalized_configuration_gradient

            configuration_local_rate = float(zeta @ generalized_velocity)
            parent_rate = float(evaluation.potential.target_position_gradient @ parent_velocity)
            dynamics_bias = model.drift_wrench(states[observer])
            filtered_velocity = evaluation.controller.filter.output
            filtered_velocity_derivative = evaluation.controller.filter.output_derivative
            desired_velocity = evaluation.controller.desired_velocity
            unlimited_desired_velocity = evaluation.controller.unlimited_desired_velocity
            velocity_error = clf.velocity_error
            command_acceleration_wrench = model.mass_matrix @ filtered_velocity_derivative

            dynamics_bias_linear_rate = float(-velocity_error[:3] @ dynamics_bias[:3])
            dynamics_bias_angular_rate = float(-velocity_error[3:] @ dynamics_bias[3:])
            dynamics_bias_rate = float(dynamics_bias_linear_rate + dynamics_bias_angular_rate)

            command_acceleration_linear_rate = float(
                -velocity_error[:3] @ command_acceleration_wrench[:3]
            )
            command_acceleration_angular_rate = float(
                -velocity_error[3:] @ command_acceleration_wrench[3:]
            )
            command_acceleration_rate = float(
                command_acceleration_linear_rate + command_acceleration_angular_rate
            )

            velocity_backstepping_rate = float(dynamics_bias_rate + command_acceleration_rate)

            velocity_error_norm = float(np.linalg.norm(velocity_error))
            velocity_error_linear_norm = float(np.linalg.norm(velocity_error[:3]))
            velocity_error_angular_norm = float(np.linalg.norm(velocity_error[3:]))
            filtered_velocity_derivative_norm = float(np.linalg.norm(filtered_velocity_derivative))
            filtered_linear_acceleration_norm = float(
                np.linalg.norm(filtered_velocity_derivative[:3])
            )
            filtered_angular_acceleration_norm = float(
                np.linalg.norm(filtered_velocity_derivative[3:])
            )

            decay = float(controller.dynamics_controller.qp.alpha(clf.value))

            feasibility = qp_result.actuation_feasibility
            if feasibility is None:
                best_actuator_contribution = np.nan
                minimum_modeled_derivative = np.nan
                hard_residual = np.nan
            else:
                minimum_modeled_derivative = float(feasibility.minimum_modeled_derivative)
                best_actuator_contribution = float(minimum_modeled_derivative - clf.drift)
                hard_residual = float(clf.drift + decay + best_actuator_contribution)

            clf_values[step, observer] = clf.value
            clf_decays[step, observer] = decay
            clf_drifts[step, observer] = clf.drift
            clf_configuration_local_rates[step, observer] = configuration_local_rate
            clf_parent_rates[step, observer] = parent_rate
            clf_velocity_backstepping_rates[step, observer] = velocity_backstepping_rate
            clf_dynamics_bias_rates[step, observer] = dynamics_bias_rate
            clf_dynamics_bias_linear_rates[step, observer] = dynamics_bias_linear_rate
            clf_dynamics_bias_angular_rates[step, observer] = dynamics_bias_angular_rate
            clf_command_acceleration_rates[step, observer] = command_acceleration_rate
            clf_command_acceleration_linear_rates[step, observer] = command_acceleration_linear_rate
            clf_command_acceleration_angular_rates[step, observer] = (
                command_acceleration_angular_rate
            )
            clf_velocity_error_norms[step, observer] = velocity_error_norm
            clf_velocity_error_linear_norms[step, observer] = velocity_error_linear_norm
            clf_velocity_error_angular_norms[step, observer] = velocity_error_angular_norm
            clf_filtered_velocity_derivative_norms[step, observer] = (
                filtered_velocity_derivative_norm
            )
            clf_filtered_linear_acceleration_norms[step, observer] = (
                filtered_linear_acceleration_norm
            )
            clf_filtered_angular_acceleration_norms[step, observer] = (
                filtered_angular_acceleration_norm
            )
            clf_generalized_velocities[step, observer] = generalized_velocity
            clf_filtered_velocities[step, observer] = filtered_velocity
            clf_desired_velocities[step, observer] = desired_velocity
            clf_unlimited_desired_velocities[step, observer] = unlimited_desired_velocity
            clf_filtered_velocity_derivatives[step, observer] = filtered_velocity_derivative
            clf_dynamics_biases[step, observer] = dynamics_bias
            clf_best_actuator_contributions[step, observer] = best_actuator_contribution
            clf_minimum_modeled_derivatives[step, observer] = minimum_modeled_derivative
            clf_hard_residuals[step, observer] = hard_residual

            if adaptive:
                assert relaxation_evaluation is not None
                relaxation_rate = relaxation_evaluation.selected_rate
                relaxation_rate_history[step, observer] = relaxation_rate
                next_relaxation_state[observer] = np.clip(
                    relaxation_state[observer] + dt * relaxation_rate,
                    0.0,
                    1.0,
                )

            filters[observer] = filter_integrator.step(
                controller.dynamics_controller.command_filter,
                filters[observer],
                evaluation.controller.desired_velocity,
                dt,
            )
            next_states[observer] = plant_integrator.step(
                model,
                states[observer],
                evaluation.wrench_body,
                dt,
            )

        states = next_states
        relaxation_state = next_relaxation_state
        record(step + 1)

    if adaptive:
        print(
            "Sampled-data funnel safeguard: "
            f"{guard_activation_count} activations, "
            f"maximum normalized correction "
            f"{guard_maximum_correction:.3e}"
        )

    trajectory = FormationTrajectory(
        times=times,
        positions=positions,
        velocities=inertial_velocities,
        controls=controls,
        quaternions=quaternions,
    )

    return (
        scenario,
        trajectory,
        camera,
        distance_domain,
        fov_domain,
        slacks,
        required_slacks,
        actuation_margins,
        rho_history,
        relaxation_rate_history,
        image_history,
        controller_times,
        CLFDiagnosticHistory(
            value=clf_values,
            decay=clf_decays,
            drift=clf_drifts,
            configuration_local_rate=clf_configuration_local_rates,
            parent_rate=clf_parent_rates,
            velocity_backstepping_rate=(clf_velocity_backstepping_rates),
            dynamics_bias_rate=clf_dynamics_bias_rates,
            dynamics_bias_linear_rate=(clf_dynamics_bias_linear_rates),
            dynamics_bias_angular_rate=(clf_dynamics_bias_angular_rates),
            command_acceleration_rate=(clf_command_acceleration_rates),
            command_acceleration_linear_rate=(clf_command_acceleration_linear_rates),
            command_acceleration_angular_rate=(clf_command_acceleration_angular_rates),
            velocity_error_norm=clf_velocity_error_norms,
            velocity_error_linear_norm=(clf_velocity_error_linear_norms),
            velocity_error_angular_norm=(clf_velocity_error_angular_norms),
            filtered_velocity_derivative_norm=(clf_filtered_velocity_derivative_norms),
            filtered_linear_acceleration_norm=(clf_filtered_linear_acceleration_norms),
            filtered_angular_acceleration_norm=(clf_filtered_angular_acceleration_norms),
            generalized_velocity=clf_generalized_velocities,
            filtered_velocity=clf_filtered_velocities,
            desired_velocity=clf_desired_velocities,
            unlimited_desired_velocity=(clf_unlimited_desired_velocities),
            filtered_velocity_derivative=(clf_filtered_velocity_derivatives),
            dynamics_bias=clf_dynamics_biases,
            best_actuator_contribution=(clf_best_actuator_contributions),
            minimum_modeled_derivative=(clf_minimum_modeled_derivatives),
            hard_clf_residual=clf_hard_residuals,
        ),
        allocation,
        SimulationStatus(
            fallback_times=fallback_times,
            fallback_reasons=tuple(fallback_reasons),
        ),
    )


def plot_thruster_forces(
    scenario: FormationScenario,
    trajectory: FormationTrajectory,
    allocation: BlueROV2HeavyThrusterAllocation,
    *,
    control_space: BlueROV2ControlSpace,
) -> plt.Figure:
    """Plot the eight QP/representative T200 force commands."""
    if trajectory.controls is None or trajectory.controls.shape[2] != 8:
        raise ValueError("trajectory must contain eight thruster-force inputs.")

    figure, axes = plt.subplots()
    control_times = trajectory.times[:-1]

    # Plot follower commands; the leader equilibrium forces are normally small
    # and can be included as well without changing the axes.
    for thruster, name in enumerate(allocation.configuration.names):
        axes.plot(
            control_times,
            trajectory.controls[:, 1, thruster],
            label=name,
        )

    axes.axhline(
        allocation.configuration.force_limits.forward,
        linestyle="--",
        label="forward limit",
    )
    axes.axhline(
        -allocation.configuration.force_limits.reverse,
        linestyle="--",
        label="reverse limit",
    )
    axes.set_xlabel(r"$t$ [s]")
    axes.set_ylabel(r"$f_k$ [N]")
    title = "Follower 1 thruster forces"
    if control_space == "wrench":
        title += " (representative allocation)"
    axes.set_title(title)
    axes.grid(True, alpha=0.3)
    axes.legend(ncol=3)
    figure.tight_layout()
    return figure


def connection_quality_history(
    scenario: FormationScenario,
    trajectory: FormationTrajectory,
    distance_domain: DistanceDomain,
    fov_domain: FieldOfViewDomain,
    image_history: np.ndarray,
    *,
    distance_constraints: bool,
    fov_constraints: bool,
) -> np.ndarray:
    """Return normalized robust sensing quality for every directed edge.

    A quality of one means that all enabled connectivity constraints are
    satisfied within the conservative domain.  Between the conservative and
    physical boundaries the quality decreases linearly to zero.  Thus the
    animation remains green in the robust region and smoothly turns
    yellow/red as physical disconnection is approached.

    Collision avoidance is intentionally not included: edge color represents
    sensing/connectivity robustness rather than collision risk.
    """
    edges = tuple(scenario.graph.edges)
    quality = np.ones((trajectory.n_samples, len(edges)))

    for edge_index, edge in enumerate(edges):
        components = []

        if distance_constraints:
            relative = trajectory.positions[:, edge.target] - trajectory.positions[:, edge.observer]
            distance = np.linalg.norm(relative, axis=1)
            denominator = distance_domain.d_max - distance_domain.d_max_conservative
            range_quality = np.clip(
                (distance_domain.d_max - distance) / denominator,
                0.0,
                1.0,
            )
            components.append(range_quality)

        if fov_constraints:
            alpha_h = np.abs(image_history[:, edge.observer, 0])
            alpha_v = np.abs(image_history[:, edge.observer, 1])

            horizontal_quality = np.clip(
                (1.0 - alpha_h) / (1.0 - fov_domain.alpha_h_conservative),
                0.0,
                1.0,
            )
            vertical_quality = np.clip(
                (1.0 - alpha_v) / (1.0 - fov_domain.alpha_v_conservative),
                0.0,
                1.0,
            )
            components.extend((horizontal_quality, vertical_quality))

        if components:
            quality[:, edge_index] = np.nan_to_num(
                np.minimum.reduce(components),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

    return quality


def plot_distance_diagnostics(
    scenario: FormationScenario,
    trajectory: FormationTrajectory,
    distance_domain: DistanceDomain,
    rho_history: np.ndarray,
    *,
    adaptive: bool,
) -> plt.Figure:
    figure, axes = plt.subplots()

    for edge in scenario.graph:
        relative = trajectory.positions[:, edge.target] - trajectory.positions[:, edge.observer]
        distance = np.linalg.norm(relative, axis=1)
        axes.plot(
            trajectory.times,
            distance,
            label=f"distance {edge.observer} → {edge.target}",
        )

        if adaptive:
            minimum = np.array(
                [
                    distance_domain.effective_minimum_distance(value)
                    for value in rho_history[:, edge.observer, 0]
                ]
            )
            maximum = np.array(
                [
                    distance_domain.effective_maximum_distance(value)
                    for value in rho_history[:, edge.observer, 1]
                ]
            )
            axes.plot(
                trajectory.times,
                minimum,
                linestyle="--",
                label=f"effective d_min, agent {edge.observer}",
            )
            axes.plot(
                trajectory.times,
                maximum,
                linestyle="--",
                label=f"effective d_max, agent {edge.observer}",
            )

    axes.axhline(
        distance_domain.d_min_conservative,
        linestyle="--",
        label="conservative d_min",
    )
    axes.axhline(
        distance_domain.d_max_conservative,
        linestyle="--",
        label="conservative d_max",
    )
    axes.axhline(
        distance_domain.d_min,
        linestyle=":",
        label="physical d_min",
    )
    axes.axhline(
        distance_domain.d_max,
        linestyle=":",
        label="physical d_max",
    )
    axes.set_xlabel(r"$t$ [s]")
    axes.set_ylabel(r"$d_{ij}$ [m]")
    axes.set_title("Inter-agent distance")
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    return figure


def plot_fov_diagnostics(
    scenario: FormationScenario,
    trajectory: FormationTrajectory,
    fov_domain: FieldOfViewDomain,
    rho_history: np.ndarray,
    image_history: np.ndarray,
    *,
    adaptive: bool,
) -> tuple[plt.Figure, plt.Figure]:
    figure_h, axes_h = plt.subplots()

    for edge in scenario.graph:
        observer = edge.observer
        axes_h.plot(
            trajectory.times,
            image_history[:, observer, 0],
            label=f"alpha_h, agent {observer}",
        )

        if adaptive:
            limit = np.array(
                [
                    fov_domain.effective_horizontal_limit(value)
                    for value in rho_history[:, observer, 2]
                ]
            )
            axes_h.plot(
                trajectory.times,
                limit,
                linestyle="--",
                label=f"+effective limit, agent {observer}",
            )
            axes_h.plot(
                trajectory.times,
                -limit,
                linestyle="--",
                label=f"-effective limit, agent {observer}",
            )

    axes_h.axhline(
        fov_domain.alpha_h_conservative,
        linestyle="--",
        label="conservative limit",
    )
    axes_h.axhline(
        -fov_domain.alpha_h_conservative,
        linestyle="--",
    )
    axes_h.axhline(1.0, linestyle=":", label="physical limit")
    axes_h.axhline(-1.0, linestyle=":")
    axes_h.set_xlabel(r"$t$ [s]")
    axes_h.set_ylabel(r"$\alpha_h$")
    axes_h.set_title("Horizontal field of view")
    axes_h.grid(True, alpha=0.3)
    axes_h.legend()
    figure_h.tight_layout()

    figure_v, axes_v = plt.subplots()

    for edge in scenario.graph:
        observer = edge.observer
        axes_v.plot(
            trajectory.times,
            image_history[:, observer, 1],
            label=f"alpha_v, agent {observer}",
        )

        if adaptive:
            limit = np.array(
                [
                    fov_domain.effective_vertical_limit(value)
                    for value in rho_history[:, observer, 3]
                ]
            )
            axes_v.plot(
                trajectory.times,
                limit,
                linestyle="--",
                label=f"+effective limit, agent {observer}",
            )
            axes_v.plot(
                trajectory.times,
                -limit,
                linestyle="--",
                label=f"-effective limit, agent {observer}",
            )

    axes_v.axhline(
        fov_domain.alpha_v_conservative,
        linestyle="--",
        label="conservative limit",
    )
    axes_v.axhline(
        -fov_domain.alpha_v_conservative,
        linestyle="--",
    )
    axes_v.axhline(1.0, linestyle=":", label="physical limit")
    axes_v.axhline(-1.0, linestyle=":")
    axes_v.set_xlabel(r"$t$ [s]")
    axes_v.set_ylabel(r"$\alpha_v$")
    axes_v.set_title("Vertical field of view")
    axes_v.grid(True, alpha=0.3)
    axes_v.legend()
    figure_v.tight_layout()
    return figure_h, figure_v


def plot_slack(
    scenario: FormationScenario,
    trajectory: FormationTrajectory,
    slacks: np.ndarray,
    required_slacks: np.ndarray,
) -> plt.Figure:
    """Compare optimal soft-QP slack with actuator-required relaxation."""
    figure, axes = plt.subplots()
    control_times = trajectory.times[:-1]

    for edge in scenario.graph:
        observer = edge.observer
        axes.plot(
            control_times,
            slacks[:, observer],
            label=rf"agent {observer}: $\delta^\star$",
        )

        if np.any(np.isfinite(required_slacks[:, observer])):
            axes.plot(
                control_times,
                required_slacks[:, observer],
                linestyle="--",
                label=rf"agent {observer}: $\delta^{{\rm req}}$",
            )

    axes.set_xlabel(r"$t$ [s]")
    axes.set_ylabel("CLF relaxation")
    axes.set_title("Optimal and actuator-required CLF relaxation")
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    return figure


def plot_actuation_margin(
    scenario: FormationScenario,
    trajectory: FormationTrajectory,
    actuation_margins: np.ndarray,
) -> plt.Figure:
    """Plot the exact zero-slack CLF feasibility margin for the thruster box."""
    figure, axes = plt.subplots()
    control_times = trajectory.times[:-1]

    for edge in scenario.graph:
        observer = edge.observer
        if not np.any(np.isfinite(actuation_margins[:, observer])):
            continue
        axes.plot(
            control_times,
            actuation_margins[:, observer],
            label=f"agent {observer}: actuation margin",
        )

    axes.axhline(
        0.0,
        linewidth=1.0,
        linestyle="--",
        label="zero-slack feasibility boundary",
    )
    axes.set_xlabel(r"$t$ [s]")
    axes.set_ylabel("CLF actuation margin")
    axes.set_title("Actuator-limited CLF feasibility")
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    return figure


def _set_signed_log_scale(axes: plt.Axes) -> None:
    """Use a readable scale for CLF terms spanning many orders of magnitude."""
    axes.set_yscale("symlog", linthresh=1.0, linscale=1.0)


def plot_clf_value_and_decay(
    scenario: FormationScenario,
    trajectory: FormationTrajectory,
    diagnostics: CLFDiagnosticHistory,
) -> plt.Figure:
    """Plot the CLF value W and requested decay alpha(W)."""
    figure, axes = plt.subplots()
    control_times = trajectory.times[:-1]

    for edge in scenario.graph:
        observer = edge.observer
        axes.plot(
            control_times,
            diagnostics.value[:, observer],
            label=rf"agent {observer}: $W$",
        )
        axes.plot(
            control_times,
            diagnostics.decay[:, observer],
            linestyle="--",
            label=rf"agent {observer}: $\alpha(W)$",
        )

    axes.set_xlabel(r"$t$ [s]")
    axes.set_ylabel("CLF value / decay request")
    axes.set_title(r"CLF value and requested decay $\alpha(W)$")
    _set_signed_log_scale(axes)
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    return figure


def plot_clf_feasibility_balance(
    scenario: FormationScenario,
    trajectory: FormationTrajectory,
    diagnostics: CLFDiagnosticHistory,
) -> plt.Figure:
    """Plot the terms deciding zero-slack actuator feasibility."""
    figure, axes = plt.subplots()
    control_times = trajectory.times[:-1]

    for edge in scenario.graph:
        observer = edge.observer
        axes.plot(
            control_times,
            diagnostics.drift[:, observer],
            label=rf"agent {observer}: $a$",
        )
        axes.plot(
            control_times,
            diagnostics.decay[:, observer],
            linestyle="--",
            label=rf"agent {observer}: $\alpha(W)$",
        )
        axes.plot(
            control_times,
            diagnostics.best_actuator_contribution[:, observer],
            linestyle=":",
            label=rf"agent {observer}: $\min b^\top f$",
        )
        axes.plot(
            control_times,
            diagnostics.hard_clf_residual[:, observer],
            linewidth=1.5,
            label=rf"agent {observer}: $a+\alpha+\min b^\top f$",
        )

    axes.axhline(0.0, linewidth=1.0, linestyle="--")
    axes.set_xlabel(r"$t$ [s]")
    axes.set_ylabel("hard-CLF balance")
    axes.set_title("Zero-slack CLF feasibility decomposition")
    _set_signed_log_scale(axes)
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    return figure


def plot_clf_drift_components(
    scenario: FormationScenario,
    trajectory: FormationTrajectory,
    diagnostics: CLFDiagnosticHistory,
) -> plt.Figure:
    """Decompose the CLF drift a into its backstepping contributions."""
    figure, axes = plt.subplots()
    control_times = trajectory.times[:-1]

    for edge in scenario.graph:
        observer = edge.observer
        axes.plot(
            control_times,
            diagnostics.configuration_local_rate[:, observer],
            label=rf"agent {observer}: $\zeta^\top\nu$",
        )
        axes.plot(
            control_times,
            diagnostics.parent_rate[:, observer],
            linestyle="--",
            label=rf"agent {observer}: $\chi$",
        )
        axes.plot(
            control_times,
            diagnostics.velocity_backstepping_rate[:, observer],
            linestyle=":",
            label=rf"agent {observer}: velocity/backstepping",
        )
        axes.plot(
            control_times,
            diagnostics.drift[:, observer],
            linewidth=1.5,
            label=rf"agent {observer}: total $a$",
        )

    axes.axhline(0.0, linewidth=1.0, linestyle="--")
    axes.set_xlabel(r"$t$ [s]")
    axes.set_ylabel("CLF drift contribution")
    axes.set_title("Backstepping CLF drift decomposition")
    _set_signed_log_scale(axes)
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    return figure


def plot_velocity_backstepping_split(
    scenario: FormationScenario,
    trajectory: FormationTrajectory,
    diagnostics: CLFDiagnosticHistory,
) -> plt.Figure:
    """Split -e_nu^T(h + M nu_c_dot) into its two physical contributions."""
    figure, axes = plt.subplots()
    control_times = trajectory.times[:-1]

    for edge in scenario.graph:
        observer = edge.observer
        axes.plot(
            control_times,
            diagnostics.dynamics_bias_rate[:, observer],
            label=rf"agent {observer}: $-e_\nu^\top h$",
        )
        axes.plot(
            control_times,
            diagnostics.command_acceleration_rate[:, observer],
            linestyle="--",
            label=rf"agent {observer}: $-e_\nu^\top M\dot\nu_c$",
        )
        axes.plot(
            control_times,
            diagnostics.velocity_backstepping_rate[:, observer],
            linewidth=1.5,
            label=rf"agent {observer}: total",
        )

    axes.axhline(0.0, linewidth=1.0, linestyle="--")
    axes.set_xlabel(r"$t$ [s]")
    axes.set_ylabel("velocity/backstepping CLF contribution")
    axes.set_title("Velocity/backstepping drift split")
    _set_signed_log_scale(axes)
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    return figure


def plot_velocity_backstepping_peak_detail(
    trajectory: FormationTrajectory,
    required_slacks: np.ndarray,
    diagnostics: CLFDiagnosticHistory,
    *,
    half_window: float = 0.15,
) -> plt.Figure | None:
    """Zoom on the two backstepping terms around the worst CLF sample."""
    finite = np.isfinite(required_slacks)
    if not np.any(finite):
        return None

    masked = np.where(finite, required_slacks, -np.inf)
    flat_index = int(np.argmax(masked))
    step, observer = np.unravel_index(flat_index, required_slacks.shape)
    if not np.isfinite(required_slacks[step, observer]):
        return None

    control_times = trajectory.times[:-1]
    peak_time = float(control_times[step])
    mask = (control_times >= peak_time - half_window) & (control_times <= peak_time + half_window)

    figure, axes = plt.subplots()
    axes.plot(
        control_times[mask],
        diagnostics.dynamics_bias_rate[mask, observer],
        label=r"$-e_\nu^\top h$",
    )
    axes.plot(
        control_times[mask],
        diagnostics.command_acceleration_rate[mask, observer],
        linestyle="--",
        label=r"$-e_\nu^\top M\dot\nu_c$",
    )
    axes.plot(
        control_times[mask],
        diagnostics.best_actuator_contribution[mask, observer],
        linestyle=":",
        label=r"$\min b^\top f$",
    )
    axes.plot(
        control_times[mask],
        diagnostics.hard_clf_residual[mask, observer],
        linewidth=1.5,
        label="hard-CLF residual",
    )
    axes.axvline(
        peak_time,
        linewidth=1.0,
        linestyle="--",
        label=f"peak at {peak_time:.3f} s",
    )
    axes.axhline(0.0, linewidth=1.0, linestyle="--")
    axes.set_xlabel(r"$t$ [s]")
    axes.set_ylabel("CLF contribution")
    axes.set_title(f"Backstepping split near worst CLF sample — agent {observer}")
    _set_signed_log_scale(axes)
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    return figure


def plot_command_velocity_peak_detail(
    trajectory: FormationTrajectory,
    required_slacks: np.ndarray,
    diagnostics: CLFDiagnosticHistory,
    *,
    half_window: float = 0.15,
) -> plt.Figure | None:
    """Compare nu, nu_c and nu_d around the worst hard-CLF sample."""
    finite = np.isfinite(required_slacks)
    if not np.any(finite):
        return None

    masked = np.where(finite, required_slacks, -np.inf)
    flat_index = int(np.argmax(masked))
    step, observer = np.unravel_index(flat_index, required_slacks.shape)
    if not np.isfinite(required_slacks[step, observer]):
        return None

    control_times = trajectory.times[:-1]
    peak_time = float(control_times[step])
    mask = (control_times >= peak_time - half_window) & (control_times <= peak_time + half_window)

    nu = diagnostics.generalized_velocity[mask, observer]
    nu_c = diagnostics.filtered_velocity[mask, observer]
    nu_d = diagnostics.desired_velocity[mask, observer]

    figure, axes = plt.subplots()
    axes.plot(
        control_times[mask],
        np.linalg.norm(nu[:, :3], axis=1),
        label=r"$\|v\|$",
    )
    axes.plot(
        control_times[mask],
        np.linalg.norm(nu_c[:, :3], axis=1),
        linestyle="--",
        label=r"$\|v_c\|$",
    )
    axes.plot(
        control_times[mask],
        np.linalg.norm(nu_d[:, :3], axis=1),
        linestyle=":",
        label=r"$\|v_d\|$",
    )
    axes.plot(
        control_times[mask],
        np.linalg.norm(nu[:, 3:], axis=1),
        label=r"$\|\omega\|$",
    )
    axes.plot(
        control_times[mask],
        np.linalg.norm(nu_c[:, 3:], axis=1),
        linestyle="--",
        label=r"$\|\omega_c\|$",
    )
    axes.plot(
        control_times[mask],
        np.linalg.norm(nu_d[:, 3:], axis=1),
        linestyle=":",
        label=r"$\|\omega_d\|$",
    )

    axes.axvline(
        peak_time,
        linewidth=1.0,
        linestyle="--",
        label=f"peak at {peak_time:.3f} s",
    )
    axes.set_xlabel(r"$t$ [s]")
    axes.set_ylabel("generalized-velocity norm")
    axes.set_title(f"Virtual-command tracking near worst CLF sample — agent {observer}")
    axes.grid(True, alpha=0.3)
    axes.legend(ncol=2)
    figure.tight_layout()
    return figure


def plot_command_acceleration_peak_detail(
    trajectory: FormationTrajectory,
    required_slacks: np.ndarray,
    diagnostics: CLFDiagnosticHistory,
    *,
    half_window: float = 0.15,
) -> plt.Figure | None:
    """Show filtered command acceleration and velocity-error norms near the peak."""
    finite = np.isfinite(required_slacks)
    if not np.any(finite):
        return None

    masked = np.where(finite, required_slacks, -np.inf)
    flat_index = int(np.argmax(masked))
    step, observer = np.unravel_index(flat_index, required_slacks.shape)
    if not np.isfinite(required_slacks[step, observer]):
        return None

    control_times = trajectory.times[:-1]
    peak_time = float(control_times[step])
    mask = (control_times >= peak_time - half_window) & (control_times <= peak_time + half_window)

    figure, axes = plt.subplots()
    axes.plot(
        control_times[mask],
        diagnostics.filtered_linear_acceleration_norm[mask, observer],
        label=r"$\|\dot v_c\|$",
    )
    axes.plot(
        control_times[mask],
        diagnostics.filtered_angular_acceleration_norm[mask, observer],
        linestyle="--",
        label=r"$\|\dot\omega_c\|$",
    )
    axes.plot(
        control_times[mask],
        diagnostics.velocity_error_linear_norm[mask, observer],
        linestyle=":",
        label=r"$\|e_v\|$",
    )
    axes.plot(
        control_times[mask],
        diagnostics.velocity_error_angular_norm[mask, observer],
        linewidth=1.5,
        label=r"$\|e_\omega\|$",
    )

    axes.axvline(
        peak_time,
        linewidth=1.0,
        linestyle="--",
        label=f"peak at {peak_time:.3f} s",
    )
    axes.set_xlabel(r"$t$ [s]")
    axes.set_ylabel("norm")
    axes.set_title(f"Command acceleration and tracking error — agent {observer}")
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    return figure


def plot_clf_peak_detail(
    scenario: FormationScenario,
    trajectory: FormationTrajectory,
    required_slacks: np.ndarray,
    diagnostics: CLFDiagnosticHistory,
    *,
    half_window: float = 0.15,
) -> plt.Figure | None:
    """Plot a focused decomposition around the largest required-slack spike."""
    finite = np.isfinite(required_slacks)
    if not np.any(finite):
        return None

    masked = np.where(finite, required_slacks, -np.inf)
    flat_index = int(np.argmax(masked))
    step, observer = np.unravel_index(flat_index, required_slacks.shape)
    peak = required_slacks[step, observer]
    if not np.isfinite(peak) or peak <= 0.0:
        return None

    control_times = trajectory.times[:-1]
    peak_time = float(control_times[step])
    mask = (control_times >= peak_time - half_window) & (control_times <= peak_time + half_window)

    figure, axes = plt.subplots()
    axes.plot(
        control_times[mask],
        diagnostics.drift[mask, observer],
        label=r"$a$",
    )
    axes.plot(
        control_times[mask],
        diagnostics.decay[mask, observer],
        linestyle="--",
        label=r"$\alpha(W)$",
    )
    axes.plot(
        control_times[mask],
        diagnostics.best_actuator_contribution[mask, observer],
        linestyle=":",
        label=r"$\min b^\top f$",
    )
    axes.plot(
        control_times[mask],
        diagnostics.hard_clf_residual[mask, observer],
        linewidth=1.5,
        label=r"$a+\alpha+\min b^\top f$",
    )
    axes.plot(
        control_times[mask],
        required_slacks[mask, observer],
        linewidth=1.5,
        label=r"$\delta^{\rm req}$",
    )

    axes.axvline(
        peak_time,
        linewidth=1.0,
        linestyle="--",
        label=f"peak at {peak_time:.3f} s",
    )
    axes.axhline(0.0, linewidth=1.0, linestyle="--")
    axes.set_xlabel(r"$t$ [s]")
    axes.set_ylabel("CLF balance")
    axes.set_title(f"CLF infeasibility spike detail — agent {observer}")
    _set_signed_log_scale(axes)
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    return figure


def save_clf_diagnostics_csv(
    scenario: FormationScenario,
    trajectory: FormationTrajectory,
    diagnostics: CLFDiagnosticHistory,
    required_slacks: np.ndarray,
    path: Path,
) -> None:
    """Save the CLF decomposition for offline inspection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    control_times = trajectory.times[:-1]

    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "time_s",
                "agent",
                "W",
                "alpha_W",
                "drift_a",
                "configuration_local_rate",
                "parent_rate_chi",
                "velocity_backstepping_rate",
                "dynamics_bias_rate",
                "dynamics_bias_linear_rate",
                "dynamics_bias_angular_rate",
                "command_acceleration_rate",
                "command_acceleration_linear_rate",
                "command_acceleration_angular_rate",
                "velocity_error_norm",
                "velocity_error_linear_norm",
                "velocity_error_angular_norm",
                "filtered_velocity_derivative_norm",
                "filtered_linear_acceleration_norm",
                "filtered_angular_acceleration_norm",
                "nu_vx",
                "nu_vy",
                "nu_vz",
                "nu_wx",
                "nu_wy",
                "nu_wz",
                "nu_c_vx",
                "nu_c_vy",
                "nu_c_vz",
                "nu_c_wx",
                "nu_c_wy",
                "nu_c_wz",
                "nu_d_vx",
                "nu_d_vy",
                "nu_d_vz",
                "nu_d_wx",
                "nu_d_wy",
                "nu_d_wz",
                "nu_d_raw_vx",
                "nu_d_raw_vy",
                "nu_d_raw_vz",
                "nu_d_raw_wx",
                "nu_d_raw_wy",
                "nu_d_raw_wz",
                "nu_c_dot_vx",
                "nu_c_dot_vy",
                "nu_c_dot_vz",
                "nu_c_dot_wx",
                "nu_c_dot_wy",
                "nu_c_dot_wz",
                "dynamics_bias_fx",
                "dynamics_bias_fy",
                "dynamics_bias_fz",
                "dynamics_bias_tx",
                "dynamics_bias_ty",
                "dynamics_bias_tz",
                "best_actuator_contribution",
                "minimum_modeled_derivative",
                "hard_clf_residual",
                "required_slack",
            ]
        )
        for edge in scenario.graph:
            observer = edge.observer
            for step, time in enumerate(control_times):
                writer.writerow(
                    [
                        time,
                        observer,
                        diagnostics.value[step, observer],
                        diagnostics.decay[step, observer],
                        diagnostics.drift[step, observer],
                        diagnostics.configuration_local_rate[step, observer],
                        diagnostics.parent_rate[step, observer],
                        diagnostics.velocity_backstepping_rate[step, observer],
                        diagnostics.dynamics_bias_rate[step, observer],
                        diagnostics.dynamics_bias_linear_rate[step, observer],
                        diagnostics.dynamics_bias_angular_rate[step, observer],
                        diagnostics.command_acceleration_rate[step, observer],
                        diagnostics.command_acceleration_linear_rate[step, observer],
                        diagnostics.command_acceleration_angular_rate[step, observer],
                        diagnostics.velocity_error_norm[step, observer],
                        diagnostics.velocity_error_linear_norm[step, observer],
                        diagnostics.velocity_error_angular_norm[step, observer],
                        diagnostics.filtered_velocity_derivative_norm[step, observer],
                        diagnostics.filtered_linear_acceleration_norm[step, observer],
                        diagnostics.filtered_angular_acceleration_norm[step, observer],
                        *diagnostics.generalized_velocity[step, observer].tolist(),
                        *diagnostics.filtered_velocity[step, observer].tolist(),
                        *diagnostics.desired_velocity[step, observer].tolist(),
                        *diagnostics.unlimited_desired_velocity[step, observer].tolist(),
                        *diagnostics.filtered_velocity_derivative[step, observer].tolist(),
                        *diagnostics.dynamics_bias[step, observer].tolist(),
                        diagnostics.best_actuator_contribution[step, observer],
                        diagnostics.minimum_modeled_derivative[step, observer],
                        diagnostics.hard_clf_residual[step, observer],
                        required_slacks[step, observer],
                    ]
                )


def print_peak_clf_diagnostic(
    trajectory: FormationTrajectory,
    required_slacks: np.ndarray,
    diagnostics: CLFDiagnosticHistory,
) -> None:
    """Print the exact decomposition at the worst hard-CLF sample."""
    finite = np.isfinite(required_slacks)
    if not np.any(finite):
        return

    masked = np.where(finite, required_slacks, -np.inf)
    flat_index = int(np.argmax(masked))
    step, observer = np.unravel_index(flat_index, required_slacks.shape)
    peak = required_slacks[step, observer]
    if not np.isfinite(peak) or peak <= 0.0:
        return

    time = trajectory.times[step]
    print(f"Peak hard-CLF infeasibility: agent {observer} at t={time:.3f} s, delta_req={peak:.4g}")
    print(
        "  CLF balance: "
        f"W={diagnostics.value[step, observer]:.4g}, "
        f"alpha(W)={diagnostics.decay[step, observer]:.4g}, "
        f"a={diagnostics.drift[step, observer]:.4g}, "
        "min b^T f="
        f"{diagnostics.best_actuator_contribution[step, observer]:.4g}, "
        "a+alpha+min b^T f="
        f"{diagnostics.hard_clf_residual[step, observer]:.4g}"
    )
    print(
        "  Drift components: "
        "zeta^T nu="
        f"{diagnostics.configuration_local_rate[step, observer]:.4g}, "
        f"chi={diagnostics.parent_rate[step, observer]:.4g}, "
        "velocity/backstepping="
        f"{diagnostics.velocity_backstepping_rate[step, observer]:.4g}"
    )
    print(
        "  Velocity/backstepping split: "
        "-e^T h="
        f"{diagnostics.dynamics_bias_rate[step, observer]:.4g} "
        "("
        f"linear={diagnostics.dynamics_bias_linear_rate[step, observer]:.4g}, "
        f"angular={diagnostics.dynamics_bias_angular_rate[step, observer]:.4g}"
        "), "
        "-e^T M nu_c_dot="
        f"{diagnostics.command_acceleration_rate[step, observer]:.4g} "
        "("
        "linear="
        f"{diagnostics.command_acceleration_linear_rate[step, observer]:.4g}, "
        "angular="
        f"{diagnostics.command_acceleration_angular_rate[step, observer]:.4g}"
        ")"
    )
    print(
        "  Norms: "
        f"||e_nu||={diagnostics.velocity_error_norm[step, observer]:.4g}, "
        "||e_v||="
        f"{diagnostics.velocity_error_linear_norm[step, observer]:.4g}, "
        "||e_omega||="
        f"{diagnostics.velocity_error_angular_norm[step, observer]:.4g}, "
        "||nu_c_dot||="
        f"{diagnostics.filtered_velocity_derivative_norm[step, observer]:.4g}, "
        "||v_c_dot||="
        f"{diagnostics.filtered_linear_acceleration_norm[step, observer]:.4g}, "
        "||omega_c_dot||="
        f"{diagnostics.filtered_angular_acceleration_norm[step, observer]:.4g}"
    )

    array_options = {
        "precision": 3,
        "suppress_small": True,
        "separator": ", ",
    }
    print(
        "  Generalized velocities [linear | angular]:\n"
        "    nu    = "
        + np.array2string(
            diagnostics.generalized_velocity[step, observer],
            **array_options,
        )
        + "\n    nu_c  = "
        + np.array2string(
            diagnostics.filtered_velocity[step, observer],
            **array_options,
        )
        + "\n    nu_d,raw = "
        + np.array2string(
            diagnostics.unlimited_desired_velocity[step, observer],
            **array_options,
        )
        + "\n    nu_d  = "
        + np.array2string(
            diagnostics.desired_velocity[step, observer],
            **array_options,
        )
        + "\n    nu_c_dot = "
        + np.array2string(
            diagnostics.filtered_velocity_derivative[step, observer],
            **array_options,
        )
    )


def plot_domain_enlargement(
    scenario: FormationScenario,
    trajectory: FormationTrajectory,
    distance_domain: DistanceDomain,
    fov_domain: FieldOfViewDomain,
    rho_history: np.ndarray,
) -> plt.Figure:
    """Plot adaptive enlargement normalized by the physical reserve."""
    maxima = np.array(
        [
            distance_domain.collision_enlargement_max,
            distance_domain.range_enlargement_max,
            fov_domain.horizontal_enlargement_max,
            fov_domain.vertical_enlargement_max,
        ],
        dtype=float,
    )
    labels = (
        r"$\rho_{\delta}/\bar{\rho}_{\delta}$",
        r"$\rho_{\Delta}/\bar{\rho}_{\Delta}$",
        r"$\rho_h/\bar{\rho}_h$",
        r"$\rho_v/\bar{\rho}_v$",
    )

    figure, axes = plt.subplots()

    for edge in scenario.graph:
        observer = edge.observer
        for channel, label in enumerate(labels):
            if maxima[channel] <= 0.0:
                continue
            axes.plot(
                trajectory.times,
                rho_history[:, observer, channel] / maxima[channel],
                label=f"agent {observer}: {label}",
            )

    axes.axhline(
        1.0,
        linestyle=":",
        label="physical-domain limit",
    )
    axes.set_xlabel(r"$t$ [s]")
    axes.set_ylabel("normalized enlargement")
    axes.set_ylim(bottom=-0.02)
    axes.set_title("Adaptive use of conservative-to-physical reserve")
    axes.grid(True, alpha=0.3)
    axes.legend(ncol=2)
    figure.tight_layout()
    return figure


def sensing_margin_summary(
    scenario: FormationScenario,
    trajectory: FormationTrajectory,
    distance_domain: DistanceDomain,
    fov_domain: FieldOfViewDomain,
    image_history: np.ndarray,
    *,
    end_time: float | None = None,
) -> dict[str, float]:
    """Return minimum conservative and physical sensing margins.

    If ``end_time`` is supplied, only samples up to that time are included.
    This separates formation-controller performance from any later fallback
    hold behavior.
    """
    conservative_collision = np.inf
    conservative_range = np.inf
    conservative_horizontal = np.inf
    conservative_vertical = np.inf
    physical_collision = np.inf
    physical_range = np.inf
    physical_horizontal = np.inf
    physical_vertical = np.inf

    if end_time is None:
        sample_mask = np.ones(trajectory.times.shape, dtype=bool)
    else:
        sample_mask = trajectory.times <= end_time + 1e-12

    for edge in scenario.graph:
        relative = (
            trajectory.positions[sample_mask, edge.target]
            - trajectory.positions[sample_mask, edge.observer]
        )
        distance = np.linalg.norm(relative, axis=1)
        alpha_h = np.abs(image_history[sample_mask, edge.observer, 0])
        alpha_v = np.abs(image_history[sample_mask, edge.observer, 1])

        conservative_collision = min(
            conservative_collision,
            float(np.min(distance - distance_domain.d_min_conservative)),
        )
        conservative_range = min(
            conservative_range,
            float(np.min(distance_domain.d_max_conservative - distance)),
        )
        conservative_horizontal = min(
            conservative_horizontal,
            float(np.min(fov_domain.alpha_h_conservative - alpha_h)),
        )
        conservative_vertical = min(
            conservative_vertical,
            float(np.min(fov_domain.alpha_v_conservative - alpha_v)),
        )

        physical_collision = min(
            physical_collision,
            float(np.min(distance - distance_domain.d_min)),
        )
        physical_range = min(
            physical_range,
            float(np.min(distance_domain.d_max - distance)),
        )
        physical_horizontal = min(
            physical_horizontal,
            float(np.nanmin(1.0 - alpha_h)),
        )
        physical_vertical = min(
            physical_vertical,
            float(np.nanmin(1.0 - alpha_v)),
        )

    return {
        "conservative_collision": conservative_collision,
        "conservative_range": conservative_range,
        "conservative_horizontal": conservative_horizontal,
        "conservative_vertical": conservative_vertical,
        "physical_collision": physical_collision,
        "physical_range": physical_range,
        "physical_horizontal": physical_horizontal,
        "physical_vertical": physical_vertical,
    }


def plot_relaxation_rates(
    scenario: FormationScenario,
    trajectory: FormationTrajectory,
    relaxation_rate_history: np.ndarray,
) -> plt.Figure:
    """Plot the four independent domain-preserving funnel rates."""
    figure, axes = plt.subplots()
    labels = (
        r"$v_{\delta}$",
        r"$v_{\Delta}$",
        r"$v_h$",
        r"$v_v$",
    )
    control_times = trajectory.times[:-1]

    for edge in scenario.graph:
        observer = edge.observer
        for channel, label in enumerate(labels):
            axes.plot(
                control_times,
                relaxation_rate_history[:, observer, channel],
                label=f"agent {observer}: {label}",
            )

    axes.axhline(0.0, linewidth=0.8)
    axes.set_xlabel(r"$t$ [s]")
    axes.set_ylabel(r"normalized funnel rate $v_\ell$ [s$^{-1}$]")
    axes.set_title("Domain-preserving funnel rates")
    axes.grid(True, alpha=0.3)
    axes.legend(ncol=2)
    figure.tight_layout()
    return figure


def plot_controller_times(
    scenario: FormationScenario,
    trajectory: FormationTrajectory,
    controller_times: np.ndarray,
    *,
    control_space: BlueROV2ControlSpace,
) -> plt.Figure:
    """Plot wall-clock time of each complete follower controller evaluation."""
    figure, axes = plt.subplots()
    control_times = trajectory.times[:-1]

    for edge in scenario.graph:
        milliseconds = 1e3 * controller_times[:, edge.observer]
        axes.plot(
            control_times,
            milliseconds,
            label=f"agent {edge.observer}",
        )

    follower_indices = [edge.observer for edge in scenario.graph]
    samples = controller_times[:, follower_indices]
    mean_ms = 1e3 * float(np.mean(samples))
    maximum_ms = 1e3 * float(np.max(samples))

    axes.axhline(
        mean_ms,
        linestyle="--",
        label=rf"mean $={mean_ms:.2f}$ ms",
    )
    axes.set_xlabel(r"$t$ [s]")
    axes.set_ylabel("controller evaluation [ms]")
    axes.set_title(
        f"{control_space.capitalize()}-space CLF-QP computation (max {maximum_ms:.2f} ms)"
    )
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    return figure


def run_thrust_authority_sweep(
    deratings: np.ndarray,
    *,
    duration: float,
    dt: float,
    thruster_voltage: int,
    distance_constraints: bool,
    fov_constraints: bool,
    adaptive: bool,
    stress_test: bool,
    stress_scale: float,
    virtual_linear_speed_limit: float,
    virtual_angular_speed_limit: float,
    relaxation_recovery_gain: float,
    relaxation_domain_margin_ratio: float,
    slack_linear_penalty: float,
    slack_quadratic_penalty: float,
) -> list[ThrustAuthoritySweepRow]:
    """Run the same scenario across a grid of available thrust authority."""
    rows: list[ThrustAuthoritySweepRow] = []

    print(
        "derating | fallback | t_fallback [s] | max s | "
        "min physical margin | max utilization | infeasible [%]"
    )
    print("-" * 96)

    for derating in deratings:
        if not 0.0 < derating <= 1.0:
            raise ValueError("sweep deratings must lie in (0, 1].")

        (
            scenario,
            trajectory,
            _camera,
            distance_domain,
            fov_domain,
            _slacks,
            required_slacks,
            _actuation_margins,
            rho_history,
            _relaxation_rate_history,
            image_history,
            _controller_times,
            _clf_diagnostics,
            allocation,
            status,
        ) = simulate(
            duration=duration,
            dt=dt,
            thruster_voltage=thruster_voltage,
            thrust_derating=float(derating),
            control_space="thruster",
            distance_constraints=distance_constraints,
            fov_constraints=fov_constraints,
            adaptive=adaptive,
            stress_test=stress_test,
            stress_scale=stress_scale,
            slack_linear_penalty=slack_linear_penalty,
            slack_quadratic_penalty=slack_quadratic_penalty,
            virtual_linear_speed_limit=virtual_linear_speed_limit,
            virtual_angular_speed_limit=virtual_angular_speed_limit,
            relaxation_recovery_gain=relaxation_recovery_gain,
            relaxation_domain_margin_ratio=(relaxation_domain_margin_ratio),
        )

        maximum = np.array(
            [
                distance_domain.collision_enlargement_max if distance_constraints else 0.0,
                distance_domain.range_enlargement_max if distance_constraints else 0.0,
                fov_domain.horizontal_enlargement_max if fov_constraints else 0.0,
                fov_domain.vertical_enlargement_max if fov_constraints else 0.0,
            ],
            dtype=float,
        )
        active = maximum > 0.0
        normalized = np.zeros_like(rho_history)
        normalized[..., active] = rho_history[..., active] / maximum[active]
        maximum_relaxation = float(np.max(normalized))

        first_fallback = status.first_fallback_time
        margins = sensing_margin_summary(
            scenario,
            trajectory,
            distance_domain,
            fov_domain,
            image_history,
            end_time=first_fallback,
        )
        minimum_physical_margin = min(
            margins["physical_collision"],
            margins["physical_range"],
            margins["physical_horizontal"],
            margins["physical_vertical"],
        )

        follower_indices = [edge.observer for edge in scenario.graph]
        utilization_values = [
            allocation.utilization(trajectory.controls[step, agent])
            for step in range(trajectory.controls.shape[0])
            for agent in follower_indices
        ]
        maximum_utilization = float(np.max(utilization_values))

        finite_required = required_slacks[np.isfinite(required_slacks)]
        infeasible_fraction = (
            float(np.mean(finite_required > 1e-10)) if finite_required.size else 0.0
        )

        row = ThrustAuthoritySweepRow(
            derating=float(derating),
            fallback=status.fallback_occurred,
            fallback_time=first_fallback,
            maximum_normalized_relaxation=maximum_relaxation,
            minimum_physical_margin_before_fallback=(minimum_physical_margin),
            maximum_thruster_utilization=maximum_utilization,
            actuator_infeasible_fraction=infeasible_fraction,
        )
        rows.append(row)

        fallback_time_text = "-" if row.fallback_time is None else f"{row.fallback_time:.3f}"
        print(
            f"{row.derating:7.2f} | "
            f"{str(row.fallback):8s} | "
            f"{fallback_time_text:14s} | "
            f"{row.maximum_normalized_relaxation:5.3f} | "
            f"{row.minimum_physical_margin_before_fallback:19.4f} | "
            f"{row.maximum_thruster_utilization:15.3f} | "
            f"{100.0 * row.actuator_infeasible_fraction:12.2f}"
        )

    return rows


def save_thrust_authority_sweep(
    rows: list[ThrustAuthoritySweepRow],
    path: Path,
) -> None:
    """Save sweep metrics as a compact CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "derating",
                "fallback",
                "fallback_time_s",
                "maximum_normalized_relaxation",
                "minimum_physical_margin_before_fallback",
                "maximum_thruster_utilization",
                "actuator_infeasible_fraction",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.derating,
                    row.fallback,
                    "" if row.fallback_time is None else row.fallback_time,
                    row.maximum_normalized_relaxation,
                    row.minimum_physical_margin_before_fallback,
                    row.maximum_thruster_utilization,
                    row.actuator_infeasible_fraction,
                ]
            )


def plot_thrust_authority_sweep(
    rows: list[ThrustAuthoritySweepRow],
) -> plt.Figure:
    """Plot the two most useful robustness indicators versus thrust authority."""
    derating = np.array([row.derating for row in rows])
    max_relaxation = np.array([row.maximum_normalized_relaxation for row in rows])
    physical_margin = np.array([row.minimum_physical_margin_before_fallback for row in rows])

    figure, axes = plt.subplots()
    axes.plot(
        derating,
        max_relaxation,
        marker="o",
        label=r"maximum normalized relaxation $s_{\max}$",
    )
    axes.plot(
        derating,
        physical_margin,
        marker="s",
        label="minimum physical sensing margin",
    )
    axes.axhline(0.0, linewidth=0.8, linestyle="--")
    axes.axhline(1.0, linewidth=0.8, linestyle=":")
    axes.set_xlabel("available thrust fraction")
    axes.set_ylabel("robustness metric")
    axes.set_title("Thrust-authority robustness sweep")
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument(
        "--thruster-voltage",
        type=int,
        choices=(12, 16, 20),
        default=16,
        help="T200 operating point used for physical force limits",
    )
    parser.add_argument(
        "--thrust-derating",
        type=float,
        default=1.0,
        help="multiply published T200 thrust limits by a factor in (0, 1]",
    )
    parser.add_argument(
        "--control-space",
        choices=("thruster", "wrench"),
        default="thruster",
        help=(
            "optimize directly over eight thruster forces or over the "
            "six-dimensional body wrench constrained by the exact wrench polytope"
        ),
    )
    parser.add_argument(
        "--distance-constraints",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fov-constraints",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--adaptive",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--slack-linear-penalty",
        type=float,
        default=100.0,
        help=("linear slack penalty p1 in p1*delta + 0.5*p2*delta^2"),
    )
    parser.add_argument(
        "--slack-quadratic-penalty",
        type=float,
        default=5e3,
        help=("quadratic slack penalty p2 in p1*delta + 0.5*p2*delta^2"),
    )
    parser.add_argument(
        "--virtual-linear-speed-limit",
        type=float,
        default=1.5,
        help=(
            "smooth norm limit [m/s] on the translational part of the "
            "virtual gradient-descent correction"
        ),
    )
    parser.add_argument(
        "--virtual-angular-speed-limit",
        type=float,
        default=2.0,
        help=(
            "smooth norm limit [rad/s] on the rotational part of the "
            "virtual gradient-descent correction"
        ),
    )
    parser.add_argument(
        "--relaxation-recovery-gain",
        type=float,
        default=0.8,
        help="nominal exponential recovery gain in v_ref = -K_s s",
    )
    parser.add_argument(
        "--relaxation-domain-margin-ratio",
        type=float,
        default=0.1,
        help=(
            "practical adaptive-domain margin as a fraction of each "
            "conservative-to-physical h-reserve"
        ),
    )
    parser.add_argument(
        "--stress-test",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "start close to the conservative sensing boundary with "
            "outward velocity and attitude-rate disturbances"
        ),
    )
    parser.add_argument(
        "--stress-scale",
        type=float,
        default=1.0,
        help="scale stress-test initial linear/angular velocities",
    )
    parser.add_argument(
        "--thrust-authority-sweep",
        nargs="*",
        type=float,
        default=None,
        metavar="D",
        help=(
            "run an automated derating sweep instead of a single simulation. "
            "With no values, uses 0.4, 0.5, ..., 1.0."
        ),
    )
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--no-animation", action="store_true")
    parser.add_argument(
        "--save",
        action="store_true",
        help="save diagnostic figures and the formation visualization",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/bluerov2_fov_clf_qp"),
    )
    parser.add_argument(
        "--paper-quality",
        action="store_true",
        help="use publication-quality rendering and export settings",
    )
    parser.add_argument(
        "--figure-format",
        choices=("png", "pdf", "svg"),
        default=None,
        help="saved figure format; defaults to PDF for paper quality, PNG otherwise",
    )
    parser.add_argument(
        "--animation-format",
        choices=("mp4", "gif"),
        default="mp4",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="do not open interactive Matplotlib windows",
    )
    args = parser.parse_args()
    apply_visualization_style(paper_quality=args.paper_quality)

    if args.thrust_authority_sweep is not None:
        if args.thrust_authority_sweep:
            deratings = np.array(
                args.thrust_authority_sweep,
                dtype=float,
            )
        else:
            deratings = np.arange(0.4, 1.01, 0.1)

        rows = run_thrust_authority_sweep(
            deratings,
            duration=args.duration,
            dt=args.dt,
            thruster_voltage=args.thruster_voltage,
            distance_constraints=args.distance_constraints,
            fov_constraints=args.fov_constraints,
            adaptive=args.adaptive,
            stress_test=args.stress_test,
            stress_scale=args.stress_scale,
            virtual_linear_speed_limit=args.virtual_linear_speed_limit,
            virtual_angular_speed_limit=args.virtual_angular_speed_limit,
            relaxation_recovery_gain=(args.relaxation_recovery_gain),
            relaxation_domain_margin_ratio=(args.relaxation_domain_margin_ratio),
            slack_linear_penalty=args.slack_linear_penalty,
            slack_quadratic_penalty=(args.slack_quadratic_penalty),
        )
        sweep_figure = plot_thrust_authority_sweep(rows)
        if args.save:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            save_thrust_authority_sweep(
                rows,
                args.output_dir / "thrust_authority_sweep.csv",
            )
            figure_format = (
                args.figure_format
                if args.figure_format is not None
                else "pdf"
                if args.paper_quality
                else "png"
            )
            save_figure(
                sweep_figure,
                args.output_dir / f"thrust_authority_sweep.{figure_format}",
                paper_quality=args.paper_quality,
            )
        if not args.no_show:
            plt.show()
        return

    (
        scenario,
        trajectory,
        camera,
        distance_domain,
        fov_domain,
        slacks,
        required_slacks,
        actuation_margins,
        rho_history,
        relaxation_rate_history,
        image_history,
        controller_times,
        clf_diagnostics,
        allocation,
        simulation_status,
    ) = simulate(
        duration=args.duration,
        dt=args.dt,
        thruster_voltage=args.thruster_voltage,
        thrust_derating=args.thrust_derating,
        control_space=args.control_space,
        distance_constraints=args.distance_constraints,
        fov_constraints=args.fov_constraints,
        adaptive=args.adaptive,
        stress_test=args.stress_test,
        stress_scale=args.stress_scale,
        slack_linear_penalty=args.slack_linear_penalty,
        slack_quadratic_penalty=args.slack_quadratic_penalty,
        virtual_linear_speed_limit=args.virtual_linear_speed_limit,
        virtual_angular_speed_limit=args.virtual_angular_speed_limit,
        relaxation_recovery_gain=args.relaxation_recovery_gain,
        relaxation_domain_margin_ratio=(args.relaxation_domain_margin_ratio),
    )

    distance_figure = plot_distance_diagnostics(
        scenario,
        trajectory,
        distance_domain,
        rho_history,
        adaptive=args.adaptive,
    )
    fov_horizontal_figure, fov_vertical_figure = plot_fov_diagnostics(
        scenario,
        trajectory,
        fov_domain,
        rho_history,
        image_history,
        adaptive=args.adaptive,
    )
    slack_figure = plot_slack(
        scenario,
        trajectory,
        slacks,
        required_slacks,
    )
    actuation_margin_figure = plot_actuation_margin(
        scenario,
        trajectory,
        actuation_margins,
    )
    clf_value_decay_figure = plot_clf_value_and_decay(
        scenario,
        trajectory,
        clf_diagnostics,
    )
    clf_feasibility_balance_figure = plot_clf_feasibility_balance(
        scenario,
        trajectory,
        clf_diagnostics,
    )
    clf_drift_components_figure = plot_clf_drift_components(
        scenario,
        trajectory,
        clf_diagnostics,
    )
    clf_velocity_backstepping_split_figure = plot_velocity_backstepping_split(
        scenario,
        trajectory,
        clf_diagnostics,
    )
    clf_velocity_backstepping_peak_figure = plot_velocity_backstepping_peak_detail(
        trajectory,
        required_slacks,
        clf_diagnostics,
    )
    command_velocity_peak_figure = plot_command_velocity_peak_detail(
        trajectory,
        required_slacks,
        clf_diagnostics,
    )
    command_acceleration_peak_figure = plot_command_acceleration_peak_detail(
        trajectory,
        required_slacks,
        clf_diagnostics,
    )
    clf_peak_detail_figure = plot_clf_peak_detail(
        scenario,
        trajectory,
        required_slacks,
        clf_diagnostics,
    )
    domain_enlargement_figure = None
    relaxation_rate_figure = None
    if args.adaptive:
        domain_enlargement_figure = plot_domain_enlargement(
            scenario,
            trajectory,
            distance_domain,
            fov_domain,
            rho_history,
        )
        relaxation_rate_figure = plot_relaxation_rates(
            scenario,
            trajectory,
            relaxation_rate_history,
        )
    thruster_figure = plot_thruster_forces(
        scenario,
        trajectory,
        allocation,
        control_space=args.control_space,
    )
    controller_time_figure = plot_controller_times(
        scenario,
        trajectory,
        controller_times,
        control_space=args.control_space,
    )

    follower_indices = [edge.observer for edge in scenario.graph]
    controller_time_samples = controller_times[:, follower_indices]
    utilization = np.stack(
        [
            allocation.utilization(trajectory.controls[step, agent])
            for step in range(trajectory.controls.shape[0])
            for agent in follower_indices
        ]
    )
    print(
        f"{args.control_space}-space controller: "
        f"mean evaluation "
        f"{1e3 * np.mean(controller_time_samples):.3f} ms, "
        f"max evaluation "
        f"{1e3 * np.max(controller_time_samples):.3f} ms, "
        f"max representative thruster utilization "
        f"{np.max(utilization):.3f}"
    )

    sensing_margins = sensing_margin_summary(
        scenario,
        trajectory,
        distance_domain,
        fov_domain,
        image_history,
        end_time=simulation_status.first_fallback_time,
    )
    finite_required = required_slacks[np.isfinite(required_slacks)]
    finite_margin = actuation_margins[np.isfinite(actuation_margins)]

    if finite_required.size:
        print(
            "Actuation feasibility: "
            f"max required slack {np.max(finite_required):.4g}, "
            f"infeasible samples "
            f"{100.0 * np.mean(finite_required > 1e-10):.1f}%"
        )
    if finite_margin.size:
        print(f"Minimum zero-slack actuation margin: {np.min(finite_margin):.4g}")

    print_peak_clf_diagnostic(
        trajectory,
        required_slacks,
        clf_diagnostics,
    )

    minimum_physical_margin = min(
        sensing_margins["physical_collision"],
        sensing_margins["physical_range"],
        sensing_margins["physical_horizontal"],
        sensing_margins["physical_vertical"],
    )
    minimum_conservative_margin = min(
        sensing_margins["conservative_collision"],
        sensing_margins["conservative_range"],
        sensing_margins["conservative_horizontal"],
        sensing_margins["conservative_vertical"],
    )
    print(
        "Sensing margins before fallback: "
        f"minimum conservative {minimum_conservative_margin:.4g}, "
        f"minimum physical {minimum_physical_margin:.4g}"
    )

    if args.adaptive:
        maxima = np.array(
            [
                distance_domain.collision_enlargement_max,
                distance_domain.range_enlargement_max,
                fov_domain.horizontal_enlargement_max,
                fov_domain.vertical_enlargement_max,
            ],
            dtype=float,
        )
        active = maxima > 0.0
        normalized = np.zeros_like(rho_history)
        normalized[..., active] = rho_history[..., active] / maxima[active]
        print(f"Maximum normalized domain enlargement: {np.max(normalized):.3f}")

    edge_quality = connection_quality_history(
        scenario,
        trajectory,
        distance_domain,
        fov_domain,
        image_history,
        distance_constraints=args.distance_constraints,
        fov_constraints=args.fov_constraints,
    )

    desired_positions = scenario.initial_positions[scenario.graph.root] + scenario.reference.offsets

    animation = None
    vehicle_geometry = BlueROV2HeavyVisualGeometry(
        thruster_configuration=allocation.configuration
    ).wireframe()
    followers = tuple(agent for agent in range(scenario.n_agents) if agent != scenario.graph.root)

    final_formation_figure = None

    if args.no_animation:
        final_formation_figure, _ = plot_formation_3d(
            trajectory,
            scenario.graph,
            desired_positions=desired_positions,
            camera=camera,
            camera_agents=followers,
            camera_depth=0.75,
            vehicle_geometry=vehicle_geometry,
            edge_quality=edge_quality,
            paper_quality=args.paper_quality,
            show_body_forward=False,
            title="BlueROV2 camera-constrained formation",
        )
    else:
        animation = animate_formation_3d(
            trajectory,
            scenario.graph,
            desired_positions=desired_positions,
            camera=camera,
            camera_agents=followers,
            camera_depth=0.75,
            vehicle_geometry=vehicle_geometry,
            edge_quality=edge_quality,
            paper_quality=args.paper_quality,
            show_body_forward=False,
            body_axis_length=0.4,
            trail_length=250,
            frame_stride=args.frame_stride,
            title="BlueROV2 camera-constrained formation",
        )

    if args.save:
        figure_format = (
            args.figure_format
            if args.figure_format is not None
            else "pdf"
            if args.paper_quality
            else "png"
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)

        figures = {
            "distance": distance_figure,
            "fov_horizontal": fov_horizontal_figure,
            "fov_vertical": fov_vertical_figure,
            "clf_slack": slack_figure,
            "actuation_margin": actuation_margin_figure,
            "clf_value_decay": clf_value_decay_figure,
            "clf_feasibility_balance": clf_feasibility_balance_figure,
            "clf_drift_components": clf_drift_components_figure,
            "clf_velocity_backstepping_split": (clf_velocity_backstepping_split_figure),
            "thruster_forces": thruster_figure,
            "controller_time": controller_time_figure,
        }
        if domain_enlargement_figure is not None:
            figures["domain_enlargement"] = domain_enlargement_figure
        if relaxation_rate_figure is not None:
            figures["relaxation_rates"] = relaxation_rate_figure
        if clf_peak_detail_figure is not None:
            figures["clf_peak_detail"] = clf_peak_detail_figure
        if clf_velocity_backstepping_peak_figure is not None:
            figures["clf_velocity_backstepping_peak_detail"] = clf_velocity_backstepping_peak_figure
        if command_velocity_peak_figure is not None:
            figures["command_velocity_peak_detail"] = command_velocity_peak_figure
        if command_acceleration_peak_figure is not None:
            figures["command_acceleration_peak_detail"] = command_acceleration_peak_figure

        save_clf_diagnostics_csv(
            scenario,
            trajectory,
            clf_diagnostics,
            required_slacks,
            args.output_dir / "clf_diagnostics.csv",
        )

        if final_formation_figure is None:
            final_formation_figure, _ = plot_formation_3d(
                trajectory,
                scenario.graph,
                desired_positions=desired_positions,
                camera=camera,
                camera_agents=followers,
                camera_depth=0.75,
                vehicle_geometry=vehicle_geometry,
                edge_quality=edge_quality,
                paper_quality=args.paper_quality,
                show_body_forward=False,
                title="BlueROV2 camera-constrained formation",
            )

        figures["formation_final"] = final_formation_figure

        for name, figure in figures.items():
            save_figure(
                figure,
                args.output_dir / f"{name}.{figure_format}",
                paper_quality=args.paper_quality,
            )

        if animation is not None:
            save_animation(
                animation.animation,
                args.output_dir / f"formation_animation.{args.animation_format}",
                paper_quality=args.paper_quality,
                fps=30,
            )

    if not args.no_show:
        plt.show()

    _ = animation


if __name__ == "__main__":
    main()
