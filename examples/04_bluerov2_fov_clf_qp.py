"""3-D BlueROV2 formation control with camera/FoV barriers and a CLF-QP.

The leader is held at a fixed 6-DoF equilibrium by applying its exact model
drift wrench.  Two BlueROV2 followers observe the leader through forward
cameras and regulate desired relative positions while keeping the target
centered in the image.

The example supports conservative distance/FoV barriers and optional
slack-driven enlargement toward the physical admissible domain.
"""

from __future__ import annotations

import argparse
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
)
from formation_control.control import (
    AdaptiveDomainDynamics,
    AdaptiveEnlargementLaw,
    BlueROV2ControllerDesign,
    BlueROV2ControlSpace,
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


def build_scenario() -> FormationScenario:
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

    return FormationScenario(
        graph=graph,
        reference=reference,
        initial_positions=np.array(
            [
                [0.0, 0.0, -1.0],
                [-2.25, -0.25, -0.8],
                [-1.55, 1.25, -1.7],
            ],
            dtype=float,
        ),
    )


def build_initial_states(
    scenario: FormationScenario,
) -> np.ndarray:
    states = np.zeros((scenario.n_agents, 13), dtype=float)

    states[0, :3] = scenario.initial_positions[0]
    states[0, 3:7] = quaternion_from_roll_pitch_yaw(0.0, 0.0, 0.0)

    orientation_offsets = {
        1: (np.deg2rad(10.0), np.deg2rad(-5.0)),
        2: (np.deg2rad(-9.0), np.deg2rad(5.0)),
    }

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
) -> BlueROV2ControllerDesign:
    """Build the same controller in thruster or exact wrench coordinates."""
    return build_bluerov2_controller_design(
        model,
        allocation,
        control_space=control_space,
    )


def build_adaptation(
    distance_domain: DistanceDomain,
    fov_domain: FieldOfViewDomain,
    *,
    distance_constraints: bool,
    fov_constraints: bool,
) -> AdaptiveDomainDynamics:
    def law(maximum: float, enabled: bool) -> AdaptiveEnlargementLaw:
        return AdaptiveEnlargementLaw(
            maximum=maximum if enabled else 0.0,
            expansion_gain=0.7 if enabled else 0.0,
            recovery_gain=0.12 if enabled else 0.0,
        )

    return AdaptiveDomainDynamics(
        collision=law(
            distance_domain.collision_enlargement_max,
            distance_constraints,
        ),
        range=law(
            distance_domain.range_enlargement_max,
            distance_constraints,
        ),
        horizontal_fov=law(
            fov_domain.horizontal_enlargement_max,
            fov_constraints,
        ),
        vertical_fov=law(
            fov_domain.vertical_enlargement_max,
            fov_constraints,
        ),
        slack_threshold=0.02,
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
    BlueROV2HeavyThrusterAllocation,
]:
    scenario = build_scenario()
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
    )
    controller = controller_design.agent_controller
    adaptation = build_adaptation(
        distance_domain,
        fov_domain,
        distance_constraints=distance_constraints,
        fov_constraints=fov_constraints,
    )

    plant_integrator = RK4Integrator()
    filter_integrator = RK4Integrator()
    adaptation_integrator = RK4Integrator()

    states = build_initial_states(scenario)
    rho = np.zeros((scenario.n_agents, 4), dtype=float)

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

    filters: dict[int, np.ndarray] = {}
    for edge in scenario.graph:
        potential = build_edge_potential(
            scenario,
            camera,
            distance_domain,
            fov_domain,
            edge.observer,
            edge.target,
            templates[edge.observer],
            rho[edge.observer],
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
    controller_times = np.zeros((steps, scenario.n_agents))
    rho_history = np.zeros((steps + 1, scenario.n_agents, 4))
    image_history = np.full(
        (steps + 1, scenario.n_agents, 2),
        np.nan,
    )

    def record(sample: int) -> None:
        positions[sample] = states[:, :3]
        quaternions[sample] = states[:, 3:7]
        rho_history[sample] = rho

        for agent in range(scenario.n_agents):
            inertial_velocities[sample, agent] = inertial_linear_velocity(
                model,
                states[agent],
            )

        for edge in scenario.graph:
            observation = camera.observe(
                states[edge.observer, :3],
                rotation_matrix_from_quaternion(states[edge.observer, 3:7]),
                states[edge.target, :3],
            )
            image_history[sample, edge.observer] = np.array(
                [
                    observation.image_point.alpha_h,
                    observation.image_point.alpha_v,
                ]
            )

    record(0)

    for step in range(steps):
        next_states = states.copy()
        next_rho = rho.copy()

        # The leader equilibrium wrench is allocated through the same physical
        # eight-thruster map used by the followers.
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

            potential = build_edge_potential(
                scenario,
                camera,
                distance_domain,
                fov_domain,
                observer,
                target,
                templates[observer],
                rho[observer],
                adaptive=adaptive,
            )

            parent_velocity = inertial_linear_velocity(
                model,
                states[target],
            )

            start_time = perf_counter()
            evaluation = controller.evaluate(
                follower_state=states[observer],
                parent_position=states[target, :3],
                parent_linear_velocity_inertial=parent_velocity,
                edge_potential=potential,
                filter_state=filters[observer],
            )
            controller_times[step, observer] = perf_counter() - start_time

            controls[step, observer] = controller_design.representative_thruster_forces(evaluation)
            slacks[step, observer] = evaluation.slack
            if evaluation.required_slack is not None:
                required_slacks[step, observer] = evaluation.required_slack

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

            if adaptive:
                adaptation_signal = (
                    evaluation.required_slack
                    if evaluation.required_slack is not None
                    else evaluation.slack
                )
                next_rho[observer] = adaptation_integrator.step(
                    adaptation,
                    rho[observer],
                    np.array([adaptation_signal]),
                    dt,
                )

        states = next_states
        rho = next_rho
        record(step + 1)

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
        rho_history,
        image_history,
        controller_times,
        allocation,
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
            quality[:, edge_index] = np.minimum.reduce(components)

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

    (
        scenario,
        trajectory,
        camera,
        distance_domain,
        fov_domain,
        slacks,
        required_slacks,
        rho_history,
        image_history,
        controller_times,
        allocation,
    ) = simulate(
        duration=args.duration,
        dt=args.dt,
        thruster_voltage=args.thruster_voltage,
        thrust_derating=args.thrust_derating,
        control_space=args.control_space,
        distance_constraints=args.distance_constraints,
        fov_constraints=args.fov_constraints,
        adaptive=args.adaptive,
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
            "thruster_forces": thruster_figure,
            "controller_time": controller_time_figure,
        }

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
