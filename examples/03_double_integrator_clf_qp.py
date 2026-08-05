"""Double-integrator formation control with the command-filtered CLF-QP."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from formation_control.constraints import (
    DistanceDomain,
    MaximumDistanceConstraint,
    MinimumDistanceConstraint,
)
from formation_control.control import (
    CLFQP,
    AdaptiveDomainDynamics,
    AdaptiveEnlargementLaw,
    BacksteppingCLF,
    DoubleIntegratorAgentController,
    FirstOrderCommandFilter,
    LinearClassK,
    PolyhedralControlSet,
    SecondOrderCLFQPController,
)
from formation_control.graphs import DirectedSensingGraph
from formation_control.models import DoubleIntegratorModel
from formation_control.potentials import (
    AdaptiveConstraintBarrierPotential,
    ConstraintBarrierPotential,
    EdgePotential,
    RelativePositionPotential,
)
from formation_control.simulation import (
    FormationReference,
    FormationScenario,
    FormationTrajectory,
    RK4Integrator,
)
from formation_control.visualization import animate_formation_2d, plot_formation_2d


@dataclass(frozen=True)
class DistanceBarrierTemplates:
    collision: AdaptiveConstraintBarrierPotential[np.ndarray] | None
    sensing_range: AdaptiveConstraintBarrierPotential[np.ndarray] | None


def build_scenario() -> FormationScenario:
    graph = DirectedSensingGraph.rooted_star(3)
    reference = FormationReference(
        offsets=np.array(
            [
                [0.0, 0.0, 0.0],
                [-1.5, -0.8, 0.0],
                [-1.6, 0.9, 0.0],
            ]
        )
    )
    return FormationScenario(
        graph=graph,
        reference=reference,
        initial_positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [-1.0, -0.55, 0.0],
                [-2.55, 1.0, 0.0],
            ]
        ),
    )


def build_agent_controller(
    acceleration_limit: float,
) -> DoubleIntegratorAgentController:
    dynamics_controller = SecondOrderCLFQPController(
        virtual_gain=1.3 * np.eye(3),
        command_filter=FirstOrderCommandFilter(
            signal_dim=3,
            bandwidth=4.0,
        ),
        clf=BacksteppingCLF(np.eye(3)),
        qp=CLFQP.isotropic(
            control_dim=3,
            control_weight=1.0,
            slack_penalty=2e3,
            alpha=LinearClassK(gain=1.5),
            control_set=PolyhedralControlSet.box(
                -acceleration_limit,
                acceleration_limit,
                dimension=3,
            ),
        ),
    )
    return DoubleIntegratorAgentController(
        dynamics_controller=dynamics_controller,
    )


def build_adaptation(
    domain: DistanceDomain,
) -> AdaptiveDomainDynamics:
    return AdaptiveDomainDynamics(
        collision=AdaptiveEnlargementLaw(
            maximum=domain.collision_enlargement_max,
            expansion_gain=0.8,
            recovery_gain=0.15,
        ),
        range=AdaptiveEnlargementLaw(
            maximum=domain.range_enlargement_max,
            expansion_gain=0.8,
            recovery_gain=0.15,
        ),
        horizontal_fov=AdaptiveEnlargementLaw(
            maximum=0.0,
            expansion_gain=0.0,
            recovery_gain=0.0,
        ),
        vertical_fov=AdaptiveEnlargementLaw(
            maximum=0.0,
            expansion_gain=0.0,
            recovery_gain=0.0,
        ),
        slack_threshold=0.02,
    )


def build_barrier_templates(
    scenario: FormationScenario,
    domain: DistanceDomain,
    observer: int,
    target: int,
    *,
    collision: bool,
    sensing_range: bool,
) -> DistanceBarrierTemplates:
    desired = scenario.desired_relative_position(observer, target)

    collision_template = None
    if collision:
        collision_template = AdaptiveConstraintBarrierPotential(
            constraint=MinimumDistanceConstraint(domain.d_min_conservative),
            reference_state=desired,
            weight=0.3,
        )

    range_template = None
    if sensing_range:
        range_template = AdaptiveConstraintBarrierPotential(
            constraint=MaximumDistanceConstraint(domain.d_max_conservative),
            reference_state=desired,
            weight=0.3,
        )

    return DistanceBarrierTemplates(
        collision=collision_template,
        sensing_range=range_template,
    )


def edge_potential(
    scenario: FormationScenario,
    observer: int,
    target: int,
    domain: DistanceDomain,
    templates: DistanceBarrierTemplates,
    rho_collision: float,
    rho_range: float,
    *,
    adaptive: bool,
) -> EdgePotential:
    desired = scenario.desired_relative_position(observer, target)

    collision_barrier = None
    if templates.collision is not None:
        if adaptive:
            collision_barrier = templates.collision.bind(rho_collision)
        else:
            collision_barrier = ConstraintBarrierPotential.from_reference(
                MinimumDistanceConstraint(domain.d_min_conservative),
                desired,
                weight=0.3,
            )

    range_barrier = None
    if templates.sensing_range is not None:
        if adaptive:
            range_barrier = templates.sensing_range.bind(rho_range)
        else:
            range_barrier = ConstraintBarrierPotential.from_reference(
                MaximumDistanceConstraint(domain.d_max_conservative),
                desired,
                weight=0.3,
            )

    return EdgePotential(
        formation=RelativePositionPotential.isotropic(
            desired,
            gain=1.0,
        ),
        collision_barrier=collision_barrier,
        range_barrier=range_barrier,
    )


def simulate(
    *,
    duration: float = 8.0,
    dt: float = 0.01,
    acceleration_limit: float = 2.0,
    collision: bool = True,
    sensing_range: bool = True,
    adaptive: bool = False,
) -> tuple[
    FormationScenario,
    FormationTrajectory,
    DistanceDomain,
    np.ndarray,
    np.ndarray,
]:
    scenario = build_scenario()
    domain = DistanceDomain(
        d_min=0.5,
        d_max=3.5,
        d_min_conservative=0.8,
        d_max_conservative=3.0,
    )
    model = DoubleIntegratorModel(dimension=3)
    controller = build_agent_controller(acceleration_limit)
    adaptation = build_adaptation(domain)

    plant_integrator = RK4Integrator()
    filter_integrator = RK4Integrator()
    adaptation_integrator = RK4Integrator()

    states = np.hstack(
        (
            scenario.initial_positions,
            np.zeros_like(scenario.initial_positions),
        )
    )

    templates = {
        edge.observer: build_barrier_templates(
            scenario,
            domain,
            edge.observer,
            edge.target,
            collision=collision,
            sensing_range=sensing_range,
        )
        for edge in scenario.graph
    }

    rho = np.zeros((scenario.n_agents, 4))
    filters: dict[int, np.ndarray] = {}

    for edge in scenario.graph:
        potential = edge_potential(
            scenario,
            edge.observer,
            edge.target,
            domain,
            templates[edge.observer],
            0.0,
            0.0,
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
    velocities = np.empty_like(positions)
    controls = np.zeros((steps, scenario.n_agents, 3))
    slacks = np.zeros((steps, scenario.n_agents))
    rho_history = np.zeros((steps + 1, scenario.n_agents, 4))

    positions[0] = states[:, :3]
    velocities[0] = states[:, 3:]
    rho_history[0] = rho

    for step in range(steps):
        next_states = states.copy()
        next_rho = rho.copy()

        for edge in scenario.graph:
            observer = edge.observer
            target = edge.target
            adaptive_state = adaptation.split_state(rho[observer])

            potential = edge_potential(
                scenario,
                observer,
                target,
                domain,
                templates[observer],
                adaptive_state.collision,
                adaptive_state.range,
                adaptive=adaptive,
            )

            evaluation = controller.evaluate(
                follower_state=states[observer],
                parent_position=states[target, :3],
                parent_velocity=states[target, 3:],
                edge_potential=potential,
                filter_state=filters[observer],
            )

            controls[step, observer] = evaluation.acceleration
            slacks[step, observer] = evaluation.slack

            filters[observer] = filter_integrator.step(
                controller.dynamics_controller.command_filter,
                filters[observer],
                evaluation.controller.desired_velocity,
                dt,
            )
            next_states[observer] = plant_integrator.step(
                model,
                states[observer],
                evaluation.acceleration,
                dt,
            )

            if adaptive:
                next_rho[observer] = adaptation_integrator.step(
                    adaptation,
                    rho[observer],
                    np.array([evaluation.slack]),
                    dt,
                )

        states = next_states
        rho = next_rho
        positions[step + 1] = states[:, :3]
        velocities[step + 1] = states[:, 3:]
        rho_history[step + 1] = rho

    return (
        scenario,
        FormationTrajectory(
            times=times,
            positions=positions,
            velocities=velocities,
            controls=controls,
        ),
        domain,
        slacks,
        rho_history,
    )


def plot_diagnostics(
    scenario: FormationScenario,
    trajectory: FormationTrajectory,
    domain: DistanceDomain,
    slacks: np.ndarray,
    rho_history: np.ndarray,
    *,
    adaptive: bool,
) -> None:
    figure, axes = plt.subplots()
    control_times = trajectory.times[:-1]

    for edge in scenario.graph:
        axes.plot(
            control_times,
            slacks[:, edge.observer],
            label=f"agent {edge.observer}",
        )

    axes.set_xlabel("time [s]")
    axes.set_ylabel("CLF-QP slack")
    axes.set_title("CLF relaxation")
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()

    if adaptive:
        figure, axes = plt.subplots()
        for edge in scenario.graph:
            collision_limit = np.array(
                [
                    domain.effective_minimum_distance(value)
                    for value in rho_history[:, edge.observer, 0]
                ]
            )
            range_limit = np.array(
                [
                    domain.effective_maximum_distance(value)
                    for value in rho_history[:, edge.observer, 1]
                ]
            )
            axes.plot(
                trajectory.times,
                collision_limit,
                label=f"d_min eff, agent {edge.observer}",
            )
            axes.plot(
                trajectory.times,
                range_limit,
                label=f"d_max eff, agent {edge.observer}",
            )

        axes.axhline(domain.d_min, linestyle=":", label="physical d_min")
        axes.axhline(domain.d_max, linestyle=":", label="physical d_max")
        axes.set_xlabel("time [s]")
        axes.set_ylabel("effective distance limit")
        axes.set_title("Adaptive admissible domain")
        axes.grid(True, alpha=0.3)
        axes.legend()
        figure.tight_layout()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--acceleration-limit", type=float, default=2.0)
    parser.add_argument(
        "--collision",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--range",
        dest="sensing_range",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--adaptive",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--no-animation", action="store_true")
    args = parser.parse_args()

    scenario, trajectory, domain, slacks, rho_history = simulate(
        duration=args.duration,
        dt=args.dt,
        acceleration_limit=args.acceleration_limit,
        collision=args.collision,
        sensing_range=args.sensing_range,
        adaptive=args.adaptive,
    )

    desired_positions = scenario.initial_positions[scenario.graph.root] + scenario.reference.offsets
    plot_diagnostics(
        scenario,
        trajectory,
        domain,
        slacks,
        rho_history,
        adaptive=args.adaptive,
    )

    animation = None

    if args.no_animation:
        plot_formation_2d(
            trajectory,
            scenario.graph,
            desired_positions=desired_positions,
            title="Double-integrator command-filtered CLF-QP",
        )
    else:
        animation = animate_formation_2d(
            trajectory,
            scenario.graph,
            desired_positions=desired_positions,
            title="Double-integrator command-filtered CLF-QP",
            trail_length=250,
        )

    plt.show()
    _ = animation


if __name__ == "__main__":
    main()
