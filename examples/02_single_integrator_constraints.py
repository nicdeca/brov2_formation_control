"""Single-integrator formation control with switchable distance barriers."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from formation_control.constraints import (
    DistanceDomain,
    MaximumDistanceConstraint,
    MinimumDistanceConstraint,
)
from formation_control.graphs import DirectedSensingGraph
from formation_control.models import SingleIntegratorModel
from formation_control.potentials import (
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


def build_scenario() -> FormationScenario:
    graph = DirectedSensingGraph.rooted_star(3)
    reference = FormationReference(
        offsets=np.array(
            [
                [0.0, 0.0, 0.0],
                [-1.5, 0.0, 0.0],
                [0.0, -2.0, 0.0],
            ]
        )
    )
    return FormationScenario(
        graph=graph,
        reference=reference,
        initial_positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [-0.9, 0.05, 0.0],
                [0.15, -2.85, 0.0],
            ]
        ),
    )


def build_edge_potential(
    scenario: FormationScenario,
    observer: int,
    target: int,
    domain: DistanceDomain,
    *,
    collision: bool,
    sensing_range: bool,
) -> EdgePotential:
    desired_relative = scenario.desired_relative_position(observer, target)

    collision_barrier = None
    if collision:
        collision_barrier = ConstraintBarrierPotential.from_reference(
            MinimumDistanceConstraint(domain.d_min_conservative),
            desired_relative,
            weight=0.35,
        )

    range_barrier = None
    if sensing_range:
        range_barrier = ConstraintBarrierPotential.from_reference(
            MaximumDistanceConstraint(domain.d_max_conservative),
            desired_relative,
            weight=0.35,
        )

    return EdgePotential(
        formation=RelativePositionPotential.isotropic(
            desired_relative,
            gain=1.0,
        ),
        collision_barrier=collision_barrier,
        range_barrier=range_barrier,
    )


def simulate(
    *,
    duration: float = 8.0,
    dt: float = 0.005,
    collision: bool = True,
    sensing_range: bool = True,
) -> tuple[FormationScenario, FormationTrajectory, DistanceDomain]:
    scenario = build_scenario()
    domain = DistanceDomain(
        d_min=0.5,
        d_max=3.5,
        d_min_conservative=0.8,
        d_max_conservative=3.0,
    )
    model = SingleIntegratorModel(dimension=3)
    integrator = RK4Integrator()

    potentials = {
        edge.observer: build_edge_potential(
            scenario,
            edge.observer,
            edge.target,
            domain,
            collision=collision,
            sensing_range=sensing_range,
        )
        for edge in scenario.graph
    }

    positions = scenario.initial_positions.copy()
    steps = int(np.ceil(duration / dt))
    times = np.arange(steps + 1, dtype=float) * dt
    position_history = np.empty((steps + 1, scenario.n_agents, 3))
    control_history = np.empty((steps, scenario.n_agents, 3))
    position_history[0] = positions

    for step in range(steps):
        controls = np.zeros_like(positions)

        for edge in scenario.graph:
            evaluation = potentials[edge.observer].evaluate(
                observer_position=positions[edge.observer],
                observer_rotation=np.eye(3),
                target_position=positions[edge.target],
            )
            controls[edge.observer] = -evaluation.observer_position_gradient

        next_positions = positions.copy()
        for agent in range(scenario.n_agents):
            next_positions[agent] = integrator.step(
                model,
                positions[agent],
                controls[agent],
                dt,
            )

        control_history[step] = controls
        positions = next_positions
        position_history[step + 1] = positions

    return (
        scenario,
        FormationTrajectory(
            times=times,
            positions=position_history,
            controls=control_history,
        ),
        domain,
    )


def plot_edge_distances(
    scenario: FormationScenario,
    trajectory: FormationTrajectory,
    domain: DistanceDomain,
) -> None:
    figure, axes = plt.subplots()

    for edge in scenario.graph:
        relative = trajectory.positions[:, edge.target] - trajectory.positions[:, edge.observer]
        distance = np.linalg.norm(relative, axis=1)
        axes.plot(
            trajectory.times,
            distance,
            label=f"{edge.observer} → {edge.target}",
        )

    axes.axhline(
        domain.d_min_conservative,
        linestyle="--",
        label="conservative minimum",
    )
    axes.axhline(
        domain.d_max_conservative,
        linestyle="--",
        label="conservative maximum",
    )
    axes.axhline(domain.d_min, linestyle=":", label="physical minimum")
    axes.axhline(domain.d_max, linestyle=":", label="physical maximum")
    axes.set_xlabel("time [s]")
    axes.set_ylabel("edge distance")
    axes.set_title("Distance constraints")
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--dt", type=float, default=0.005)
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
    parser.add_argument("--no-animation", action="store_true")
    args = parser.parse_args()

    scenario, trajectory, domain = simulate(
        duration=args.duration,
        dt=args.dt,
        collision=args.collision,
        sensing_range=args.sensing_range,
    )
    desired_positions = scenario.initial_positions[scenario.graph.root] + scenario.reference.offsets

    plot_edge_distances(scenario, trajectory, domain)

    animation = None

    if args.no_animation:
        plot_formation_2d(
            trajectory,
            scenario.graph,
            desired_positions=desired_positions,
            title="Single-integrator constrained formation",
        )
    else:
        animation = animate_formation_2d(
            trajectory,
            scenario.graph,
            desired_positions=desired_positions,
            title="Single-integrator constrained formation",
            trail_length=300,
        )

    plt.show()
    _ = animation


if __name__ == "__main__":
    main()
