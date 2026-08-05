"""Nominal directed formation regulation with single-integrator agents."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from formation_control.graphs import DirectedSensingGraph
from formation_control.models import SingleIntegratorModel
from formation_control.potentials import EdgePotential, RelativePositionPotential
from formation_control.simulation import (
    FormationReference,
    FormationScenario,
    FormationTrajectory,
    RK4Integrator,
)
from formation_control.visualization import animate_formation_2d, plot_formation_2d


def build_scenario() -> FormationScenario:
    graph = DirectedSensingGraph.rooted_star(4)
    reference = FormationReference(
        offsets=np.array(
            [
                [0.0, 0.0, 0.0],
                [-1.2, -1.0, 0.0],
                [-1.2, 1.0, 0.0],
                [-2.2, 0.0, 0.0],
            ]
        )
    )
    return FormationScenario(
        graph=graph,
        reference=reference,
        initial_positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [-2.0, -0.2, 0.0],
                [-0.5, 2.0, 0.0],
                [-3.0, -1.2, 0.0],
            ]
        ),
    )


def simulate(
    *,
    duration: float = 6.0,
    dt: float = 0.01,
    descent_gain: float = 1.2,
) -> tuple[FormationScenario, FormationTrajectory]:
    scenario = build_scenario()
    model = SingleIntegratorModel(dimension=3)
    integrator = RK4Integrator()

    positions = scenario.initial_positions.copy()
    steps = int(np.ceil(duration / dt))
    times = np.arange(steps + 1, dtype=float) * dt

    position_history = np.empty((steps + 1, scenario.n_agents, 3))
    control_history = np.empty((steps, scenario.n_agents, 3))
    position_history[0] = positions

    edge_potentials = {
        edge.observer: EdgePotential(
            formation=RelativePositionPotential.isotropic(
                scenario.desired_relative_position(
                    edge.observer,
                    edge.target,
                ),
                gain=1.0,
            )
        )
        for edge in scenario.graph
    }

    for step in range(steps):
        controls = np.zeros_like(positions)

        for edge in scenario.graph:
            evaluation = edge_potentials[edge.observer].evaluate(
                observer_position=positions[edge.observer],
                observer_rotation=np.eye(3),
                target_position=positions[edge.target],
            )
            controls[edge.observer] = -descent_gain * evaluation.observer_position_gradient

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

    trajectory = FormationTrajectory(
        times=times,
        positions=position_history,
        controls=control_history,
    )
    return scenario, trajectory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument(
        "--no-animation",
        action="store_true",
        help="show only the final trajectory plot",
    )
    args = parser.parse_args()

    scenario, trajectory = simulate(
        duration=args.duration,
        dt=args.dt,
    )
    desired_positions = scenario.initial_positions[scenario.graph.root] + scenario.reference.offsets

    animation = None

    if args.no_animation:
        plot_formation_2d(
            trajectory,
            scenario.graph,
            desired_positions=desired_positions,
            title="Single-integrator nominal formation",
        )
    else:
        animation = animate_formation_2d(
            trajectory,
            scenario.graph,
            desired_positions=desired_positions,
            title="Single-integrator nominal formation",
            trail_length=250,
        )

    plt.show()
    _ = animation


if __name__ == "__main__":
    main()
