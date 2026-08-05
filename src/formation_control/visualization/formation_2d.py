"""Two-dimensional plotting and animation for formation trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from formation_control.graphs import DirectedSensingGraph
from formation_control.models.base import FloatArray
from formation_control.simulation import FormationTrajectory

from .style import apply_visualization_style


def _xy(positions: FloatArray) -> FloatArray:
    if positions.shape[-1] < 2:
        raise ValueError("2-D visualization requires at least two coordinates.")
    return positions[..., :2]


def _axis_limits(points: FloatArray, padding_fraction: float = 0.15) -> tuple:
    minimum = np.min(points, axis=(0, 1))
    maximum = np.max(points, axis=(0, 1))
    span = np.maximum(maximum - minimum, 1.0)
    padding = padding_fraction * span
    return (
        (float(minimum[0] - padding[0]), float(maximum[0] + padding[0])),
        (float(minimum[1] - padding[1]), float(maximum[1] + padding[1])),
    )


def plot_formation_2d(
    trajectory: FormationTrajectory,
    graph: DirectedSensingGraph,
    *,
    desired_positions: FloatArray | None = None,
    title: str | None = None,
    paper_quality: bool = False,
) -> tuple[Figure, Axes]:
    """Plot complete XY paths and the final sensing graph."""
    apply_visualization_style(paper_quality=paper_quality)
    if graph.n_agents != trajectory.n_agents:
        raise ValueError("graph and trajectory must contain the same number of agents.")

    positions = _xy(trajectory.positions)
    figure, axes = plt.subplots()

    for agent in range(trajectory.n_agents):
        axes.plot(
            positions[:, agent, 0],
            positions[:, agent, 1],
            label=f"agent {agent}",
        )
        axes.scatter(
            positions[-1, agent, 0],
            positions[-1, agent, 1],
            s=45,
        )

    if desired_positions is not None:
        desired = np.asarray(desired_positions, dtype=float)
        if desired.shape[0] != trajectory.n_agents or desired.shape[1] < 2:
            raise ValueError("desired_positions must have shape (n_agents, dimension >= 2).")
        axes.scatter(
            desired[:, 0],
            desired[:, 1],
            marker="x",
            s=65,
            label="desired",
        )

    final_positions = positions[-1]
    for edge in graph:
        observer = final_positions[edge.observer]
        target = final_positions[edge.target]
        axes.plot(
            [observer[0], target[0]],
            [observer[1], target[1]],
            linestyle="--",
            linewidth=1.0,
        )

    all_points = positions
    if desired_positions is not None:
        desired_xy = np.asarray(desired_positions, dtype=float)[None, :, :2]
        all_points = np.concatenate((positions, desired_xy), axis=0)

    x_limits, y_limits = _axis_limits(all_points)
    axes.set_xlim(*x_limits)
    axes.set_ylim(*y_limits)
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlabel(r"$x$ [m]")
    axes.set_ylabel(r"$y$ [m]")
    axes.grid(True, alpha=0.3)
    axes.legend()
    if title is not None:
        axes.set_title(title)

    return figure, axes


@dataclass(slots=True)
class FormationAnimation2D:
    """Matplotlib objects associated with an animated formation."""

    figure: Figure
    axes: Axes
    animation: FuncAnimation


def animate_formation_2d(
    trajectory: FormationTrajectory,
    graph: DirectedSensingGraph,
    *,
    desired_positions: FloatArray | None = None,
    title: str | None = None,
    interval_ms: int = 30,
    trail_length: int | None = None,
    paper_quality: bool = False,
) -> FormationAnimation2D:
    """Animate XY motion, paths, and directed sensing edges."""
    apply_visualization_style(paper_quality=paper_quality)
    if graph.n_agents != trajectory.n_agents:
        raise ValueError("graph and trajectory must contain the same number of agents.")
    if interval_ms <= 0:
        raise ValueError("interval_ms must be positive.")
    if trail_length is not None and trail_length <= 0:
        raise ValueError("trail_length must be positive when supplied.")

    positions = _xy(trajectory.positions)
    figure, axes = plt.subplots()

    all_points = positions
    desired = None
    if desired_positions is not None:
        desired = np.asarray(desired_positions, dtype=float)
        if desired.shape[0] != trajectory.n_agents or desired.shape[1] < 2:
            raise ValueError("desired_positions must have shape (n_agents, dimension >= 2).")
        all_points = np.concatenate(
            (positions, desired[None, :, :2]),
            axis=0,
        )

    x_limits, y_limits = _axis_limits(all_points)
    axes.set_xlim(*x_limits)
    axes.set_ylim(*y_limits)
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlabel(r"$x$ [m]")
    axes.set_ylabel(r"$y$ [m]")
    axes.grid(True, alpha=0.3)

    if desired is not None:
        axes.scatter(
            desired[:, 0],
            desired[:, 1],
            marker="x",
            s=65,
            label="desired",
        )

    markers = [
        axes.plot([], [], marker="o", linestyle="None", label=f"agent {agent}")[0]
        for agent in range(trajectory.n_agents)
    ]
    trails = [axes.plot([], [], linewidth=1.2, alpha=0.7)[0] for _ in range(trajectory.n_agents)]
    edges = [axes.plot([], [], linestyle="--", linewidth=1.0, alpha=0.7)[0] for _ in graph.edges]
    time_text = axes.text(
        0.02,
        0.98,
        "",
        transform=axes.transAxes,
        va="top",
    )

    axes.legend(loc="best")

    def update(frame: int):
        current = positions[frame]

        for agent, marker in enumerate(markers):
            marker.set_data(
                [current[agent, 0]],
                [current[agent, 1]],
            )

            start = 0 if trail_length is None else max(0, frame - trail_length + 1)
            trail = positions[start : frame + 1, agent]
            trails[agent].set_data(trail[:, 0], trail[:, 1])

        for artist, edge in zip(edges, graph.edges, strict=True):
            observer = current[edge.observer]
            target = current[edge.target]
            artist.set_data(
                [observer[0], target[0]],
                [observer[1], target[1]],
            )

        time_text.set_text(f"t = {trajectory.times[frame]:.2f} s")

        artists = [*markers, *trails, *edges, time_text]
        return artists

    animation = FuncAnimation(
        figure,
        update,
        frames=trajectory.n_samples,
        interval=interval_ms,
        blit=True,
        repeat=False,
    )

    if title is not None:
        axes.set_title(title)

    return FormationAnimation2D(
        figure=figure,
        axes=axes,
        animation=animation,
    )
