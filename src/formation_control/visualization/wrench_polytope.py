"""Visualization helpers for achievable 6-D wrench polytopes."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from formation_control.actuation import AchievableWrenchPolytope

from .style import MATLAB_COLORS, apply_visualization_style

WRENCH_LABELS = (
    r"$F_x$ [N]",
    r"$F_y$ [N]",
    r"$F_z$ [N]",
    r"$M_x$ [N m]",
    r"$M_y$ [N m]",
    r"$M_z$ [N m]",
)


def plot_wrench_axis_authority(polytope: AchievableWrenchPolytope, *, paper_quality: bool = False):
    apply_visualization_style(paper_quality=paper_quality)
    intervals = polytope.canonical_axis_intervals()
    labels = (r"$F_x$", r"$F_y$", r"$F_z$", r"$M_x$", r"$M_y$", r"$M_z$")
    indices = np.arange(6)
    figure, axes = plt.subplots(figsize=(7.15, 2.8) if paper_quality else (8.5, 3.5))
    width = 0.36
    axes.bar(indices - width / 2, -intervals[:, 0], width, label="negative-direction magnitude")
    axes.bar(indices + width / 2, intervals[:, 1], width, label="positive-direction magnitude")
    axes.set_xticks(indices)
    axes.set_xticklabels(labels)
    axes.set_ylabel("maximum achievable magnitude")
    axes.set_title("BlueROV2 Heavy canonical wrench authority")
    axes.grid(True, axis="y")
    axes.legend()
    figure.tight_layout()
    return figure, axes


def _plot_projection(axes, polytope, first, second, *, title):
    projection = polytope.projection(first, second)
    vertices = projection.vertices
    closed = np.vstack((vertices, vertices[0]))
    axes.fill(closed[:, 0], closed[:, 1], alpha=0.18)
    axes.plot(closed[:, 0], closed[:, 1], linewidth=1.5)
    center = polytope.chebyshev_ball.center
    axes.scatter(
        [center[first]],
        [center[second]],
        marker="x",
        s=45,
        color=MATLAB_COLORS[6],
        label="Chebyshev center",
    )
    axes.axhline(0.0, linewidth=0.7, color="0.55")
    axes.axvline(0.0, linewidth=0.7, color="0.55")
    axes.set_xlabel(WRENCH_LABELS[first])
    axes.set_ylabel(WRENCH_LABELS[second])
    axes.set_title(title)
    axes.set_aspect("equal", adjustable="box")
    axes.grid(True)


def plot_wrench_polytope_projections(
    polytope: AchievableWrenchPolytope, *, paper_quality: bool = False
):
    apply_visualization_style(paper_quality=paper_quality)
    figure, arr = plt.subplots(1, 3, figsize=(7.15, 2.55) if paper_quality else (10.2, 3.25))
    axes = tuple(arr)
    _plot_projection(axes[0], polytope, 0, 1, title="Horizontal force")
    _plot_projection(axes[1], polytope, 3, 4, title="Roll/pitch torque")
    _plot_projection(axes[2], polytope, 2, 5, title="Vertical force / yaw")
    axes[2].legend(loc="best")
    figure.tight_layout()
    return figure, axes
