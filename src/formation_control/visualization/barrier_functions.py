"""Publication and tuning plots for recentered logarithmic barriers."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from formation_control.models.base import FloatArray
from formation_control.potentials import RecenteredLogBarrier

from .style import MATLAB_COLORS, apply_visualization_style


@dataclass(frozen=True)
class DistanceBarrierTuning:
    """Parameters for collision and sensing-range recentered barriers."""

    minimum_distance: float = 0.8
    maximum_distance: float = 3.0
    desired_distance: float = 1.8
    collision_weight: float = 1.0
    range_weight: float = 1.0
    squared_distance_constraints: bool = False

    def __post_init__(self) -> None:
        if self.minimum_distance < 0.0:
            raise ValueError("minimum_distance must be nonnegative.")
        if self.maximum_distance <= self.minimum_distance:
            raise ValueError("maximum_distance must be larger than minimum_distance.")
        if not (self.minimum_distance < self.desired_distance < self.maximum_distance):
            raise ValueError("desired_distance must lie strictly inside the admissible interval.")
        if self.collision_weight < 0.0 or self.range_weight < 0.0:
            raise ValueError("barrier weights must be nonnegative.")


@dataclass(frozen=True)
class FoVBarrierTuning:
    """Parameters for normalized horizontal and vertical FoV barriers."""

    horizontal_limit: float = 0.72
    vertical_limit: float = 0.72
    desired_horizontal: float = 0.0
    desired_vertical: float = 0.0
    horizontal_weight: float = 1.0
    vertical_weight: float = 1.0

    def __post_init__(self) -> None:
        for name, limit in (
            ("horizontal_limit", self.horizontal_limit),
            ("vertical_limit", self.vertical_limit),
        ):
            if not 0.0 < limit <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1].")

        if abs(self.desired_horizontal) >= self.horizontal_limit:
            raise ValueError("desired_horizontal must lie strictly inside the horizontal FoV.")
        if abs(self.desired_vertical) >= self.vertical_limit:
            raise ValueError("desired_vertical must lie strictly inside the vertical FoV.")
        if self.horizontal_weight < 0.0 or self.vertical_weight < 0.0:
            raise ValueError("barrier weights must be nonnegative.")


@dataclass(frozen=True)
class DistanceBarrierSweep:
    distance: FloatArray
    collision: FloatArray
    sensing_range: FloatArray
    total: FloatArray
    derivative_collision: FloatArray
    derivative_range: FloatArray
    derivative_total: FloatArray


@dataclass(frozen=True)
class FoVBarrierSweep:
    coordinate: FloatArray
    horizontal: FloatArray
    vertical: FloatArray
    derivative_horizontal: FloatArray
    derivative_vertical: FloatArray


def _barrier_values(
    constraint_values: FloatArray,
    reference_value: float,
    *,
    weight: float,
) -> FloatArray:
    barrier = RecenteredLogBarrier(reference_value)
    values = np.asarray(constraint_values, dtype=float)
    return weight * np.array(
        [barrier.value(float(value)) for value in values],
        dtype=float,
    )


def _barrier_derivatives(
    constraint_values: FloatArray,
    reference_value: float,
    *,
    weight: float,
) -> FloatArray:
    barrier = RecenteredLogBarrier(reference_value)
    values = np.asarray(constraint_values, dtype=float)
    return weight * np.array(
        [barrier.derivative(float(value)) for value in values],
        dtype=float,
    )


def distance_barrier_sweep(
    tuning: DistanceBarrierTuning,
    *,
    n_samples: int = 1200,
    boundary_fraction: float = 0.003,
) -> DistanceBarrierSweep:
    """Evaluate collision/range barriers across the admissible distance interval."""
    if n_samples < 3:
        raise ValueError("n_samples must be at least three.")
    if not 0.0 < boundary_fraction < 0.5:
        raise ValueError("boundary_fraction must lie in (0, 0.5).")

    interval = tuning.maximum_distance - tuning.minimum_distance
    epsilon = boundary_fraction * interval
    distance = np.linspace(
        tuning.minimum_distance + epsilon,
        tuning.maximum_distance - epsilon,
        n_samples,
    )

    if tuning.squared_distance_constraints:
        h_collision = distance**2 - tuning.minimum_distance**2
        h_collision_reference = tuning.desired_distance**2 - tuning.minimum_distance**2
        dh_collision_dd = 2.0 * distance

        h_range = tuning.maximum_distance**2 - distance**2
        h_range_reference = tuning.maximum_distance**2 - tuning.desired_distance**2
        dh_range_dd = -2.0 * distance
    else:
        # Paper/default formulation:
        # h_delta = d - d_min^c, h_Delta = d_max^c - d.
        h_collision = distance - tuning.minimum_distance
        h_collision_reference = tuning.desired_distance - tuning.minimum_distance
        dh_collision_dd = np.ones_like(distance)

        h_range = tuning.maximum_distance - distance
        h_range_reference = tuning.maximum_distance - tuning.desired_distance
        dh_range_dd = -np.ones_like(distance)

    collision = _barrier_values(
        h_collision,
        h_collision_reference,
        weight=tuning.collision_weight,
    )
    derivative_collision = (
        _barrier_derivatives(
            h_collision,
            h_collision_reference,
            weight=tuning.collision_weight,
        )
        * dh_collision_dd
    )

    sensing_range = _barrier_values(
        h_range,
        h_range_reference,
        weight=tuning.range_weight,
    )
    derivative_range = (
        _barrier_derivatives(
            h_range,
            h_range_reference,
            weight=tuning.range_weight,
        )
        * dh_range_dd
    )

    return DistanceBarrierSweep(
        distance=distance,
        collision=collision,
        sensing_range=sensing_range,
        total=collision + sensing_range,
        derivative_collision=derivative_collision,
        derivative_range=derivative_range,
        derivative_total=derivative_collision + derivative_range,
    )


def _single_fov_sweep(
    coordinate: FloatArray,
    *,
    limit: float,
    desired: float,
    weight: float,
) -> tuple[FloatArray, FloatArray]:
    constraint = limit**2 - coordinate**2
    reference = limit**2 - desired**2

    values = _barrier_values(
        constraint,
        reference,
        weight=weight,
    )
    derivative = (
        _barrier_derivatives(
            constraint,
            reference,
            weight=weight,
        )
        * (-2.0)
        * coordinate
    )
    return values, derivative


def fov_barrier_sweep(
    tuning: FoVBarrierTuning,
    *,
    n_samples: int = 1200,
    boundary_fraction: float = 0.003,
) -> FoVBarrierSweep:
    """Evaluate horizontal/vertical FoV barriers on normalized image coordinates."""
    if n_samples < 3:
        raise ValueError("n_samples must be at least three.")
    if not 0.0 < boundary_fraction < 0.5:
        raise ValueError("boundary_fraction must lie in (0, 0.5).")

    maximum_limit = max(tuning.horizontal_limit, tuning.vertical_limit)
    coordinate = np.linspace(
        -maximum_limit * (1.0 - boundary_fraction),
        maximum_limit * (1.0 - boundary_fraction),
        n_samples,
    )

    horizontal = np.full_like(coordinate, np.nan)
    derivative_horizontal = np.full_like(coordinate, np.nan)
    horizontal_mask = np.abs(coordinate) < tuning.horizontal_limit
    (
        horizontal[horizontal_mask],
        derivative_horizontal[horizontal_mask],
    ) = _single_fov_sweep(
        coordinate[horizontal_mask],
        limit=tuning.horizontal_limit,
        desired=tuning.desired_horizontal,
        weight=tuning.horizontal_weight,
    )

    vertical = np.full_like(coordinate, np.nan)
    derivative_vertical = np.full_like(coordinate, np.nan)
    vertical_mask = np.abs(coordinate) < tuning.vertical_limit
    (
        vertical[vertical_mask],
        derivative_vertical[vertical_mask],
    ) = _single_fov_sweep(
        coordinate[vertical_mask],
        limit=tuning.vertical_limit,
        desired=tuning.desired_vertical,
        weight=tuning.vertical_weight,
    )

    return FoVBarrierSweep(
        coordinate=coordinate,
        horizontal=horizontal,
        vertical=vertical,
        derivative_horizontal=derivative_horizontal,
        derivative_vertical=derivative_vertical,
    )


def _panel_label(axes: Axes, label: str) -> None:
    axes.text(
        0.02,
        0.96,
        label,
        transform=axes.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )


def plot_recentered_barrier_figure(
    *,
    distance: DistanceBarrierTuning | None = None,
    fov: FoVBarrierTuning | None = None,
    scalar_reference: float = 1.5,
    scalar_h_max: float | None = None,
    paper_quality: bool = False,
) -> tuple[Figure, tuple[Axes, Axes, Axes]]:
    """Create a three-panel paper figure for the recentered barriers.

    Panel (a) compares the original logarithmic barrier with its recentered
    counterpart. Panels (b) and (c) show the resulting barriers in the
    physical distance and normalized-image coordinates used by the controller.
    """
    if distance is None:
        distance = DistanceBarrierTuning()
    if fov is None:
        fov = FoVBarrierTuning()

    if not np.isfinite(scalar_reference) or scalar_reference <= 0.0:
        raise ValueError("scalar_reference must be finite and positive.")

    if scalar_h_max is None:
        scalar_h_max = 2.5 * scalar_reference
    if not np.isfinite(scalar_h_max) or scalar_h_max <= scalar_reference:
        raise ValueError("scalar_h_max must be finite and larger than scalar_reference.")

    apply_visualization_style(paper_quality=paper_quality)

    figure, axes_array = plt.subplots(
        1,
        3,
        figsize=(7.15, 2.35) if paper_quality else (10.5, 3.2),
    )
    axes = tuple(axes_array)

    # ------------------------------------------------------------------
    # (a) Original and recentered scalar logarithmic barriers.
    # ------------------------------------------------------------------
    h_min = max(0.01 * scalar_reference, 1e-4)
    h = np.linspace(h_min, scalar_h_max, 1200)

    original = -np.log(h)
    recentered = original + np.log(scalar_reference) + (h - scalar_reference) / scalar_reference

    axes[0].plot(
        h,
        recentered,
        linewidth=2.0,
        label=r"$\overline{\beta}(h;h^d)$",
    )
    axes[0].plot(
        h,
        original,
        linestyle="--",
        linewidth=1.3,
        label=r"$\beta(h)=-\ln(h)$",
    )
    axes[0].axvline(
        scalar_reference,
        linestyle=":",
        linewidth=1.0,
        color="0.35",
    )
    axes[0].axhline(
        0.0,
        linewidth=0.7,
        color="0.55",
    )
    axes[0].scatter(
        [scalar_reference],
        [0.0],
        s=24,
        zorder=5,
        color=MATLAB_COLORS[4],
    )
    axes[0].text(
        scalar_reference,
        0.04,
        r"$h=h^d$",
        transform=axes[0].get_xaxis_transform(),
        ha="center",
        va="bottom",
    )
    axes[0].set_xlabel(r"$h$")
    axes[0].set_ylabel("barrier value")
    axes[0].set_xlim(h[0], h[-1])
    axes[0].set_ylim(bottom=min(-1.0, float(np.nanmin(original)) - 0.1))
    axes[0].grid(True)
    axes[0].legend(loc="upper right")
    _panel_label(axes[0], "(a)")

    # ------------------------------------------------------------------
    # (b) Distance-domain barriers.
    # ------------------------------------------------------------------
    distance_sweep = distance_barrier_sweep(distance)

    axes[1].plot(
        distance_sweep.distance,
        distance_sweep.collision,
        label=r"$V_{\delta}$",
    )
    axes[1].plot(
        distance_sweep.distance,
        distance_sweep.sensing_range,
        label=r"$V_{\Delta}$",
    )
    axes[1].plot(
        distance_sweep.distance,
        distance_sweep.total,
        linewidth=2.0,
        label=r"$V_{\delta}+V_{\Delta}$",
    )
    axes[1].axvline(
        distance.minimum_distance,
        linestyle="--",
        linewidth=1.0,
        color="0.35",
    )
    axes[1].axvline(
        distance.maximum_distance,
        linestyle="--",
        linewidth=1.0,
        color="0.35",
    )
    axes[1].axvline(
        distance.desired_distance,
        linestyle=":",
        linewidth=1.1,
        color=MATLAB_COLORS[4],
    )
    axes[1].scatter(
        [distance.desired_distance],
        [0.0],
        s=24,
        zorder=5,
        color=MATLAB_COLORS[4],
    )
    axes[1].text(
        distance.minimum_distance,
        0.03,
        r"$d_{\min}^c$",
        transform=axes[1].get_xaxis_transform(),
        ha="left",
        va="bottom",
    )
    axes[1].text(
        distance.maximum_distance,
        0.03,
        r"$d_{\max}^c$",
        transform=axes[1].get_xaxis_transform(),
        ha="right",
        va="bottom",
    )
    axes[1].text(
        distance.desired_distance,
        0.03,
        r"$d_{ij}^d$",
        transform=axes[1].get_xaxis_transform(),
        ha="center",
        va="bottom",
        color=MATLAB_COLORS[4],
    )
    axes[1].set_xlabel(r"$d_{ij}$ [m]")
    axes[1].set_ylabel(r"$V_d$")
    axes[1].set_xlim(
        distance.minimum_distance,
        distance.maximum_distance,
    )
    axes[1].set_ylim(bottom=0.0)
    axes[1].grid(True)
    axes[1].legend(loc="upper center")
    _panel_label(axes[1], "(b)")

    # ------------------------------------------------------------------
    # (c) FoV-domain barriers.
    # ------------------------------------------------------------------
    fov_sweep = fov_barrier_sweep(fov)

    equal_curves = (
        np.isclose(fov.horizontal_limit, fov.vertical_limit)
        and np.isclose(fov.desired_horizontal, fov.desired_vertical)
        and np.isclose(fov.horizontal_weight, fov.vertical_weight)
    )

    if equal_curves:
        axes[2].plot(
            fov_sweep.coordinate,
            fov_sweep.horizontal,
            linewidth=2.0,
            label=r"$V_{\mathrm{FoV}}$",
        )
        limit_values = (fov.horizontal_limit,)
        desired_values = (fov.desired_horizontal,)
    else:
        axes[2].plot(
            fov_sweep.coordinate,
            fov_sweep.horizontal,
            label=r"$V_h$",
        )
        axes[2].plot(
            fov_sweep.coordinate,
            fov_sweep.vertical,
            label=r"$V_v$",
        )
        limit_values = (
            fov.horizontal_limit,
            fov.vertical_limit,
        )
        desired_values = (
            fov.desired_horizontal,
            fov.desired_vertical,
        )

    for limit_value in dict.fromkeys(limit_values):
        axes[2].axvline(
            -limit_value,
            linestyle="--",
            linewidth=1.0,
            color="0.35",
        )
        axes[2].axvline(
            limit_value,
            linestyle="--",
            linewidth=1.0,
            color="0.35",
        )

    for desired_value in dict.fromkeys(desired_values):
        axes[2].axvline(
            desired_value,
            linestyle=":",
            linewidth=1.1,
            color=MATLAB_COLORS[4],
        )

    axes[2].scatter(
        [fov.desired_horizontal],
        [0.0],
        s=24,
        zorder=5,
        color=MATLAB_COLORS[4],
    )
    axes[2].text(
        -fov.horizontal_limit,
        0.03,
        r"$-\alpha^c$",
        transform=axes[2].get_xaxis_transform(),
        ha="left",
        va="bottom",
    )
    axes[2].text(
        fov.horizontal_limit,
        0.03,
        r"$\alpha^c$",
        transform=axes[2].get_xaxis_transform(),
        ha="right",
        va="bottom",
    )
    if np.isclose(fov.desired_horizontal, 0.0):
        axes[2].text(
            0.0,
            0.03,
            r"$\alpha^d=0$",
            transform=axes[2].get_xaxis_transform(),
            ha="center",
            va="bottom",
            color=MATLAB_COLORS[4],
        )
    axes[2].set_xlabel(r"normalized image coordinate $\alpha$")
    axes[2].set_ylabel(r"$V_{\mathrm{FoV}}$")
    axes[2].set_ylim(bottom=0.0)
    axes[2].grid(True)
    axes[2].legend(loc="upper center")
    _panel_label(axes[2], "(c)")

    figure.tight_layout(pad=0.55, w_pad=0.8)
    return figure, axes


def plot_recentered_barrier_tuning(
    *,
    distance: DistanceBarrierTuning | None = None,
    fov: FoVBarrierTuning | None = None,
    paper_quality: bool = False,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Plot physical-coordinate barrier gradients for controller tuning."""
    if distance is None:
        distance = DistanceBarrierTuning()
    if fov is None:
        fov = FoVBarrierTuning()

    apply_visualization_style(paper_quality=paper_quality)

    figure, axes_array = plt.subplots(
        1,
        2,
        figsize=(7.15, 2.7) if paper_quality else (8.5, 3.4),
    )
    axes = tuple(axes_array)

    distance_sweep = distance_barrier_sweep(distance)
    axes[0].plot(
        distance_sweep.distance,
        distance_sweep.derivative_collision,
        label=r"$\partial V_{\delta}/\partial d_{ij}$",
    )
    axes[0].plot(
        distance_sweep.distance,
        distance_sweep.derivative_range,
        label=r"$\partial V_{\Delta}/\partial d_{ij}$",
    )
    axes[0].plot(
        distance_sweep.distance,
        distance_sweep.derivative_total,
        linewidth=2.0,
        label=r"$\partial V_d/\partial d_{ij}$",
    )
    axes[0].axhline(
        0.0,
        linewidth=0.7,
        color="0.55",
    )
    axes[0].axvline(
        distance.desired_distance,
        linestyle=":",
        linewidth=1.1,
        color=MATLAB_COLORS[4],
    )
    axes[0].set_xlabel(r"$d_{ij}$ [m]")
    axes[0].set_ylabel("potential gradient")
    axes[0].set_title("Distance-barrier sensitivity")
    axes[0].grid(True)
    axes[0].legend()

    fov_sweep = fov_barrier_sweep(fov)

    axes[1].plot(
        fov_sweep.coordinate,
        fov_sweep.derivative_horizontal,
        label=r"$\partial V_h/\partial \alpha_h$",
    )

    if not (
        np.isclose(fov.horizontal_limit, fov.vertical_limit)
        and np.isclose(fov.desired_horizontal, fov.desired_vertical)
        and np.isclose(fov.horizontal_weight, fov.vertical_weight)
    ):
        axes[1].plot(
            fov_sweep.coordinate,
            fov_sweep.derivative_vertical,
            label=r"$\partial V_v/\partial \alpha_v$",
        )

    axes[1].axhline(
        0.0,
        linewidth=0.7,
        color="0.55",
    )
    axes[1].axvline(
        fov.desired_horizontal,
        linestyle=":",
        linewidth=1.1,
        color=MATLAB_COLORS[4],
    )
    axes[1].set_xlabel(r"normalized image coordinate $\alpha$")
    axes[1].set_ylabel("potential gradient")
    axes[1].set_title("FoV-barrier sensitivity")
    axes[1].grid(True)
    axes[1].legend()

    figure.tight_layout()
    return figure, axes
