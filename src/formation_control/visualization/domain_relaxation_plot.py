"""Paper-quality visualization of adaptive sensing-domain relaxation."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from formation_control.models.base import FloatArray


@dataclass(frozen=True)
class FoVDomainRelaxationCurves:
    """Derived FoV limits associated with one adaptive relaxation history."""

    normalized_relaxation: FloatArray
    adaptive_limit: FloatArray


def fov_domain_relaxation_curves(
    enlargement: FloatArray,
    *,
    alpha_conservative: float,
    maximum_enlargement: float,
    domain_margin_ratio: float | None = None,
    minimum_constraint_margin: float | None = None,
) -> FoVDomainRelaxationCurves:
    """Return normalized relaxation and the corresponding FoV limits.

    The conservative FoV constraint is

        h_c = alpha_c^2 - alpha^2,

    and the adaptive constraint is

        h_a = h_c + rho.

    Therefore the instantaneous adaptive-domain boundary is

        |alpha| = sqrt(alpha_c^2 + rho).

    The positive adaptive-domain margin used by the controller is deliberately
    not represented as an additional boundary in this visualization. The
    purpose of the figure is only to show how the conservative domain is
    relaxed toward the physical sensing domain.
    """
    rho = np.asarray(enlargement, dtype=float).reshape(-1)

    if not np.all(np.isfinite(rho)):
        raise ValueError("enlargement must contain only finite values.")
    if not 0.0 < alpha_conservative < 1.0:
        raise ValueError("alpha_conservative must lie strictly in (0, 1).")
    if not np.isfinite(maximum_enlargement) or maximum_enlargement <= 0.0:
        raise ValueError("maximum_enlargement must be finite and positive.")
    if np.any(rho < -1e-10) or np.any(rho > maximum_enlargement + 1e-10):
        raise ValueError("enlargement must lie in [0, maximum_enlargement].")

    rho = np.clip(rho, 0.0, maximum_enlargement)
    normalized = rho / maximum_enlargement
    adaptive_limit = np.sqrt(alpha_conservative**2 + rho)

    return FoVDomainRelaxationCurves(
        normalized_relaxation=normalized,
        adaptive_limit=adaptive_limit,
    )


def plot_fov_domain_relaxation(
    times: FloatArray,
    alpha: FloatArray,
    enlargement: FloatArray,
    *,
    alpha_conservative: float,
    maximum_enlargement: float,
    domain_margin_ratio: float | None = None,
    minimum_constraint_margin: float | None = None,
    channel_symbol: str = r"\alpha_v",
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Plot one representative adaptive FoV relaxation history.

    The upper panel compares the realized normalized image coordinate with

    - the conservative FoV boundary;
    - the instantaneous adaptive-domain boundary;
    - the physical FoV boundary.

    The lower panel shows the normalized relaxation state

        s = rho / rho_max.

    The area between the conservative and adaptive limits highlights the
    portion of the conservative-to-physical reserve activated by the
    relaxation mechanism.

    ``domain_margin_ratio`` and ``minimum_constraint_margin`` are accepted only
    for backward compatibility with the first plotting draft. They are not
    visualized; the paper figure intentionally shows no separate
    margin-adjusted or "guarded" boundary.
    """
    times = np.asarray(times, dtype=float).reshape(-1)
    alpha = np.asarray(alpha, dtype=float).reshape(-1)
    rho = np.asarray(enlargement, dtype=float).reshape(-1)

    if not (times.size == alpha.size == rho.size):
        raise ValueError("times, alpha, and enlargement must have the same length.")
    if times.size < 2:
        raise ValueError("at least two samples are required.")
    if not (np.all(np.isfinite(times)) and np.all(np.isfinite(alpha)) and np.all(np.isfinite(rho))):
        raise ValueError("plot histories must contain only finite values.")

    curves = fov_domain_relaxation_curves(
        rho,
        alpha_conservative=alpha_conservative,
        maximum_enlargement=maximum_enlargement,
        domain_margin_ratio=domain_margin_ratio,
        minimum_constraint_margin=minimum_constraint_margin,
    )

    figure, (domain_axes, relaxation_axes) = plt.subplots(
        2,
        1,
        figsize=(6.4, 3.15),
        sharex=True,
        gridspec_kw={
            "height_ratios": (1.9, 0.75),
            "hspace": 0.05,
        },
    )

    # Plot first and use the project color cycle rather than hard-coding a
    # separate palette in this module.
    actual_line = domain_axes.plot(
        times,
        np.abs(alpha),
        linewidth=1.8,
        label=rf"$|{channel_symbol}|$",
        zorder=5,
    )[0]
    adaptive_line = domain_axes.plot(
        times,
        curves.adaptive_limit,
        linewidth=1.55,
        label=rf"${channel_symbol}^a$",
        zorder=4,
    )[0]
    conservative_line = domain_axes.axhline(
        alpha_conservative,
        linewidth=1.3,
        linestyle="--",
        label=rf"${channel_symbol}^c$",
        zorder=4,
    )
    physical_line = domain_axes.axhline(
        1.0,
        linewidth=1.3,
        linestyle=":",
        label=r"$\alpha_{\max}=1$",
        zorder=4,
    )

    domain_axes.fill_between(
        times,
        alpha_conservative,
        curves.adaptive_limit,
        where=curves.adaptive_limit >= alpha_conservative,
        color=adaptive_line.get_color(),
        alpha=0.13,
        linewidth=0.0,
        label="activated domain relaxation",
        zorder=1,
    )

    domain_axes.set_ylabel(rf"$|{channel_symbol}|$")
    domain_axes.set_ylim(
        0.0,
        max(
            1.035,
            1.04 * float(np.max(np.abs(alpha))),
        ),
    )
    domain_axes.grid(True)

    # Put the physically important curves first in the legend.
    handles = [
        actual_line,
        conservative_line,
        adaptive_line,
        physical_line,
    ]
    labels = [handle.get_label() for handle in handles]
    domain_axes.legend(
        handles,
        labels,
        loc="upper right",
        ncol=2,
        columnspacing=1.0,
        handlelength=2.4,
    )

    relaxation_line = relaxation_axes.plot(
        times,
        curves.normalized_relaxation,
        linewidth=1.7,
        label=r"normalized relaxation $s$",
    )[0]
    relaxation_axes.fill_between(
        times,
        0.0,
        curves.normalized_relaxation,
        color=relaxation_line.get_color(),
        alpha=0.12,
        linewidth=0.0,
    )
    relaxation_axes.axhline(
        1.0,
        linewidth=1.0,
        linestyle=":",
        label="full physical enlargement",
    )
    relaxation_axes.set_ylim(-0.035, 1.05)
    relaxation_axes.set_ylabel(r"$s$")
    relaxation_axes.set_xlabel(r"time $t$ [s]")
    relaxation_axes.grid(True)
    relaxation_axes.legend(loc="upper right")

    for axes in (domain_axes, relaxation_axes):
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)

    figure.align_ylabels((domain_axes, relaxation_axes))
    figure.tight_layout()

    return figure, (domain_axes, relaxation_axes)


def plot_vertical_fov_relaxation_from_histories(
    times: FloatArray,
    image_history: FloatArray,
    relaxation_state_history: FloatArray,
    *,
    observer: int,
    alpha_conservative: float,
    alpha_physical: float = 1.0,
    domain_margin_ratio: float = 0.1,
    paper_quality: bool = True,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Plot the logged vertical-FoV adaptive relaxation history.

    This is a compatibility adapter for experiment histories, which store the
    normalized relaxation state ``s`` rather than the dimensional enlargement
    ``rho`` expected by :func:`plot_fov_domain_relaxation`.

    History channel conventions are

    ``image_history[..., :] = [alpha_h, alpha_v]``

    and

    ``relaxation_state_history[..., :] =
    [s_collision, s_range, s_horizontal_fov, s_vertical_fov]``.
    """
    image_history = np.asarray(image_history, dtype=float)
    relaxation_state_history = np.asarray(
        relaxation_state_history,
        dtype=float,
    )

    if image_history.ndim != 3 or image_history.shape[2] < 2:
        raise ValueError(
            "image_history must have shape (n_samples, n_agents, >=2)."
        )
    if (
        relaxation_state_history.ndim != 3
        or relaxation_state_history.shape[2] < 4
    ):
        raise ValueError(
            "relaxation_state_history must have shape "
            "(n_samples, n_agents, >=4)."
        )
    if image_history.shape[:2] != relaxation_state_history.shape[:2]:
        raise ValueError(
            "image_history and relaxation_state_history must have matching "
            "sample and agent dimensions."
        )
    if not 0 <= observer < image_history.shape[1]:
        raise ValueError(
            f"observer index {observer} is outside the logged agent range."
        )
    if not np.isclose(alpha_physical, 1.0):
        raise ValueError(
            "The current normalized FoV plot assumes alpha_physical=1."
        )

    alpha_v = image_history[:, observer, 1]
    s_v = relaxation_state_history[:, observer, 3]

    if not np.all(np.isfinite(s_v)):
        raise ValueError(
            "vertical-FoV relaxation history must contain only finite values."
        )
    if np.any(s_v < -1e-10) or np.any(s_v > 1.0 + 1e-10):
        raise ValueError(
            "normalized vertical-FoV relaxation must lie in [0, 1]."
        )

    maximum_enlargement = alpha_physical**2 - alpha_conservative**2
    enlargement = np.clip(s_v, 0.0, 1.0) * maximum_enlargement

    # Styling remains owned by the existing project visualization stack.
    # ``paper_quality`` is accepted because the experiment plotting adapter
    # uses that common interface.
    _ = paper_quality

    return plot_fov_domain_relaxation(
        times,
        alpha_v,
        enlargement,
        alpha_conservative=alpha_conservative,
        maximum_enlargement=maximum_enlargement,
        domain_margin_ratio=domain_margin_ratio,
        channel_symbol=r"\alpha_v",
    )

