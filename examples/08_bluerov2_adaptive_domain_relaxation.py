"""Generate the representative adaptive-domain-relaxation paper figure.

This script reuses the exact simulation implemented in
``04_bluerov2_fov_clf_qp.py``.  It runs the adaptive stress-test case, selects
the follower/FoV channel that uses the largest normalized relaxation, and
plots that representative sensing-domain relaxation.

Run from the repository root with

    uv run python examples/08_bluerov2_adaptive_domain_relaxation.py

The default output is

    outputs/paper/adaptive_domain_relaxation.pdf
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import matplotlib.pyplot as plt
import numpy as np

from formation_control.visualization import (
    apply_visualization_style,
    plot_fov_domain_relaxation,
    save_figure,
)


def load_controller_example() -> ModuleType:
    """Load the current BlueROV example despite its numeric filename prefix."""
    path = Path(__file__).with_name("04_bluerov2_fov_clf_qp.py")
    module_name = "_bluerov2_fov_clf_qp_paper_case"

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def select_representative_fov_channel(
    scenario: object,
    fov_domain: object,
    rho_history: np.ndarray,
    *,
    observer: int | None,
    channel: str,
) -> tuple[int, str, int, int, float, float]:
    """Select the follower/channel with largest normalized FoV relaxation.

    Returns
    -------
    observer, channel_name, rho_index, alpha_index, alpha_conservative, rho_max
    """
    candidates: list[tuple[float, int, str, int, int, float, float]] = []

    for edge in scenario.graph:
        current_observer = edge.observer
        if observer is not None and current_observer != observer:
            continue

        if channel in {"auto", "horizontal"}:
            rho_max = fov_domain.horizontal_enlargement_max
            maximum_state = float(np.max(rho_history[:, current_observer, 2]) / rho_max)
            candidates.append(
                (
                    maximum_state,
                    current_observer,
                    "horizontal",
                    2,
                    0,
                    fov_domain.alpha_h_conservative,
                    rho_max,
                )
            )

        if channel in {"auto", "vertical"}:
            rho_max = fov_domain.vertical_enlargement_max
            maximum_state = float(np.max(rho_history[:, current_observer, 3]) / rho_max)
            candidates.append(
                (
                    maximum_state,
                    current_observer,
                    "vertical",
                    3,
                    1,
                    fov_domain.alpha_v_conservative,
                    rho_max,
                )
            )

    if not candidates:
        raise ValueError("the requested observer/channel is not a follower FoV channel")

    (
        _maximum_state,
        selected_observer,
        selected_channel,
        rho_index,
        alpha_index,
        alpha_conservative,
        rho_max,
    ) = max(candidates, key=lambda candidate: candidate[0])

    return (
        selected_observer,
        selected_channel,
        rho_index,
        alpha_index,
        alpha_conservative,
        rho_max,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument(
        "--thruster-voltage",
        type=int,
        choices=(12, 16, 20),
        default=16,
    )
    parser.add_argument(
        "--thrust-derating",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--stress-scale",
        type=float,
        default=1.0,
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
        "--observer",
        type=int,
        default=None,
        help=("follower to plot; by default choose the follower with the largest FoV relaxation"),
    )
    parser.add_argument(
        "--channel",
        choices=("auto", "horizontal", "vertical"),
        default="auto",
        help=("FoV channel to plot; 'auto' selects the most-relaxed channel"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/paper/adaptive_domain_relaxation.pdf"),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="also display the figure interactively",
    )
    args = parser.parse_args()

    apply_visualization_style(paper_quality=True)
    example = load_controller_example()

    (
        scenario,
        trajectory,
        _camera,
        _distance_domain,
        fov_domain,
        _slacks,
        _required_slacks,
        _actuation_margins,
        rho_history,
        _relaxation_rate_history,
        image_history,
        _controller_times,
        _clf_diagnostics,
        _allocation,
        simulation_status,
    ) = example.simulate(
        duration=args.duration,
        dt=args.dt,
        thruster_voltage=args.thruster_voltage,
        thrust_derating=args.thrust_derating,
        control_space="thruster",
        distance_constraints=True,
        fov_constraints=True,
        adaptive=True,
        stress_test=True,
        stress_scale=args.stress_scale,
        relaxation_recovery_gain=args.relaxation_recovery_gain,
        relaxation_domain_margin_ratio=(args.relaxation_domain_margin_ratio),
    )

    if simulation_status.fallback_occurred:
        raise RuntimeError(
            "adaptive paper case entered fallback at "
            f"t={simulation_status.first_fallback_time:.3f} s; "
            "not saving a nominal representative paper figure"
        )

    (
        observer,
        channel,
        rho_index,
        alpha_index,
        alpha_conservative,
        rho_max,
    ) = select_representative_fov_channel(
        scenario,
        fov_domain,
        rho_history,
        observer=args.observer,
        channel=args.channel,
    )

    alpha = image_history[:, observer, alpha_index]
    rho = rho_history[:, observer, rho_index]

    finite = np.isfinite(alpha)
    if not np.all(finite):
        raise RuntimeError("selected FoV history contains non-finite image coordinates")

    channel_symbol = r"\alpha_h" if channel == "horizontal" else r"\alpha_v"

    figure, _ = plot_fov_domain_relaxation(
        trajectory.times,
        alpha,
        rho,
        alpha_conservative=alpha_conservative,
        maximum_enlargement=rho_max,
        channel_symbol=channel_symbol,
    )

    saved = save_figure(
        figure,
        args.output,
        paper_quality=True,
    )

    maximum_normalized_relaxation = float(np.max(rho) / rho_max)
    print(
        "Representative adaptive-domain relaxation: "
        f"agent {observer}, {channel} FoV, "
        f"max s={maximum_normalized_relaxation:.3f}"
    )
    print(f"Saved paper figure to {saved}")

    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
