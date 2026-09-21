"""Plot recentered logarithmic barriers for the paper and controller tuning."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from formation_control.visualization import (
    DistanceBarrierTuning,
    FoVBarrierTuning,
    plot_recentered_barrier_figure,
    plot_recentered_barrier_tuning,
    save_figure,
)


def plot_weight_sweep(
    *,
    channel: str,
    weights: list[float],
    d_min: float,
    d_max: float,
    d_des: float,
    alpha_h: float,
    alpha_v: float,
    alpha_h_des: float,
    alpha_v_des: float,
) -> plt.Figure:
    """Visualize how the barrier weight changes potential and physical gradient."""
    if not weights or any((not np.isfinite(w) or w <= 0.0) for w in weights):
        raise ValueError("weight-sweep values must be finite and positive.")

    if channel in {"collision", "range"}:
        x = np.linspace(d_min + 1e-3, d_max - 1e-3, 900)
        if channel == "collision":
            h = x - d_min
            h_d = d_des - d_min
            dh_dx = np.ones_like(x)
            xlabel = r"distance $d$ [m]"
            boundary = d_min
        else:
            h = d_max - x
            h_d = d_max - d_des
            dh_dx = -np.ones_like(x)
            xlabel = r"distance $d$ [m]"
            boundary = d_max
    else:
        limit = alpha_h if channel == "horizontal_fov" else alpha_v
        desired = alpha_h_des if channel == "horizontal_fov" else alpha_v_des
        x = np.linspace(-0.995 * limit, 0.995 * limit, 900)
        h = limit**2 - x**2
        h_d = limit**2 - desired**2
        dh_dx = -2.0 * x
        xlabel = (
            r"normalized horizontal coordinate $\alpha_h$"
            if channel == "horizontal_fov"
            else r"normalized vertical coordinate $\alpha_v$"
        )
        boundary = None

    beta = -np.log(h / h_d) + h / h_d - 1.0
    beta_prime = 1.0 / h_d - 1.0 / h

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.7))
    for weight in weights:
        axes[0].plot(x, weight * beta, label=rf"$\mu={weight:g}$")
        axes[1].plot(
            x,
            np.abs(weight * beta_prime * dh_dx),
            label=rf"$\mu={weight:g}$",
        )

    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(r"$\mu\,\overline{\beta}$")
    axes[0].set_title(f"{channel.replace('_', ' ').title()} barrier")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("absolute physical gradient")
    axes[1].set_title("Barrier action versus state")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
    if boundary is not None:
        for ax in axes:
            ax.axvline(boundary, linestyle="--", linewidth=1.0)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--d-min", type=float, default=0.8)
    parser.add_argument("--d-max", type=float, default=3.0)
    parser.add_argument("--d-des", type=float, default=1.8)
    parser.add_argument("--collision-weight", type=float, default=1.0)
    parser.add_argument("--range-weight", type=float, default=1.0)
    parser.add_argument(
        "--squared-distance",
        action="store_true",
        help="use the legacy squared-distance barrier coordinates",
    )

    parser.add_argument("--alpha-h", type=float, default=0.72)
    parser.add_argument("--alpha-v", type=float, default=0.72)
    parser.add_argument("--alpha-h-des", type=float, default=0.0)
    parser.add_argument("--alpha-v-des", type=float, default=0.0)
    parser.add_argument("--fov-h-weight", type=float, default=1.0)
    parser.add_argument("--fov-v-weight", type=float, default=1.0)

    parser.add_argument(
        "--weight-sweep",
        nargs="+",
        type=float,
        default=None,
        metavar="MU",
        help=(
            "plot a sweep of absolute barrier weights mu to visualize how "
            "early/strongly the selected barrier acts"
        ),
    )
    parser.add_argument(
        "--weight-sweep-channel",
        choices=("collision", "range", "horizontal_fov", "vertical_fov"),
        default="vertical_fov",
        help="constraint channel used by --weight-sweep",
    )
    parser.add_argument(
        "--tuning",
        action="store_true",
        help="also show/save the physical-coordinate barrier gradients",
    )
    parser.add_argument(
        "--paper-quality",
        action="store_true",
        help="use publication-oriented dimensions and rendering",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="save the generated figure(s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/recentered_barriers"),
    )
    parser.add_argument(
        "--figure-format",
        choices=("png", "pdf", "svg"),
        default=None,
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="do not open interactive Matplotlib windows",
    )
    args = parser.parse_args()

    distance = DistanceBarrierTuning(
        minimum_distance=args.d_min,
        maximum_distance=args.d_max,
        desired_distance=args.d_des,
        collision_weight=args.collision_weight,
        range_weight=args.range_weight,
        squared_distance_constraints=args.squared_distance,
    )
    fov = FoVBarrierTuning(
        horizontal_limit=args.alpha_h,
        vertical_limit=args.alpha_v,
        desired_horizontal=args.alpha_h_des,
        desired_vertical=args.alpha_v_des,
        horizontal_weight=args.fov_h_weight,
        vertical_weight=args.fov_v_weight,
    )

    paper_figure, _ = plot_recentered_barrier_figure(
        distance=distance,
        fov=fov,
        paper_quality=args.paper_quality,
    )

    tuning_figure = None
    if args.tuning:
        tuning_figure, _ = plot_recentered_barrier_tuning(
            distance=distance,
            fov=fov,
            paper_quality=args.paper_quality,
        )

    sweep_figure = None
    if args.weight_sweep is not None:
        sweep_figure = plot_weight_sweep(
            channel=args.weight_sweep_channel,
            weights=args.weight_sweep,
            d_min=args.d_min,
            d_max=args.d_max,
            d_des=args.d_des,
            alpha_h=args.alpha_h,
            alpha_v=args.alpha_v,
            alpha_h_des=args.alpha_h_des,
            alpha_v_des=args.alpha_v_des,
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

        save_figure(
            paper_figure,
            args.output_dir / f"recentered_barriers.{figure_format}",
            paper_quality=args.paper_quality,
        )

        if tuning_figure is not None:
            save_figure(
                tuning_figure,
                args.output_dir / f"recentered_barrier_tuning.{figure_format}",
                paper_quality=args.paper_quality,
            )
        if sweep_figure is not None:
            save_figure(
                sweep_figure,
                args.output_dir
                / f"barrier_weight_sweep_{args.weight_sweep_channel}.{figure_format}",
                paper_quality=args.paper_quality,
            )

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
