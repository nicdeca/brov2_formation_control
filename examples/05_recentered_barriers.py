"""Plot recentered logarithmic barriers for the paper and controller tuning."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from formation_control.visualization import (
    DistanceBarrierTuning,
    FoVBarrierTuning,
    plot_recentered_barrier_figure,
    plot_recentered_barrier_tuning,
    save_figure,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--d-min", type=float, default=0.8)
    parser.add_argument("--d-max", type=float, default=3.0)
    parser.add_argument("--d-des", type=float, default=1.8)
    parser.add_argument("--collision-weight", type=float, default=1.0)
    parser.add_argument("--range-weight", type=float, default=1.0)

    parser.add_argument("--alpha-h", type=float, default=0.72)
    parser.add_argument("--alpha-v", type=float, default=0.72)
    parser.add_argument("--alpha-h-des", type=float, default=0.0)
    parser.add_argument("--alpha-v-des", type=float, default=0.0)
    parser.add_argument("--fov-h-weight", type=float, default=1.0)
    parser.add_argument("--fov-v-weight", type=float, default=1.0)

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

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
