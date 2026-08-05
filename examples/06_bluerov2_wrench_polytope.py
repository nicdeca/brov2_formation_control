"""Inspect the achievable BlueROV2 Heavy wrench polytope."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from formation_control.actuation import (
    BlueROV2HeavyThrusterAllocation,
    wrench_polytope_from_allocation,
)
from formation_control.visualization import (
    plot_wrench_axis_authority,
    plot_wrench_polytope_projections,
    save_figure,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thruster-voltage", type=int, choices=(12, 16, 20), default=16)
    parser.add_argument("--thrust-derating", type=float, default=1.0)
    parser.add_argument("--paper-quality", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/bluerov2_wrench_polytope"))
    parser.add_argument("--figure-format", choices=("png", "pdf", "svg"), default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    allocation = BlueROV2HeavyThrusterAllocation.default_45deg(
        voltage=args.thruster_voltage, derating=args.thrust_derating
    )
    polytope = wrench_polytope_from_allocation(allocation)

    intervals = polytope.canonical_axis_intervals()
    for label, interval in zip(("Fx", "Fy", "Fz", "Mx", "My", "Mz"), intervals, strict=True):
        print(f"{label}: [{interval[0]:.3f}, {interval[1]:.3f}]")
    print(f"facets: {polytope.halfspaces.n_facets}")
    print(f"Chebyshev radius: {polytope.chebyshev_ball.radius:.5f}")
    print(
        "Chebyshev center:",
        np.array2string(polytope.chebyshev_ball.center, precision=4, suppress_small=True),
    )
    print(
        "box-midpoint wrench:",
        np.array2string(polytope.wrench_center(), precision=4, suppress_small=True),
    )

    projection_figure, _ = plot_wrench_polytope_projections(
        polytope, paper_quality=args.paper_quality
    )
    authority_figure, _ = plot_wrench_axis_authority(polytope, paper_quality=args.paper_quality)

    if args.save:
        fmt = args.figure_format or ("pdf" if args.paper_quality else "png")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        save_figure(
            projection_figure,
            args.output_dir / f"wrench_projections.{fmt}",
            paper_quality=args.paper_quality,
        )
        save_figure(
            authority_figure,
            args.output_dir / f"wrench_authority.{fmt}",
            paper_quality=args.paper_quality,
        )
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
