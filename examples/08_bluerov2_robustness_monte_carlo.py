"""Short Monte Carlo robustness campaign before ROS integration."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from moving_formation_validation import (
    sensing_margin_histories,
    simulate,
    thruster_utilization_history,
)

from formation_control.simulation.realism import RealismConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/bluerov2_moving_formation/monte_carlo_summary.csv"),
    )
    args = parser.parse_args()

    if args.runs <= 0:
        parser.error("--runs must be positive")

    rows = []
    for run in range(args.runs):
        seed = args.seed + run
        result = simulate(
            motion="trajectory",
            duration=args.duration,
            dt=args.dt,
            velocity_command=np.array([0.35, 0.08, 0.0]),
            velocity_command_bandwidth=0.2,
            thruster_voltage=16,
            thrust_derating=1.0,
            control_space="thruster",
            adaptive=True,
            use_parent_velocity_in_clf=False,
            virtual_linear_speed_limit=1.5,
            virtual_angular_speed_limit=2.0,
            relaxation_recovery_gain=0.8,
            slack_linear_penalty=100.0,
            slack_quadratic_penalty=5e3,
            realism=RealismConfig(),
            random_seed=seed,
        )

        _, physical = sensing_margin_histories(result)
        utilization = thruster_utilization_history(result)
        finite_required = result.required_slack[np.isfinite(result.required_slack)]
        max_required = float(np.max(finite_required)) if finite_required.size else 0.0
        infeasible_fraction = (
            float(np.mean(finite_required > 1e-10)) if finite_required.size else 0.0
        )

        row = [
            seed,
            float(np.nanmin(physical)),
            float(np.nanmax(result.formation_error_norm)),
            float(np.max(result.normalized_relaxation)),
            float(np.max(utilization)),
            max_required,
            infeasible_fraction,
            int(np.any(np.isfinite(result.fallback_times))),
        ]
        rows.append(row)
        print(
            f"run {run + 1:02d}/{args.runs}: "
            f"min physical {row[1]:.3f}, "
            f"max formation error {row[2]:.3f} m, "
            f"max s {row[3]:.3f}, "
            f"max util {row[4]:.3f}, "
            f"fallback {row[7]}"
        )

    values = np.asarray([row[1:] for row in rows], dtype=float)
    print("\nMonte Carlo summary")
    print(f"physical margin: mean {np.mean(values[:, 0]):.3f}, worst {np.min(values[:, 0]):.3f}")
    print(
        f"formation error: mean max {np.mean(values[:, 1]):.3f} m, "
        f"worst {np.max(values[:, 1]):.3f} m"
    )
    print(
        f"adaptive enlargement: mean max {np.mean(values[:, 2]):.3f}, "
        f"worst {np.max(values[:, 2]):.3f}"
    )
    print(
        f"thruster utilization: mean max {np.mean(values[:, 3]):.3f}, "
        f"worst {np.max(values[:, 3]):.3f}"
    )
    print(f"fallback rate: {100.0 * np.mean(values[:, 6]):.1f}%")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "seed",
                "minimum_physical_margin",
                "maximum_formation_error",
                "maximum_normalized_enlargement",
                "maximum_thruster_utilization",
                "maximum_required_slack",
                "clf_infeasible_fraction",
                "fallback",
            ]
        )
        writer.writerows(rows)


if __name__ == "__main__":
    main()
