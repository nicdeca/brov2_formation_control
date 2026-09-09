#!/usr/bin/env python3
"""Plot every available exported phase of one experiment run.

Use this as the maintained run-level plotting entry point.

Behavior:
- initialization history only -> plot initialization;
- mission history only        -> plot mission;
- both                        -> plot both;
- neither                     -> fail clearly.

For each available phase, the PX4 / EKF / transformed-raw-MoCap comparison is
also attempted unless ``--no-estimator-comparison`` is supplied.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(
    command: list[str],
    label: str,
    *,
    required: bool = True,
) -> bool:
    print(f"\n=== {label} ===", flush=True)
    result = subprocess.run(command, check=False)
    if result.returncode == 0:
        return True

    message = (
        f"{label} returned exit code {result.returncode}: "
        + " ".join(command)
    )
    if required:
        raise RuntimeError(message)

    print(f"WARNING: {message}")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--mission-preset",
        choices=("paper", "all"),
        default="paper",
        help="mission plot preset; default: paper",
    )
    parser.add_argument(
        "--show-legends",
        action="store_true",
        help="show legends on compact mission formation/sensing plots",
    )
    parser.add_argument(
        "--format",
        choices=("pdf", "png", "svg"),
        default="pdf",
        help="figure format for mission and estimator-comparison plots",
    )
    parser.add_argument(
        "--no-estimator-comparison",
        action="store_true",
        help="skip estimator-comparison figures for all available phases",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run directory does not exist: {run_dir}")

    initialization_history = (
        run_dir / "initialization" / "initialization_history.npz"
    )
    mission_history = run_dir / "mission" / "formation_history.npz"

    available = []
    if initialization_history.exists():
        available.append("initialization")
    if mission_history.exists():
        available.append("mission")

    if not available:
        raise FileNotFoundError(
            "No exported phase history was found. Run "
            "`python3 scripts/export_experiment.py <RUN>` first. "
            "Expected at least one of:\n"
            f"  - {initialization_history}\n"
            f"  - {mission_history}"
        )

    scripts_dir = Path(__file__).resolve().parent
    initialization_plotter = (
        scripts_dir / "plot_initialization_experiment.py"
    )
    mission_plotter = scripts_dir / "plot_formation_experiment.py"
    estimator_plotter = (
        scripts_dir / "plot_state_estimator_comparison.py"
    )

    plotted = []

    if "initialization" in available:
        _run(
            [
                sys.executable,
                str(initialization_plotter),
                str(initialization_history),
                "--save",
            ],
            "Plot initialization",
        )
        plotted.append("initialization")

        if not args.no_estimator_comparison:
            # Comparison arrays may legitimately be unavailable for an old
            # recording. Do not prevent the normal initialization plots from
            # being produced in that case.
            _run(
                [
                    sys.executable,
                    str(estimator_plotter),
                    str(initialization_history),
                    "--save",
                    "--format",
                    args.format,
                ],
                "Compare state estimators (initialization)",
                required=False,
            )

    if "mission" in available:
        # The mission plotter can invoke the estimator comparison by itself
        # when used standalone. Suppress that here so the run-level wrapper
        # performs exactly one comparison for this phase.
        mission_command = [
            sys.executable,
            str(mission_plotter),
            str(mission_history),
            "--paper-quality",
            "--save",
            "--format",
            args.format,
            "--no-estimator-comparison",
            "--paper" if args.mission_preset == "paper" else "--all",
        ]
        if args.show_legends:
            mission_command.append("--show-legends")

        _run(mission_command, "Plot mission")
        plotted.append("mission")

        if not args.no_estimator_comparison:
            _run(
                [
                    sys.executable,
                    str(estimator_plotter),
                    str(mission_history),
                    "--save",
                    "--format",
                    args.format,
                ],
                "Compare state estimators (mission)",
                required=False,
            )

    print("\nExperiment plotting complete.")
    for phase in plotted:
        print(
            f"{phase.capitalize():14s}: "
            f"{(run_dir / phase / 'plots').resolve()}"
        )


if __name__ == "__main__":
    main()
