#!/usr/bin/env python3
"""Plot initialization and mission data for one experiment run.

Typical usage:

    uv run python scripts/plot_experiment.py outputs/experiments/<run>

By default every exported phase history that exists is plotted and saved in
its own phase-specific ``plots/`` directory.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(command: list[str], label: str) -> None:
    print(f"\n=== {label} ===", flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--phase",
        choices=("all", "initialization", "mission"),
        default="all",
        help="phase(s) to plot; default: all exported phases",
    )
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
        help="mission figure format; default: pdf",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run directory does not exist: {run_dir}")

    scripts_dir = Path(__file__).resolve().parent
    initialization_plotter = scripts_dir / "plot_initialization_experiment.py"
    mission_plotter = scripts_dir / "plot_formation_experiment.py"

    requested = (
        ("initialization", "mission")
        if args.phase == "all"
        else (args.phase,)
    )

    plotted: list[str] = []
    skipped: list[str] = []

    for phase in requested:
        if phase == "initialization":
            history = run_dir / "initialization" / "initialization_history.npz"
            if not history.exists():
                skipped.append(phase)
                print(
                    f"Skipping initialization: history does not exist at {history}",
                    flush=True,
                )
                continue
            command = [
                sys.executable,
                str(initialization_plotter),
                str(history),
                "--save",
            ]
        else:
            history = run_dir / "mission" / "formation_history.npz"
            if not history.exists():
                skipped.append(phase)
                print(
                    f"Skipping mission: history does not exist at {history}",
                    flush=True,
                )
                continue

            command = [
                sys.executable,
                str(mission_plotter),
                str(history),
                "--paper-quality",
                "--save",
                "--format",
                args.format,
            ]
            command.append(
                "--paper" if args.mission_preset == "paper" else "--all"
            )
            if args.show_legends:
                command.append("--show-legends")

        _run(command, f"Plot {phase}")
        plotted.append(phase)

    if not plotted:
        raise SystemExit(
            "No phase was plotted. Export the run first with "
            "scripts/export_experiment.py."
        )

    print("\nExperiment plotting complete.")
    print(f"Plotted phases: {', '.join(plotted)}")
    if skipped:
        print(f"Unavailable histories: {', '.join(skipped)}")
    print(f"Absolute run folder: {run_dir}")
    for phase in plotted:
        output_dir = (run_dir / phase / "plots").resolve()
        print(f"Absolute {phase} plot folder: {output_dir}")


if __name__ == "__main__":
    main()
