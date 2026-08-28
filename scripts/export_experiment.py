#!/usr/bin/env python3
"""Export initialization and mission data for one experiment run.

Typical usage:

    python scripts/export_experiment.py outputs/experiments/<run>

By default every phase whose rosbag exists is exported. This is intentionally
compatible with failed initialization runs, where the mission bag may not
exist.
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
        help="phase(s) to export; default: all available phases",
    )
    parser.add_argument(
        "--reference-robot",
        default=None,
        help="optional reference robot passed to the phase exporters",
    )
    parser.add_argument(
        "--max-sync-ms",
        type=float,
        default=None,
        help="optional synchronization tolerance passed to both exporters",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run directory does not exist: {run_dir}")

    scripts_dir = Path(__file__).resolve().parent
    initialization_exporter = scripts_dir / "export_initialization_bag.py"
    mission_exporter = scripts_dir / "export_formation_bag.py"

    requested = (
        ("initialization", "mission")
        if args.phase == "all"
        else (args.phase,)
    )

    exported: list[str] = []
    skipped: list[str] = []

    for phase in requested:
        bag_dir = run_dir / phase / "bag"
        if not bag_dir.exists():
            skipped.append(phase)
            print(
                f"Skipping {phase}: bag does not exist at {bag_dir}",
                flush=True,
            )
            continue

        if phase == "initialization":
            command = [
                sys.executable,
                str(initialization_exporter),
                str(run_dir),
            ]
        else:
            command = [
                sys.executable,
                str(mission_exporter),
                str(run_dir),
                "--phase",
                "mission",
            ]

        if args.reference_robot is not None:
            command.extend(["--reference-robot", args.reference_robot])
        if args.max_sync_ms is not None:
            command.extend(["--max-sync-ms", str(args.max_sync_ms)])

        _run(command, f"Export {phase}")
        exported.append(phase)

    if not exported:
        raise SystemExit(
            "No phase was exported. Expected initialization/bag and/or "
            "mission/bag inside the run directory."
        )

    print("\nExperiment export complete.")
    print(f"Exported phases: {', '.join(exported)}")
    if skipped:
        print(f"Unavailable phases: {', '.join(skipped)}")
    print(f"Absolute run folder: {run_dir}")
    for phase in exported:
        phase_dir = (run_dir / phase).resolve()
        print(f"Absolute {phase} output folder: {phase_dir}")


if __name__ == "__main__":
    main()
