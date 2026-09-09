#!/usr/bin/env python3
"""Export every available phase of a split formation-experiment run."""

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
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run directory does not exist: {run_dir}")

    scripts_dir = Path(__file__).resolve().parent
    exported = []

    initialization_bag = run_dir / "initialization" / "bag"
    if initialization_bag.exists():
        _run(
            [
                sys.executable,
                str(scripts_dir / "export_initialization_bag.py"),
                str(run_dir),
            ],
            "Export initialization",
        )
        exported.append("initialization")
    else:
        print(
            f"Skipping initialization: bag does not exist at "
            f"{initialization_bag}",
            flush=True,
        )

    mission_bag = run_dir / "mission" / "bag"
    if mission_bag.exists():
        _run(
            [
                sys.executable,
                str(scripts_dir / "export_formation_bag.py"),
                str(run_dir),
                "--phase",
                "mission",
            ],
            "Export mission",
        )
        exported.append("mission")
    else:
        print(
            f"Skipping mission: bag does not exist at {mission_bag}",
            flush=True,
        )

    if not exported:
        raise SystemExit("No initialization or mission bag was found.")

    print("\nExperiment export complete.")
    print(f"Exported phases: {', '.join(exported)}")


if __name__ == "__main__":
    main()
