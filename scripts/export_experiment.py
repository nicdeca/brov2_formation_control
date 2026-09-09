#!/usr/bin/env python3
"""Export every available phase of one formation-experiment run.

The maintained user-facing exporter uses a single command for the complete
run directory. It automatically discovers the split phases:

    <run>/initialization/bag
    <run>/mission/bag

Behavior:
- initialization only -> export initialization;
- mission only        -> export mission;
- both                -> export both;
- neither             -> fail clearly.

The phase-specific exporters remain available as lower-level debugging tools.
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
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run directory does not exist: {run_dir}")

    scripts_dir = Path(__file__).resolve().parent

    initialization_bag = run_dir / "initialization" / "bag"
    mission_bag = run_dir / "mission" / "bag"

    available = []
    if initialization_bag.exists():
        available.append("initialization")
    if mission_bag.exists():
        available.append("mission")

    if not available:
        raise FileNotFoundError(
            "No experiment phase bag was found. Expected at least one of:\n"
            f"  - {initialization_bag}\n"
            f"  - {mission_bag}"
        )

    exported = []

    if "initialization" in available:
        _run(
            [
                sys.executable,
                str(scripts_dir / "export_initialization_bag.py"),
                str(run_dir),
            ],
            "Export initialization",
        )
        history = (
            run_dir / "initialization" / "initialization_history.npz"
        )
        if not history.exists():
            raise RuntimeError(
                "Initialization exporter completed but did not create "
                f"{history}"
            )
        exported.append(("initialization", history))

    if "mission" in available:
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
        history = run_dir / "mission" / "formation_history.npz"
        if not history.exists():
            raise RuntimeError(
                "Mission exporter completed but did not create "
                f"{history}"
            )
        exported.append(("mission", history))

    print("\nExperiment export complete.")
    for phase, history in exported:
        print(f"{phase.capitalize():14s}: {history}")


if __name__ == "__main__":
    main()
