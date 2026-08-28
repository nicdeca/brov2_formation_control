#!/usr/bin/env python3
"""Plot diagnostics for the initialization phase of a formation experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import json
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class InitializationHistory:
    arrays: dict[str, np.ndarray]
    metadata: dict

    @classmethod
    def load(cls, path: Path) -> "InitializationHistory":
        with np.load(path, allow_pickle=False) as data:
            arrays = {
                key: np.asarray(data[key])
                for key in data.files
                if key != "metadata_json"
            }
            metadata = json.loads(str(data["metadata_json"].item()))
        return cls(arrays=arrays, metadata=metadata)


WORKSPACE_NAMES = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")


def _world_speed(history: InitializationHistory) -> np.ndarray:
    # The Euclidean norm is invariant under the body/world rotation, so the
    # logged body-frame linear velocity is sufficient for initialization speed.
    return np.linalg.norm(
        np.asarray(history.arrays["linear_velocity_body"], dtype=float),
        axis=2,
    )


def _save(
    figure: plt.Figure,
    path: Path,
    *,
    dpi: int = 180,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _reference_position(
    arrays: dict[str, np.ndarray],
    robot_index: int,
) -> np.ndarray | None:
    if "reference_position" not in arrays:
        return None
    reference = np.asarray(
        arrays["reference_position"][:, robot_index, :],
        dtype=float,
    )
    finite = np.all(np.isfinite(reference), axis=1)
    if not np.any(finite):
        return None

    # Initialization references are normally constant. Forward/back-fill any
    # sparse logged copies so the diagnostic remains legible.
    result = reference.copy()
    first = int(np.flatnonzero(finite)[0])
    result[:first] = result[first]
    for k in range(first + 1, result.shape[0]):
        if not np.all(np.isfinite(result[k])):
            result[k] = result[k - 1]
    return result


def plot_position_tracking(
    history: InitializationHistory,
    output_dir: Path,
) -> int:
    arrays = history.arrays
    times = np.asarray(arrays["times"], dtype=float)
    positions = np.asarray(arrays["positions"], dtype=float)
    robots = [str(robot) for robot in history.metadata["robots"]]
    labels = ("x", "y", "z")
    count = 0

    for agent, robot in enumerate(robots):
        reference = _reference_position(arrays, agent)

        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 7))
        for axis_index, axis in enumerate(axes):
            axis.plot(times, positions[:, agent, axis_index], label="actual")
            if reference is not None:
                axis.plot(
                    times,
                    reference[:, axis_index],
                    "--",
                    label="initialization reference",
                )
            axis.set_ylabel(f"{labels[axis_index]} [m]")
            axis.grid(True, alpha=0.3)
        axes[0].set_title(f"Initialization position: {robot}")
        if reference is not None:
            axes[0].legend()
        axes[-1].set_xlabel("t [s]")
        _save(fig, output_dir / f"position_{robot}.pdf")
        count += 1

    return count


def plot_error_and_speed(
    history: InitializationHistory,
    output_dir: Path,
    *,
    position_tolerance: float | None,
    speed_tolerance: float | None,
) -> int:
    arrays = history.arrays
    times = np.asarray(arrays["times"], dtype=float)
    positions = np.asarray(arrays["positions"], dtype=float)
    robots = [str(robot) for robot in history.metadata["robots"]]
    speed = _world_speed(history)

    fig, (ax_error, ax_speed) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    have_error = False

    for agent, robot in enumerate(robots):
        reference = _reference_position(arrays, agent)
        if reference is not None:
            error = np.linalg.norm(positions[:, agent, :] - reference, axis=1)
            ax_error.plot(times, error, label=robot)
            have_error = True
        ax_speed.plot(times, speed[:, agent], label=robot)

    if position_tolerance is not None:
        ax_error.axhline(
            float(position_tolerance),
            linestyle="--",
            label="phase-manager tolerance",
        )
    if speed_tolerance is not None:
        ax_speed.axhline(
            float(speed_tolerance),
            linestyle="--",
            label="phase-manager tolerance",
        )

    ax_error.set_ylabel("position error [m]")
    ax_error.set_title("Initialization convergence")
    ax_error.grid(True, alpha=0.3)
    if have_error:
        ax_error.legend()

    ax_speed.set_ylabel("speed [m/s]")
    ax_speed.set_xlabel("t [s]")
    ax_speed.grid(True, alpha=0.3)
    ax_speed.legend()

    _save(fig, output_dir / "convergence.pdf")
    return 1


def plot_workspace(
    history: InitializationHistory,
    output_dir: Path,
) -> int:
    arrays = history.arrays
    required = (
        "workspace_physical_constraint_values",
        "workspace_relaxation",
    )
    if not all(name in arrays for name in required):
        return 0

    times = np.asarray(arrays["times"], dtype=float)
    physical = np.asarray(
        arrays["workspace_physical_constraint_values"],
        dtype=float,
    )
    relaxation = np.asarray(arrays["workspace_relaxation"], dtype=float)
    robots = [str(robot) for robot in history.metadata["robots"]]
    count = 0

    for agent, robot in enumerate(robots):
        fig, (ax_margin, ax_relax) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(8, 6),
        )

        values = physical[:, agent, :]
        for channel, name in enumerate(WORKSPACE_NAMES):
            ax_margin.plot(times, values[:, channel], label=name)
        ax_margin.axhline(0.0, linestyle="--")
        ax_margin.set_ylabel("physical margin [m]")
        ax_margin.set_title(f"Initialization workspace: {robot}")
        ax_margin.grid(True, alpha=0.3)
        ax_margin.legend(ncol=3)

        s = relaxation[:, agent, :]
        for channel, name in enumerate(WORKSPACE_NAMES):
            ax_relax.plot(times, s[:, channel], label=name)
        ax_relax.set_ylabel("relaxation state")
        ax_relax.set_xlabel("t [s]")
        ax_relax.grid(True, alpha=0.3)

        _save(fig, output_dir / f"workspace_{robot}.pdf")
        count += 1

    return count


def plot_controller_health(
    history: InitializationHistory,
    output_dir: Path,
) -> int:
    arrays = history.arrays
    robots = [str(robot) for robot in history.metadata["robots"]]
    times = np.asarray(arrays["times"], dtype=float)

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(8, 9))
    names = (
        ("thruster_utilization", "thruster utilization"),
        ("required_slack", "required CLF slack"),
        ("px4_armed", "armed"),
        ("px4_offboard_enabled", "Offboard enabled"),
    )

    plotted = False
    for axis, (field, ylabel) in zip(axes, names):
        if field not in arrays:
            axis.set_visible(False)
            continue
        values = np.asarray(arrays[field], dtype=float)
        for agent, robot in enumerate(robots):
            axis.plot(times, values[:, agent], label=robot)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        plotted = True

    if not plotted:
        plt.close(fig)
        return 0

    axes[0].set_title("Initialization controller/PX4 status")
    visible_axes = [axis for axis in axes if axis.get_visible()]
    if visible_axes:
        visible_axes[0].legend()
        visible_axes[-1].set_xlabel("t [s]")

    _save(fig, output_dir / "controller_health.pdf")
    return 1


def print_summary(
    history: InitializationHistory,
    *,
    position_tolerance: float | None,
    speed_tolerance: float | None,
) -> None:
    arrays = history.arrays
    robots = [str(robot) for robot in history.metadata["robots"]]
    positions = np.asarray(arrays["positions"], dtype=float)
    speed = _world_speed(history)

    print("Initialization summary")
    print(f"Duration: {float(arrays['times'][-1]):.3f} s")

    for agent, robot in enumerate(robots):
        pieces = [robot]
        reference = _reference_position(arrays, agent)
        if reference is not None:
            error = np.linalg.norm(positions[:, agent, :] - reference, axis=1)
            pieces.append(f"final position error {error[-1]:.4f} m")
            pieces.append(f"minimum position error {np.nanmin(error):.4f} m")
            if position_tolerance is not None:
                pieces.append(
                    "position tolerance "
                    + ("met" if error[-1] <= position_tolerance else "NOT met")
                )

        pieces.append(f"final speed {speed[-1, agent]:.4f} m/s")
        if speed_tolerance is not None:
            pieces.append(
                "speed tolerance "
                + ("met" if speed[-1, agent] <= speed_tolerance else "NOT met")
            )

        if "workspace_physical_constraint_values" in arrays:
            values = np.asarray(
                arrays["workspace_physical_constraint_values"][:, agent, :],
                dtype=float,
            )
            if np.any(np.isfinite(values)):
                pieces.append(
                    f"min workspace physical margin {np.nanmin(values):.4f} m"
                )

        if "workspace_relaxation" in arrays:
            values = np.asarray(
                arrays["workspace_relaxation"][:, agent, :],
                dtype=float,
            )
            if np.any(np.isfinite(values)):
                pieces.append(f"max workspace relaxation {np.nanmax(values):.3f}")

        if "required_slack" in arrays:
            values = np.asarray(arrays["required_slack"][:, agent], dtype=float)
            if np.any(np.isfinite(values)):
                pieces.append(f"max required slack {np.nanmax(values):.4g}")

        print("  " + ", ".join(pieces))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--position-tolerance",
        type=float,
        default=0.65,
        help="phase-manager position tolerance used for the current canonical launches",
    )
    parser.add_argument(
        "--speed-tolerance",
        type=float,
        default=0.08,
        help="phase-manager speed tolerance used for the current canonical launches",
    )
    args = parser.parse_args()

    history_path = args.history.expanduser().resolve()
    history = InitializationHistory.load(history_path)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (history_path.parent / "plots").resolve()
    )

    print_summary(
        history,
        position_tolerance=args.position_tolerance,
        speed_tolerance=args.speed_tolerance,
    )

    saved = 0
    if args.save:
        output_dir.mkdir(parents=True, exist_ok=True)
        saved += plot_position_tracking(history, output_dir)
        saved += plot_error_and_speed(
            history,
            output_dir,
            position_tolerance=args.position_tolerance,
            speed_tolerance=args.speed_tolerance,
        )
        saved += plot_workspace(history, output_dir)
        saved += plot_controller_health(history, output_dir)
        print(f"Saved {saved} initialization figure(s).")
        print(f"Absolute output folder: {output_dir}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
