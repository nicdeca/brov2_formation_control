#!/usr/bin/env python3
"""Plot diagnostics for the initialization phase of a formation experiment."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

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


def _fill_piecewise_constant(values: np.ndarray) -> np.ndarray:
    """Forward/back-fill finite vector samples without inventing new values."""
    result = np.asarray(values, dtype=float).copy()
    finite = np.all(np.isfinite(result), axis=1)
    if not np.any(finite):
        return result

    first = int(np.flatnonzero(finite)[0])
    result[:first] = result[first]
    for k in range(first + 1, result.shape[0]):
        if not np.all(np.isfinite(result[k])):
            result[k] = result[k - 1]
    return result


def _initialization_reference_history(
    history: InitializationHistory,
) -> np.ndarray:
    """Return the absolute initialization reference logged for every robot.

    Initialization is an absolute-position phase for both the leader and the
    followers.  Do not reconstruct follower targets from formation vectors:
    the follower has its own explicit ``initialization_position`` parameter.
    """
    arrays = history.arrays
    robots = [str(robot) for robot in history.metadata["robots"]]
    n_samples = int(np.asarray(arrays["times"]).size)
    references = np.full((n_samples, len(robots), 3), np.nan)

    direct = arrays.get("reference_position")
    if direct is None:
        return references

    direct = np.asarray(direct, dtype=float)
    if direct.shape != references.shape:
        raise ValueError(
            "reference_position has unexpected shape: "
            f"{direct.shape}; expected {references.shape}"
        )

    for agent in range(len(robots)):
        references[:, agent, :] = _fill_piecewise_constant(
            direct[:, agent, :]
        )
    return references


def _finite_series(values: np.ndarray) -> bool:
    return bool(np.any(np.isfinite(np.asarray(values, dtype=float))))


def _workspace_bounds_from_margins(
    position: np.ndarray,
    margins: np.ndarray,
) -> np.ndarray:
    """Recover lower/upper coordinate bounds from six wall margins."""
    position = np.asarray(position, dtype=float)
    margins = np.asarray(margins, dtype=float)
    n = position.shape[0]
    bounds = np.full((n, 3, 2), np.nan)

    for axis in range(3):
        lower_channel = 2 * axis
        upper_channel = lower_channel + 1
        bounds[:, axis, 0] = position[:, axis] - margins[:, lower_channel]
        bounds[:, axis, 1] = position[:, axis] + margins[:, upper_channel]
    return bounds


def plot_position_tracking(
    history: InitializationHistory,
    output_dir: Path,
) -> int:
    arrays = history.arrays
    times = np.asarray(arrays["times"], dtype=float)
    positions = np.asarray(arrays["positions"], dtype=float)
    robots = [str(robot) for robot in history.metadata["robots"]]
    references = _initialization_reference_history(history)
    labels = ("x", "y", "z")
    count = 0

    for agent, robot in enumerate(robots):
        reference = references[:, agent, :]
        have_reference = np.any(np.all(np.isfinite(reference), axis=1))

        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 7))
        for axis_index, axis in enumerate(axes):
            valid_actual = (
                np.isfinite(times)
                & np.isfinite(positions[:, agent, axis_index])
            )
            axis.plot(
                times[valid_actual],
                positions[valid_actual, agent, axis_index],
                label="actual",
            )
            if have_reference:
                valid = np.isfinite(reference[:, axis_index]) & np.isfinite(times)
                axis.plot(
                    times[valid],
                    reference[valid, axis_index],
                    "--",
                    label="initialization reference",
                )
            axis.set_ylabel(f"{labels[axis_index]} [m]")
            axis.grid(True, alpha=0.3)
        axes[0].set_title(f"Initialization position: {robot}")
        if have_reference:
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
    references = _initialization_reference_history(history)
    speed = _world_speed(history)

    fig, (ax_error, ax_speed) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    have_error = False
    have_speed = False

    for agent, robot in enumerate(robots):
        reference = references[:, agent, :]
        valid_reference = np.all(np.isfinite(reference), axis=1)
        valid_position = np.all(np.isfinite(positions[:, agent, :]), axis=1)
        valid_error = valid_reference & valid_position & np.isfinite(times)
        if np.count_nonzero(valid_error) >= 2:
            error = np.linalg.norm(
                positions[valid_error, agent, :] - reference[valid_error],
                axis=1,
            )
            ax_error.plot(times[valid_error], error, label=robot)
            have_error = True

        valid_speed = np.isfinite(speed[:, agent]) & np.isfinite(times)
        if np.count_nonzero(valid_speed) >= 2:
            ax_speed.plot(
                times[valid_speed],
                speed[valid_speed, agent],
                label=robot,
            )
            have_speed = True

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
    if have_error or position_tolerance is not None:
        ax_error.legend()

    ax_speed.set_ylabel("speed [m/s]")
    ax_speed.set_xlabel("t [s]")
    ax_speed.grid(True, alpha=0.3)
    if have_speed or speed_tolerance is not None:
        ax_speed.legend()

    _save(fig, output_dir / "convergence.pdf")
    return 1


def plot_workspace(
    history: InitializationHistory,
    output_dir: Path,
) -> int:
    """Plot robot position directly against conservative/physical pool bounds.

    This is intentionally different from the previous six-margin/two-column
    diagnostic figure.  During normal initialization the workspace relaxation
    is usually exactly zero, so mixing it into the same figure produced large,
    mostly empty panels.  The primary workspace figure now answers the useful
    question directly: where was the vehicle relative to the pool bounds?
    """
    arrays = history.arrays
    if "workspace_physical_constraint_values" not in arrays:
        return 0

    times = np.asarray(arrays["times"], dtype=float)
    positions = np.asarray(arrays["positions"], dtype=float)
    physical = np.asarray(
        arrays["workspace_physical_constraint_values"],
        dtype=float,
    )
    conservative = np.asarray(
        arrays.get(
            "workspace_conservative_constraint_values",
            np.full_like(physical, np.nan),
        ),
        dtype=float,
    )
    robots = [str(robot) for robot in history.metadata["robots"]]
    references = _initialization_reference_history(history)
    axis_names = ("x", "y", "z")
    count = 0

    for agent, robot in enumerate(robots):
        p = positions[:, agent, :]
        hp = physical[:, agent, :]
        hc = conservative[:, agent, :]

        have_physical = _finite_series(hp)
        have_conservative = _finite_series(hc)
        if not (have_physical or have_conservative):
            print(
                f"Skipping workspace plot for {robot}: no finite workspace "
                "constraint diagnostics were exported."
            )
            continue

        physical_bounds = (
            _workspace_bounds_from_margins(p, hp)
            if have_physical
            else np.full((len(times), 3, 2), np.nan)
        )
        conservative_bounds = (
            _workspace_bounds_from_margins(p, hc)
            if have_conservative
            else np.full((len(times), 3, 2), np.nan)
        )
        reference = references[:, agent, :]

        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 7))
        for spatial_axis, axis_name in enumerate(axis_names):
            axis = axes[spatial_axis]
            valid_position = np.isfinite(times) & np.isfinite(p[:, spatial_axis])
            axis.plot(
                times[valid_position],
                p[valid_position, spatial_axis],
                label="actual" if spatial_axis == 0 else None,
            )

            ref_valid = np.isfinite(times) & np.isfinite(reference[:, spatial_axis])
            if np.any(ref_valid):
                axis.plot(
                    times[ref_valid],
                    reference[ref_valid, spatial_axis],
                    "--",
                    label="initialization reference" if spatial_axis == 0 else None,
                )

            if have_conservative:
                for side in (0, 1):
                    values = conservative_bounds[:, spatial_axis, side]
                    valid = np.isfinite(times) & np.isfinite(values)
                    if np.any(valid):
                        axis.plot(
                            times[valid],
                            values[valid],
                            ":",
                            label=(
                                "conservative bounds"
                                if spatial_axis == 0 and side == 0
                                else None
                            ),
                        )

            if have_physical:
                for side in (0, 1):
                    values = physical_bounds[:, spatial_axis, side]
                    valid = np.isfinite(times) & np.isfinite(values)
                    if np.any(valid):
                        axis.plot(
                            times[valid],
                            values[valid],
                            "-.",
                            label=(
                                "physical bounds"
                                if spatial_axis == 0 and side == 0
                                else None
                            ),
                        )

            axis.set_ylabel(f"{axis_name} [m]")
            axis.grid(True, alpha=0.3)

        axes[0].set_title(f"Initialization workspace: {robot}")
        if axes[0].get_legend_handles_labels()[0]:
            axes[0].legend(loc="best")
        axes[-1].set_xlabel("t [s]")
        _save(fig, output_dir / f"workspace_{robot}.pdf")
        count += 1

    return count


def plot_workspace_relaxation(
    history: InitializationHistory,
    output_dir: Path,
    *,
    zero_tolerance: float = 1e-10,
) -> int:
    """Plot workspace relaxation only when it actually becomes nonzero."""
    arrays = history.arrays
    if "workspace_relaxation" not in arrays:
        return 0

    times = np.asarray(arrays["times"], dtype=float)
    relaxation = np.asarray(arrays["workspace_relaxation"], dtype=float)
    robots = [str(robot) for robot in history.metadata["robots"]]
    count = 0

    for agent, robot in enumerate(robots):
        values = relaxation[:, agent, :]
        finite = np.isfinite(values)
        if not np.any(finite):
            continue
        if float(np.nanmax(np.abs(values))) <= zero_tolerance:
            print(
                f"Skipping workspace-relaxation plot for {robot}: "
                "relaxation remained zero during initialization."
            )
            continue

        fig, ax = plt.subplots(figsize=(8, 4.5))
        for channel, name in enumerate(WORKSPACE_NAMES):
            valid = np.isfinite(times) & np.isfinite(values[:, channel])
            if np.count_nonzero(valid) >= 2:
                ax.plot(times[valid], values[valid, channel], label=name)
        ax.set_title(f"Initialization workspace relaxation: {robot}")
        ax.set_ylabel("normalized relaxation")
        ax.set_xlabel("t [s]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        _save(fig, output_dir / f"workspace_relaxation_{robot}.pdf")
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
        field_plotted = False
        for agent, robot in enumerate(robots):
            valid = np.isfinite(times) & np.isfinite(values[:, agent])
            if np.count_nonzero(valid) >= 2:
                axis.plot(times[valid], values[valid, agent], label=robot)
                field_plotted = True
        if not field_plotted:
            axis.set_visible(False)
            continue
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        plotted = True

    if not plotted:
        plt.close(fig)
        return 0

    visible_axes = [axis for axis in axes if axis.get_visible()]
    visible_axes[0].set_title("Initialization controller/PX4 status")
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
    references = _initialization_reference_history(history)
    speed = _world_speed(history)

    print("Initialization summary")
    print(f"Duration: {float(arrays['times'][-1]):.3f} s")

    for agent, robot in enumerate(robots):
        pieces = [robot]
        reference = references[:, agent, :]
        valid_reference = np.all(np.isfinite(reference), axis=1)
        valid_position = np.all(np.isfinite(positions[:, agent, :]), axis=1)
        valid_error = valid_reference & valid_position
        if np.any(valid_error):
            error = np.linalg.norm(
                positions[valid_error, agent, :] - reference[valid_error],
                axis=1,
            )
            pieces.append(f"final position error {error[-1]:.4f} m")
            pieces.append(f"minimum position error {np.nanmin(error):.4f} m")
            if position_tolerance is not None:
                pieces.append(
                    "position tolerance "
                    + ("met" if error[-1] <= position_tolerance else "NOT met")
                )
        else:
            pieces.append("initialization reference unavailable")

        finite_speed = np.isfinite(speed[:, agent])
        if np.any(finite_speed):
            final_speed = speed[np.flatnonzero(finite_speed)[-1], agent]
            pieces.append(f"final speed {final_speed:.4f} m/s")
            if speed_tolerance is not None:
                pieces.append(
                    "speed tolerance "
                    + ("met" if final_speed <= speed_tolerance else "NOT met")
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
        saved += plot_workspace_relaxation(history, output_dir)
        saved += plot_controller_health(history, output_dir)
        print(f"Saved {saved} initialization figure(s).")
        print(f"Absolute output folder: {output_dir}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
