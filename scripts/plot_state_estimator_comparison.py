#!/usr/bin/env python3
"""Compare PX4, MoCap estimator, and transformed raw MoCap state histories.

Required parallel streams:
    PX4 VehicleOdometry
    /mocap/<robot>/odom_ekf

Optional third stream:
    /mocap/<robot>/pose_core

When the transformed raw MoCap pose is available, exporters also provide
finite-difference body linear/angular velocities. These are shown explicitly as
"raw MoCap FD" and are diagnostic only.

Typical usage:

    uv run python scripts/plot_state_estimator_comparison.py \
        RUN/mission/formation_history.npz --save
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


_REQUIRED = (
    "times",
    "px4_positions",
    "px4_quaternions",
    "px4_linear_velocity_body",
    "px4_angular_velocity_body",
    "ekf_positions",
    "ekf_quaternions",
    "ekf_linear_velocity_body",
    "ekf_angular_velocity_body",
)

_MOCAP_OPTIONAL = (
    "mocap_positions",
    "mocap_quaternions",
    "mocap_linear_velocity_body_fd",
    "mocap_angular_velocity_body_fd",
)


def _load(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path, allow_pickle=False) as data:
        arrays = {
            key: np.asarray(data[key])
            for key in data.files
            if key != "metadata_json"
        }
        metadata = {}
        if "metadata_json" in data.files:
            metadata = json.loads(
                str(data["metadata_json"].item())
            )
    return arrays, metadata


def _normalized_quaternion_rows(values: np.ndarray) -> np.ndarray:
    q = np.asarray(values, dtype=float).copy()
    norms = np.linalg.norm(q, axis=1)
    valid = np.isfinite(norms) & (norms > 1e-12)
    q[valid] /= norms[valid, None]
    q[~valid] = np.nan
    return q


def _attitude_disagreement_rad(
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    q1 = _normalized_quaternion_rows(first)
    q2 = _normalized_quaternion_rows(second)
    dot = np.sum(q1 * q2, axis=1)
    valid = np.isfinite(dot)
    result = np.full(dot.shape, np.nan)
    result[valid] = 2.0 * np.arccos(
        np.clip(np.abs(dot[valid]), 0.0, 1.0)
    )
    return result


def _paired_valid(
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.ndim == 1:
        return np.isfinite(first) & np.isfinite(second)
    return (
        np.all(np.isfinite(first), axis=1)
        & np.all(np.isfinite(second), axis=1)
    )


def _has_series(values: np.ndarray | None) -> bool:
    return (
        values is not None
        and np.count_nonzero(np.isfinite(values)) >= 2
    )


def _component_overlay(
    axis: plt.Axes,
    times: np.ndarray,
    *,
    px4: np.ndarray,
    ekf: np.ndarray,
    mocap: np.ndarray | None,
    ylabel: str,
    mocap_label: str,
) -> None:
    component_labels = ("x", "y", "z")
    for component, label in enumerate(component_labels):
        px4_valid = (
            np.isfinite(times)
            & np.isfinite(px4[:, component])
        )
        if np.count_nonzero(px4_valid) >= 2:
            line = axis.plot(
                times[px4_valid],
                px4[px4_valid, component],
                label=f"PX4 {label}",
            )[0]
            component_color = line.get_color()
        else:
            component_color = None

        ekf_valid = (
            np.isfinite(times)
            & np.isfinite(ekf[:, component])
        )
        if np.count_nonzero(ekf_valid) >= 2:
            kwargs = {"linestyle": "--"}
            if component_color is not None:
                kwargs["color"] = component_color
            axis.plot(
                times[ekf_valid],
                ekf[ekf_valid, component],
                label=f"MoCap estimator {label}",
                **kwargs,
            )

        if mocap is not None:
            mocap_valid = (
                np.isfinite(times)
                & np.isfinite(mocap[:, component])
            )
            if np.count_nonzero(mocap_valid) >= 2:
                kwargs = {"linestyle": ":"}
                if component_color is not None:
                    kwargs["color"] = component_color
                axis.plot(
                    times[mocap_valid],
                    mocap[mocap_valid, component],
                    label=f"{mocap_label} {label}",
                    **kwargs,
                )

    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)


def _plot_pairwise_metric(
    axis: plt.Axes,
    times: np.ndarray,
    *,
    px4_ekf: np.ndarray,
    px4_mocap: np.ndarray | None,
    ekf_mocap: np.ndarray | None,
    ylabel: str,
) -> None:
    pairs = [
        ("PX4 - estimator", px4_ekf),
        ("PX4 - raw MoCap", px4_mocap),
        ("estimator - raw MoCap", ekf_mocap),
    ]
    for label, values in pairs:
        if values is None:
            continue
        valid = np.isfinite(times) & np.isfinite(values)
        if np.count_nonzero(valid) >= 2:
            axis.plot(times[valid], values[valid], label=label)
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    handles, _ = axis.get_legend_handles_labels()
    if handles:
        axis.legend(fontsize="small")


def _norm_difference(
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    result = np.full(first.shape[0], np.nan)
    valid = _paired_valid(first, second)
    if np.any(valid):
        result[valid] = np.linalg.norm(
            first[valid] - second[valid],
            axis=1,
        )
    return result


def _state_figure(
    *,
    times: np.ndarray,
    robot: str,
    px4_position: np.ndarray,
    ekf_position: np.ndarray,
    mocap_position: np.ndarray | None,
    px4_quaternion: np.ndarray,
    ekf_quaternion: np.ndarray,
    mocap_quaternion: np.ndarray | None,
    px4_linear_velocity: np.ndarray,
    ekf_linear_velocity: np.ndarray,
    mocap_linear_velocity: np.ndarray | None,
    px4_angular_velocity: np.ndarray,
    ekf_angular_velocity: np.ndarray,
    mocap_angular_velocity: np.ndarray | None,
) -> plt.Figure:
    figure, axes = plt.subplots(
        4,
        1,
        sharex=True,
        figsize=(11, 11),
    )

    _component_overlay(
        axes[0],
        times,
        px4=px4_position,
        ekf=ekf_position,
        mocap=mocap_position,
        ylabel="position [m]",
        mocap_label="raw MoCap",
    )

    _component_overlay(
        axes[1],
        times,
        px4=px4_linear_velocity,
        ekf=ekf_linear_velocity,
        mocap=mocap_linear_velocity,
        ylabel="body linear velocity [m/s]",
        mocap_label="raw MoCap FD",
    )

    px4_ekf_attitude = np.rad2deg(
        _attitude_disagreement_rad(
            px4_quaternion,
            ekf_quaternion,
        )
    )
    px4_mocap_attitude = None
    ekf_mocap_attitude = None
    if mocap_quaternion is not None:
        px4_mocap_attitude = np.rad2deg(
            _attitude_disagreement_rad(
                px4_quaternion,
                mocap_quaternion,
            )
        )
        ekf_mocap_attitude = np.rad2deg(
            _attitude_disagreement_rad(
                ekf_quaternion,
                mocap_quaternion,
            )
        )

    _plot_pairwise_metric(
        axes[2],
        times,
        px4_ekf=px4_ekf_attitude,
        px4_mocap=px4_mocap_attitude,
        ekf_mocap=ekf_mocap_attitude,
        ylabel="attitude difference [deg]",
    )

    _component_overlay(
        axes[3],
        times,
        px4=px4_angular_velocity,
        ekf=ekf_angular_velocity,
        mocap=mocap_angular_velocity,
        ylabel="body angular velocity [rad/s]",
        mocap_label="raw MoCap FD",
    )
    axes[3].set_xlabel("t [s]")

    axes[0].set_title(
        f"PX4 / MoCap estimator / raw MoCap: {robot}"
    )
    for index in (0, 1, 3):
        handles, _ = axes[index].get_legend_handles_labels()
        if handles:
            axes[index].legend(
                ncol=3,
                fontsize="x-small",
            )

    figure.tight_layout()
    return figure


def _disagreement_figure(
    *,
    times: np.ndarray,
    robot: str,
    px4_position: np.ndarray,
    ekf_position: np.ndarray,
    mocap_position: np.ndarray | None,
    px4_quaternion: np.ndarray,
    ekf_quaternion: np.ndarray,
    mocap_quaternion: np.ndarray | None,
    px4_linear_velocity: np.ndarray,
    ekf_linear_velocity: np.ndarray,
    mocap_linear_velocity: np.ndarray | None,
    px4_angular_velocity: np.ndarray,
    ekf_angular_velocity: np.ndarray,
    mocap_angular_velocity: np.ndarray | None,
) -> plt.Figure:
    figure, axes = plt.subplots(
        4,
        1,
        sharex=True,
        figsize=(10, 9),
    )

    def optional_norm(
        first: np.ndarray,
        second: np.ndarray | None,
    ) -> np.ndarray | None:
        if second is None:
            return None
        return _norm_difference(first, second)

    _plot_pairwise_metric(
        axes[0],
        times,
        px4_ekf=_norm_difference(px4_position, ekf_position),
        px4_mocap=optional_norm(px4_position, mocap_position),
        ekf_mocap=(
            optional_norm(ekf_position, mocap_position)
        ),
        ylabel="position disagreement [m]",
    )

    px4_ekf_attitude = np.rad2deg(
        _attitude_disagreement_rad(
            px4_quaternion,
            ekf_quaternion,
        )
    )
    px4_mocap_attitude = None
    ekf_mocap_attitude = None
    if mocap_quaternion is not None:
        px4_mocap_attitude = np.rad2deg(
            _attitude_disagreement_rad(
                px4_quaternion,
                mocap_quaternion,
            )
        )
        ekf_mocap_attitude = np.rad2deg(
            _attitude_disagreement_rad(
                ekf_quaternion,
                mocap_quaternion,
            )
        )
    _plot_pairwise_metric(
        axes[1],
        times,
        px4_ekf=px4_ekf_attitude,
        px4_mocap=px4_mocap_attitude,
        ekf_mocap=ekf_mocap_attitude,
        ylabel="attitude disagreement [deg]",
    )

    _plot_pairwise_metric(
        axes[2],
        times,
        px4_ekf=_norm_difference(
            px4_linear_velocity,
            ekf_linear_velocity,
        ),
        px4_mocap=optional_norm(
            px4_linear_velocity,
            mocap_linear_velocity,
        ),
        ekf_mocap=optional_norm(
            ekf_linear_velocity,
            mocap_linear_velocity,
        ),
        ylabel="linear-velocity disagreement [m/s]",
    )

    _plot_pairwise_metric(
        axes[3],
        times,
        px4_ekf=_norm_difference(
            px4_angular_velocity,
            ekf_angular_velocity,
        ),
        px4_mocap=optional_norm(
            px4_angular_velocity,
            mocap_angular_velocity,
        ),
        ekf_mocap=optional_norm(
            ekf_angular_velocity,
            mocap_angular_velocity,
        ),
        ylabel="angular-velocity disagreement [rad/s]",
    )

    axes[0].set_title(f"State-source disagreement: {robot}")
    axes[-1].set_xlabel("t [s]")
    figure.tight_layout()
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--format",
        choices=("pdf", "png", "svg"),
        default="pdf",
    )
    args = parser.parse_args()

    history_path = args.history.expanduser().resolve()
    arrays, metadata = _load(history_path)

    missing = [name for name in _REQUIRED if name not in arrays]
    if missing:
        print(
            "Skipping estimator comparison: exported history is "
            "missing: " + ", ".join(missing)
        )
        return

    has_mocap = all(
        name in arrays for name in _MOCAP_OPTIONAL
    )
    if not has_mocap:
        print(
            "Raw transformed MoCap is unavailable in this history; "
            "plotting PX4 vs MoCap estimator only."
        )

    robots = [
        str(value) for value in metadata.get("robots", [])
    ]
    if not robots:
        n_agents = int(
            np.asarray(arrays["px4_positions"]).shape[1]
        )
        robots = [
            f"robot_{index + 1}"
            for index in range(n_agents)
        ]

    times = np.asarray(arrays["times"], dtype=float)
    px4_positions = np.asarray(
        arrays["px4_positions"], dtype=float
    )
    px4_quaternions = np.asarray(
        arrays["px4_quaternions"], dtype=float
    )
    px4_linear = np.asarray(
        arrays["px4_linear_velocity_body"],
        dtype=float,
    )
    px4_angular = np.asarray(
        arrays["px4_angular_velocity_body"],
        dtype=float,
    )

    ekf_positions = np.asarray(
        arrays["ekf_positions"], dtype=float
    )
    ekf_quaternions = np.asarray(
        arrays["ekf_quaternions"], dtype=float
    )
    ekf_linear = np.asarray(
        arrays["ekf_linear_velocity_body"],
        dtype=float,
    )
    ekf_angular = np.asarray(
        arrays["ekf_angular_velocity_body"],
        dtype=float,
    )

    mocap_positions = (
        np.asarray(arrays["mocap_positions"], dtype=float)
        if has_mocap
        else None
    )
    mocap_quaternions = (
        np.asarray(arrays["mocap_quaternions"], dtype=float)
        if has_mocap
        else None
    )
    mocap_linear = (
        np.asarray(
            arrays["mocap_linear_velocity_body_fd"],
            dtype=float,
        )
        if has_mocap
        else None
    )
    mocap_angular = (
        np.asarray(
            arrays["mocap_angular_velocity_body_fd"],
            dtype=float,
        )
        if has_mocap
        else None
    )

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (history_path.parent / "plots").resolve()
    )

    generated = 0
    figures: list[plt.Figure] = []

    for agent, robot in enumerate(robots):
        valid_position = _paired_valid(
            px4_positions[:, agent],
            ekf_positions[:, agent],
        )
        valid_velocity = _paired_valid(
            px4_linear[:, agent],
            ekf_linear[:, agent],
        )
        if (
            np.count_nonzero(valid_position) < 2
            and np.count_nonzero(valid_velocity) < 2
        ):
            print(
                f"Skipping estimator comparison for {robot}: "
                "PX4 and estimator do not overlap."
            )
            continue

        mocap_position_agent = (
            mocap_positions[:, agent]
            if mocap_positions is not None
            and _has_series(mocap_positions[:, agent])
            else None
        )
        mocap_quaternion_agent = (
            mocap_quaternions[:, agent]
            if mocap_quaternions is not None
            and _has_series(mocap_quaternions[:, agent])
            else None
        )
        mocap_linear_agent = (
            mocap_linear[:, agent]
            if mocap_linear is not None
            and _has_series(mocap_linear[:, agent])
            else None
        )
        mocap_angular_agent = (
            mocap_angular[:, agent]
            if mocap_angular is not None
            and _has_series(mocap_angular[:, agent])
            else None
        )

        state_figure = _state_figure(
            times=times,
            robot=robot,
            px4_position=px4_positions[:, agent],
            ekf_position=ekf_positions[:, agent],
            mocap_position=mocap_position_agent,
            px4_quaternion=px4_quaternions[:, agent],
            ekf_quaternion=ekf_quaternions[:, agent],
            mocap_quaternion=mocap_quaternion_agent,
            px4_linear_velocity=px4_linear[:, agent],
            ekf_linear_velocity=ekf_linear[:, agent],
            mocap_linear_velocity=mocap_linear_agent,
            px4_angular_velocity=px4_angular[:, agent],
            ekf_angular_velocity=ekf_angular[:, agent],
            mocap_angular_velocity=mocap_angular_agent,
        )
        disagreement_figure = _disagreement_figure(
            times=times,
            robot=robot,
            px4_position=px4_positions[:, agent],
            ekf_position=ekf_positions[:, agent],
            mocap_position=mocap_position_agent,
            px4_quaternion=px4_quaternions[:, agent],
            ekf_quaternion=ekf_quaternions[:, agent],
            mocap_quaternion=mocap_quaternion_agent,
            px4_linear_velocity=px4_linear[:, agent],
            ekf_linear_velocity=ekf_linear[:, agent],
            mocap_linear_velocity=mocap_linear_agent,
            px4_angular_velocity=px4_angular[:, agent],
            ekf_angular_velocity=ekf_angular[:, agent],
            mocap_angular_velocity=mocap_angular_agent,
        )
        figures.extend([state_figure, disagreement_figure])

        if args.save:
            output_dir.mkdir(parents=True, exist_ok=True)
            state_path = output_dir / (
                f"estimator_state_comparison_{robot}.{args.format}"
            )
            disagreement_path = output_dir / (
                f"estimator_disagreement_{robot}.{args.format}"
            )
            state_figure.savefig(
                state_path,
                bbox_inches="tight",
            )
            disagreement_figure.savefig(
                disagreement_path,
                bbox_inches="tight",
            )
            print(f"Saved {state_path}")
            print(f"Saved {disagreement_path}")

        generated += 2

    print(
        "Controller state source recorded in history: "
        f"{metadata.get('state_source', 'px4')}"
    )
    print(
        "Raw MoCap comparison: "
        + ("available" if has_mocap else "not available")
    )

    if generated == 0:
        print("No estimator-comparison figures were generated.")
        return

    if args.show:
        plt.show()
    else:
        for figure in figures:
            plt.close(figure)


if __name__ == "__main__":
    main()
