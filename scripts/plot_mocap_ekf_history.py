#!/usr/bin/env python3
"""Plot one exported standalone MoCap-EKF history.

The EKF is the only required state stream. Transformed raw MoCap and PX4 are
overlaid independently when available, so this plotter remains useful even
when PX4 odometry was not published during the standalone test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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


def _has(values: np.ndarray | None, minimum: int = 2) -> bool:
    return (
        values is not None
        and np.count_nonzero(np.isfinite(values)) >= minimum
    )


def _normalize_quaternions(q: np.ndarray) -> np.ndarray:
    result = np.asarray(q, dtype=float).copy()
    norms = np.linalg.norm(result, axis=1)
    valid = np.isfinite(norms) & (norms > 1e-12)
    result[valid] /= norms[valid, None]
    result[~valid] = np.nan
    return result


def _attitude_error_deg(
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    q1 = _normalize_quaternions(first)
    q2 = _normalize_quaternions(second)
    dot = np.sum(q1 * q2, axis=1)
    output = np.full(dot.shape, np.nan)
    valid = np.isfinite(dot)
    output[valid] = np.rad2deg(
        2.0
        * np.arccos(
            np.clip(np.abs(dot[valid]), 0.0, 1.0)
        )
    )
    return output


def _norm_error(
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    output = np.full(first.shape[0], np.nan)
    valid = (
        np.all(np.isfinite(first), axis=1)
        & np.all(np.isfinite(second), axis=1)
    )
    output[valid] = np.linalg.norm(
        first[valid] - second[valid],
        axis=1,
    )
    return output


def _overlay_components(
    axis,
    times,
    *,
    ekf,
    mocap=None,
    px4=None,
    ylabel,
):
    labels = ("x", "y", "z")
    for component, label in enumerate(labels):
        valid = np.isfinite(times) & np.isfinite(ekf[:, component])
        if np.count_nonzero(valid) >= 2:
            line = axis.plot(
                times[valid],
                ekf[valid, component],
                label=f"EKF {label}",
            )[0]
            color = line.get_color()
        else:
            color = None

        if mocap is not None:
            valid = (
                np.isfinite(times)
                & np.isfinite(mocap[:, component])
            )
            if np.count_nonzero(valid) >= 2:
                kwargs = {"linestyle": ":"}
                if color is not None:
                    kwargs["color"] = color
                axis.plot(
                    times[valid],
                    mocap[valid, component],
                    label=f"raw MoCap {label}",
                    **kwargs,
                )

        if px4 is not None:
            valid = (
                np.isfinite(times)
                & np.isfinite(px4[:, component])
            )
            if np.count_nonzero(valid) >= 2:
                kwargs = {"linestyle": "--"}
                if color is not None:
                    kwargs["color"] = color
                axis.plot(
                    times[valid],
                    px4[valid, component],
                    label=f"PX4 {label}",
                    **kwargs,
                )

    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)


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

    required = (
        "times",
        "ekf_positions",
        "ekf_quaternions",
        "ekf_linear_velocity_body",
        "ekf_angular_velocity_body",
    )
    missing = [key for key in required if key not in arrays]
    if missing:
        raise SystemExit(
            "Exported history is missing EKF arrays: "
            + ", ".join(missing)
        )

    times = np.asarray(arrays["times"], dtype=float)
    ekf_pos = np.asarray(arrays["ekf_positions"], dtype=float)
    ekf_q = np.asarray(arrays["ekf_quaternions"], dtype=float)
    ekf_v = np.asarray(
        arrays["ekf_linear_velocity_body"],
        dtype=float,
    )
    ekf_w = np.asarray(
        arrays["ekf_angular_velocity_body"],
        dtype=float,
    )

    px4_pos = arrays.get("px4_positions")
    px4_q = arrays.get("px4_quaternions")
    px4_v = arrays.get("px4_linear_velocity_body")
    px4_w = arrays.get("px4_angular_velocity_body")

    mocap_pos = arrays.get("mocap_positions")
    mocap_q = arrays.get("mocap_quaternions")
    mocap_v = arrays.get("mocap_linear_velocity_body_fd")
    mocap_w = arrays.get("mocap_angular_velocity_body_fd")

    robots = [
        str(value) for value in metadata.get("robots", [])
    ]
    if not robots:
        robots = [
            f"robot_{index + 1}"
            for index in range(ekf_pos.shape[1])
        ]

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (history_path.parent / "plots").resolve()
    )

    figures = []
    for agent, robot in enumerate(robots):
        if not _has(ekf_pos[:, agent]):
            print(f"Skipping {robot}: no EKF position history.")
            continue

        p_px4 = (
            np.asarray(px4_pos[:, agent], dtype=float)
            if px4_pos is not None and _has(px4_pos[:, agent])
            else None
        )
        q_px4 = (
            np.asarray(px4_q[:, agent], dtype=float)
            if px4_q is not None and _has(px4_q[:, agent])
            else None
        )
        v_px4 = (
            np.asarray(px4_v[:, agent], dtype=float)
            if px4_v is not None and _has(px4_v[:, agent])
            else None
        )
        w_px4 = (
            np.asarray(px4_w[:, agent], dtype=float)
            if px4_w is not None and _has(px4_w[:, agent])
            else None
        )

        p_mocap = (
            np.asarray(mocap_pos[:, agent], dtype=float)
            if mocap_pos is not None and _has(mocap_pos[:, agent])
            else None
        )
        q_mocap = (
            np.asarray(mocap_q[:, agent], dtype=float)
            if mocap_q is not None and _has(mocap_q[:, agent])
            else None
        )
        v_mocap = (
            np.asarray(mocap_v[:, agent], dtype=float)
            if mocap_v is not None and _has(mocap_v[:, agent])
            else None
        )
        w_mocap = (
            np.asarray(mocap_w[:, agent], dtype=float)
            if mocap_w is not None and _has(mocap_w[:, agent])
            else None
        )

        fig, axes = plt.subplots(
            4,
            1,
            sharex=True,
            figsize=(11, 11),
        )

        _overlay_components(
            axes[0],
            times,
            ekf=ekf_pos[:, agent],
            mocap=p_mocap,
            px4=p_px4,
            ylabel="position [m]",
        )
        _overlay_components(
            axes[1],
            times,
            ekf=ekf_v[:, agent],
            mocap=v_mocap,
            px4=v_px4,
            ylabel="body linear velocity [m/s]",
        )

        if q_mocap is not None:
            axes[2].plot(
                times,
                _attitude_error_deg(
                    ekf_q[:, agent],
                    q_mocap,
                ),
                label="EKF - raw MoCap",
            )
        if q_px4 is not None:
            axes[2].plot(
                times,
                _attitude_error_deg(
                    ekf_q[:, agent],
                    q_px4,
                ),
                label="EKF - PX4",
            )
        if q_mocap is None and q_px4 is None:
            axes[2].text(
                0.5,
                0.5,
                "No attitude comparison stream available",
                transform=axes[2].transAxes,
                ha="center",
                va="center",
            )
        axes[2].set_ylabel("attitude error [deg]")
        axes[2].grid(True, alpha=0.3)
        if axes[2].get_legend_handles_labels()[0]:
            axes[2].legend(fontsize="small")

        _overlay_components(
            axes[3],
            times,
            ekf=ekf_w[:, agent],
            mocap=w_mocap,
            px4=w_px4,
            ylabel="body angular velocity [rad/s]",
        )
        axes[3].set_xlabel("t [s]")

        axes[0].set_title(
            f"Standalone MoCap-EKF state check: {robot}"
        )
        for index in (0, 1, 3):
            if axes[index].get_legend_handles_labels()[0]:
                axes[index].legend(
                    ncol=3,
                    fontsize="x-small",
                )

        fig.tight_layout()
        figures.append(fig)

        # Dedicated residual figure when references are available.
        fig_err, err_axes = plt.subplots(
            4,
            1,
            sharex=True,
            figsize=(10, 9),
        )

        references = []
        if p_mocap is not None:
            references.append(
                ("raw MoCap", p_mocap, q_mocap, v_mocap, w_mocap)
            )
        if p_px4 is not None:
            references.append(
                ("PX4", p_px4, q_px4, v_px4, w_px4)
            )

        for label, pos_ref, q_ref, v_ref, w_ref in references:
            err_axes[0].plot(
                times,
                _norm_error(ekf_pos[:, agent], pos_ref),
                label=f"EKF - {label}",
            )
            if q_ref is not None:
                err_axes[1].plot(
                    times,
                    _attitude_error_deg(
                        ekf_q[:, agent],
                        q_ref,
                    ),
                    label=f"EKF - {label}",
                )
            if v_ref is not None:
                err_axes[2].plot(
                    times,
                    _norm_error(ekf_v[:, agent], v_ref),
                    label=f"EKF - {label}",
                )
            if w_ref is not None:
                err_axes[3].plot(
                    times,
                    _norm_error(ekf_w[:, agent], w_ref),
                    label=f"EKF - {label}",
                )

        labels = (
            "position error [m]",
            "attitude error [deg]",
            "linear-velocity error [m/s]",
            "angular-velocity error [rad/s]",
        )
        for axis, ylabel in zip(err_axes, labels):
            axis.set_ylabel(ylabel)
            axis.grid(True, alpha=0.3)
            if axis.get_legend_handles_labels()[0]:
                axis.legend(fontsize="small")
        err_axes[-1].set_xlabel("t [s]")
        err_axes[0].set_title(
            f"Standalone MoCap-EKF disagreement: {robot}"
        )
        fig_err.tight_layout()
        figures.append(fig_err)

        if args.save:
            output_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                output_dir
                / f"standalone_ekf_state_{robot}.{args.format}",
                bbox_inches="tight",
            )
            fig_err.savefig(
                output_dir
                / f"standalone_ekf_error_{robot}.{args.format}",
                bbox_inches="tight",
            )

    if args.save:
        print(f"Standalone EKF plots: {output_dir}")
    if args.show:
        plt.show()
    else:
        for figure in figures:
            plt.close(figure)


if __name__ == "__main__":
    main()
