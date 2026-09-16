#!/usr/bin/env python3
"""Export and plot one standalone MoCap-EKF diagnostic run.

This is the user-facing plot command for runs created by
``record_mocap_ekf.py`` / ``record_mocap_ekf.sh``.

It:
1. exports the rosbag through the maintained initialization exporter;
2. plots EKF state against transformed raw MoCap and PX4 when available;
3. plots message timing for raw MoCap, EKF, PX4 and both gyro sources;
4. prints message counts and approximate rates for each stream.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import rosbag2_py
except ImportError as error:
    raise SystemExit(
        "ROS Python packages are unavailable. Source the ROS workspace first."
    ) from error


def _open_reader(bag_dir: Path):
    storage_options = rosbag2_py.StorageOptions(
        uri=str(bag_dir),
        storage_id="sqlite3",
    )
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader = rosbag2_py.SequentialReader()
    try:
        reader.open(storage_options, converter_options)
    except RuntimeError:
        storage_options = rosbag2_py.StorageOptions(
            uri=str(bag_dir),
            storage_id="mcap",
        )
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)
    return reader


def _topic_timestamps(bag_dir: Path) -> dict[str, np.ndarray]:
    reader = _open_reader(bag_dir)
    values: dict[str, list[int]] = {}
    while reader.has_next():
        topic, _, timestamp_ns = reader.read_next()
        values.setdefault(topic, []).append(int(timestamp_ns))
    return {
        topic: np.asarray(stamps, dtype=np.int64)
        for topic, stamps in values.items()
    }


def _relative_seconds(
    stamps: np.ndarray,
    origin_ns: int,
) -> np.ndarray:
    return (stamps.astype(np.float64) - float(origin_ns)) * 1e-9


def _rate_summary(
    stamps: np.ndarray | None,
) -> tuple[int, float | None, float | None]:
    if stamps is None or stamps.size == 0:
        return 0, None, None
    if stamps.size == 1:
        return 1, None, None

    dt = np.diff(stamps.astype(np.float64)) * 1e-9
    valid = dt[np.isfinite(dt) & (dt > 0.0)]
    if valid.size == 0:
        return int(stamps.size), None, None

    duration = (
        float(stamps[-1] - stamps[0]) * 1e-9
    )
    rate = (
        float(stamps.size - 1) / duration
        if duration > 0.0
        else None
    )
    max_gap = float(np.max(valid))
    return int(stamps.size), rate, max_gap


def _plot_timing(
    *,
    run_dir: Path,
    robots: list[str],
    topic_times: dict[str, np.ndarray],
    save: bool,
    show: bool,
    fmt: str,
) -> None:
    all_nonempty = [
        stamps
        for stamps in topic_times.values()
        if stamps.size > 0
    ]
    if not all_nonempty:
        print("No recorded topic timestamps found.")
        return

    origin_ns = min(int(stamps[0]) for stamps in all_nonempty)
    output_dir = run_dir / "initialization" / "plots"

    figures = []
    for robot in robots:
        topics = [
            ("raw MoCap", f"/mocap/{robot}/pose"),
            ("pose_core", f"/mocap/{robot}/pose_core"),
            ("EKF odom", f"/mocap/{robot}/odom_ekf"),
            ("PX4 odom", f"/{robot}/fmu/out/vehicle_odometry"),
            ("MAVROS gyro", f"/{robot}/mavros/imu/data"),
            ("sim gyro", f"/mocap/{robot}/imu"),
        ]

        print(f"\nStream summary: {robot}")
        for label, topic in topics:
            count, rate, max_gap = _rate_summary(
                topic_times.get(topic)
            )
            rate_text = (
                f"{rate:.2f} Hz" if rate is not None else "n/a"
            )
            gap_text = (
                f"{max_gap:.3f} s"
                if max_gap is not None
                else "n/a"
            )
            print(
                f"  {label:12s}: {count:7d} samples, "
                f"rate {rate_text:>10s}, max gap {gap_text}"
            )

        # Availability/event raster.
        fig, axis = plt.subplots(figsize=(11, 4.8))
        y_ticks = []
        y_labels = []
        for index, (label, topic) in enumerate(topics):
            stamps = topic_times.get(topic)
            if stamps is None or stamps.size == 0:
                continue
            times = _relative_seconds(stamps, origin_ns)
            axis.scatter(
                times,
                np.full(times.shape, float(index)),
                s=4,
                marker="|",
            )
            y_ticks.append(index)
            y_labels.append(label)

        axis.set_yticks(y_ticks, y_labels)
        axis.set_xlabel("recording time [s]")
        axis.set_title(f"Estimator stream availability: {robot}")
        axis.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        figures.append(fig)

        # Inter-message interval plot for the streams most useful for dropout
        # and gyro diagnosis.
        fig_dt, axis_dt = plt.subplots(figsize=(11, 4.8))
        for label, topic in (
            ("raw MoCap", f"/mocap/{robot}/pose"),
            ("EKF odom", f"/mocap/{robot}/odom_ekf"),
            ("MAVROS gyro", f"/{robot}/mavros/imu/data"),
            ("sim gyro", f"/mocap/{robot}/imu"),
        ):
            stamps = topic_times.get(topic)
            if stamps is None or stamps.size < 2:
                continue
            times = _relative_seconds(stamps[1:], origin_ns)
            dt = np.diff(stamps.astype(np.float64)) * 1e-9
            axis_dt.plot(times, dt, label=label)

        axis_dt.set_xlabel("recording time [s]")
        axis_dt.set_ylabel("inter-message interval [s]")
        axis_dt.set_title(f"Estimator stream timing: {robot}")
        axis_dt.grid(True, alpha=0.3)
        if axis_dt.get_legend_handles_labels()[0]:
            axis_dt.legend()
        fig_dt.tight_layout()
        figures.append(fig_dt)

        if save:
            output_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                output_dir
                / f"standalone_ekf_stream_availability_{robot}.{fmt}",
                bbox_inches="tight",
            )
            fig_dt.savefig(
                output_dir
                / f"standalone_ekf_stream_timing_{robot}.{fmt}",
                bbox_inches="tight",
            )

    if show:
        plt.show()
    else:
        for figure in figures:
            plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--format",
        choices=("pdf", "png", "svg"),
        default="pdf",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="reuse an existing exported history instead of re-exporting the bag",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    bag_dir = run_dir / "initialization" / "bag"
    history = (
        run_dir
        / "initialization"
        / "initialization_history.npz"
    )

    if not bag_dir.exists():
        raise FileNotFoundError(
            f"standalone EKF bag does not exist: {bag_dir}"
        )

    scripts_dir = Path(__file__).resolve().parent

    if not args.no_export:
        export_script = (
            scripts_dir / "export_initialization_bag.py"
        )
        command = [
            sys.executable,
            str(export_script),
            str(run_dir),
        ]
        print("\n=== Export standalone EKF bag ===", flush=True)
        subprocess.run(command, check=True)

    if not history.exists():
        raise FileNotFoundError(
            f"exported history does not exist: {history}"
        )

    state_plotter = (
        scripts_dir / "plot_mocap_ekf_history.py"
    )
    command = [
        sys.executable,
        str(state_plotter),
        str(history),
        "--format",
        args.format,
    ]
    if args.save:
        command.append("--save")
    if args.show:
        command.append("--show")

    print("\n=== Plot standalone EKF state ===", flush=True)
    subprocess.run(command, check=True)

    # Read robot names from the manifest.
    import yaml

    manifest_path = run_dir / "run_manifest.yaml"
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)
    robots = [str(value) for value in manifest.get("robots", [])]

    print("\n=== Inspect estimator stream timing ===", flush=True)
    topic_times = _topic_timestamps(bag_dir)
    _plot_timing(
        run_dir=run_dir,
        robots=robots,
        topic_times=topic_times,
        save=args.save,
        show=args.show,
        fmt=args.format,
    )

    if args.save:
        print(
            "\nStandalone EKF diagnostics saved in:\n  "
            f"{run_dir / 'initialization' / 'plots'}"
        )


if __name__ == "__main__":
    main()
