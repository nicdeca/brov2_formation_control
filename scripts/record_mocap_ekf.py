#!/usr/bin/env python3
"""Record a standalone MoCap-EKF diagnostic session.

This recorder is intended to run while only the estimator is active, e.g.

    ros2 launch formation_control_ros mocap_odom_ekf.launch.py \
      robots:=splash,bubble

It records, when present, for every robot:

    /mocap/<robot>/pose
    /mocap/<robot>/pose_core
    /mocap/<robot>/odom_ekf
    /<robot>/fmu/out/vehicle_odometry
    /<robot>/mavros/imu/data
    /mocap/<robot>/imu

The bag is stored in an ``initialization`` phase layout so the maintained
state exporter can be reused without duplicating conversion logic.

By default, Ctrl+C closes the bag and automatically runs the standalone EKF
plotter. Use ``--no-postprocess`` to record only.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence


def _expand(template: str, robot: str) -> str:
    return (
        template
        .replace("{robot}", robot)
        .replace("{robot_lower}", robot.lower())
    )


def _unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _topics(robots: Sequence[str]) -> list[str]:
    result = ["/clock"]
    for robot in robots:
        result.extend(
            [
                f"/mocap/{robot}/pose",
                f"/mocap/{robot}/pose_core",
                f"/mocap/{robot}/odom_ekf",
                f"/{robot}/fmu/out/vehicle_odometry",
                f"/{robot}/mavros/imu/data",
                f"/mocap/{robot}/imu",
            ]
        )
    return _unique(result)


def _write_manifest(
    run_dir: Path,
    *,
    name: str,
    robots: Sequence[str],
) -> None:
    created = datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [
        "schema_version: 4",
        f'name: "{name}"',
        'state_source: "nav_msgs"',
        'state_topic_template: "/mocap/{robot}/odom_ekf"',
        'mocap_world_frame: "core_nwu"',
        'odom_twist_frame: "body"',
        (
            'comparison_ekf_topic_template: '
            '"/mocap/{robot}/odom_ekf"'
        ),
        'mocap_pose_topic_template: "/mocap/{robot}/pose"',
        (
            'mocap_core_pose_topic_template: '
            '"/mocap/{robot}/pose_core"'
        ),
        (
            'comparison_imu_topic_template: '
            '"/{robot}/mavros/imu/data"'
        ),
        f'created_local: "{created}"',
        'recording_layout: "standalone_mocap_ekf_debug"',
        "robots:",
    ]
    lines.extend(f'  - "{robot}"' for robot in robots)
    lines.append("edges: []")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_manifest.yaml").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


class RosbagRecorder:
    def __init__(self, bag_dir: Path, topics: Sequence[str]) -> None:
        self.bag_dir = bag_dir.resolve()
        self.topics = list(topics)
        self.process: subprocess.Popen[str] | None = None

    def start(self) -> None:
        self.bag_dir.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ros2",
            "bag",
            "record",
            "-o",
            str(self.bag_dir),
            "--topics",
            *self.topics,
        ]
        print("\nRecording standalone MoCap-EKF diagnostics")
        print(f"Bag: {self.bag_dir}")
        print("Topics:")
        for topic in self.topics:
            print(f"  {topic}")
        print("\nPress Ctrl+C to stop.\n", flush=True)

        self.process = subprocess.Popen(
            command,
            text=True,
            start_new_session=True,
        )

    def stop(self, timeout: float = 12.0) -> None:
        if self.process is None:
            return

        process = self.process
        self.process = None
        if process.poll() is not None:
            return

        print("\nClosing EKF diagnostic bag...", flush=True)
        try:
            os.killpg(process.pid, signal.SIGINT)
            process.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            pass

        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=3.0)
            return
        except subprocess.TimeoutExpired:
            pass

        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--robots",
        required=True,
        help="comma-separated robot names, e.g. splash,bubble",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/ekf_checks"),
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help=(
            "optional fixed recording duration [s]; default 0 means record "
            "until Ctrl+C"
        ),
    )
    parser.add_argument(
        "--no-postprocess",
        action="store_true",
        help="close the bag without automatically exporting/plotting it",
    )
    parser.add_argument(
        "--format",
        choices=("pdf", "png", "svg"),
        default="pdf",
    )
    args = parser.parse_args()

    robots = [
        item.strip()
        for item in args.robots.split(",")
        if item.strip()
    ]
    if not robots:
        raise SystemExit("--robots must contain at least one robot")
    if len(set(robots)) != len(robots):
        raise SystemExit("--robots must contain unique robot names")
    if args.duration < 0.0:
        raise SystemExit("--duration must be nonnegative")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = args.name.strip().replace(" ", "_")
    if not safe_name:
        raise SystemExit("--name must not be empty")

    run_dir = (
        args.output_root.expanduser().resolve()
        / f"{timestamp}_{safe_name}"
    )
    bag_dir = run_dir / "initialization" / "bag"

    _write_manifest(
        run_dir,
        name=args.name,
        robots=robots,
    )

    recorder = RosbagRecorder(bag_dir, _topics(robots))
    recorder.start()

    interrupted = False
    try:
        if args.duration > 0.0:
            deadline = time.monotonic() + args.duration
            while time.monotonic() < deadline:
                if (
                    recorder.process is not None
                    and recorder.process.poll() is not None
                ):
                    raise RuntimeError(
                        "ros2 bag record exited unexpectedly"
                    )
                time.sleep(0.2)
        else:
            while True:
                if (
                    recorder.process is not None
                    and recorder.process.poll() is not None
                ):
                    raise RuntimeError(
                        "ros2 bag record exited unexpectedly"
                    )
                time.sleep(0.5)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        recorder.stop()

    print(f"\nEKF diagnostic run: {run_dir}")

    if not args.no_postprocess:
        plotter = Path(__file__).resolve().parent / "plot_mocap_ekf.py"
        command = [
            sys.executable,
            str(plotter),
            str(run_dir),
            "--save",
            "--format",
            args.format,
        ]
        print("\n=== Export and plot EKF diagnostics ===", flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            print(
                "WARNING: automatic EKF post-processing failed. "
                "The rosbag was saved and can be processed manually with:\n"
                f"  python3 {plotter} {run_dir} --save",
                file=sys.stderr,
            )

    if interrupted:
        print("Recording stopped by user.")


if __name__ == "__main__":
    main()
