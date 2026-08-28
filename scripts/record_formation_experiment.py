#!/usr/bin/env python3
"""Record initialization and mission into separate rosbag directories.

The recorder is meant to be started before arming. It immediately records the
initialization phase. When /formation_control/experiment_phase becomes
FORMATION, the initialization bag is closed. It then waits for the experiment
runner to publish /formation_control/mission_status = RUNNING, records the
mission, and closes the mission bag on COMPLETE or ABORTED.

By default, both phases are exported after recording. Initialization receives
dedicated diagnostic plots; mission data uses the standard experiment plotter.
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

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import String


PHASE_TOPIC = "/formation_control/experiment_phase"
MISSION_STATUS_TOPIC = "/formation_control/mission_status"


def recorder_topics(robots: Sequence[str]) -> list[str]:
    topics = [
        "/clock",
        PHASE_TOPIC,
        MISSION_STATUS_TOPIC,
        "/formation_control/desired_formation",
    ]

    for robot in robots:
        prefix = f"/{robot}"
        topics.extend(
            [
                f"{prefix}/fmu/out/vehicle_odometry",
                f"{prefix}/fmu/out/vehicle_control_mode",
                f"{prefix}/fmu/in/vehicle_thrust_setpoint",
                f"{prefix}/fmu/in/vehicle_torque_setpoint",
                f"{prefix}/formation_control/diagnostic_snapshot",
                f"{prefix}/formation_control/cmd_vel",
                f"{prefix}/formation_control/fallback",
                f"{prefix}/formation_control/slack",
                f"{prefix}/formation_control/required_slack",
                f"{prefix}/formation_control/actuation_margin",
                f"{prefix}/formation_control/thruster_utilization",
                f"{prefix}/formation_control/minimum_physical_margin",
                f"{prefix}/formation_control/conservative_constraint_values",
                f"{prefix}/formation_control/domain_relaxation",
                f"{prefix}/formation_control/workspace_barrier_enabled",
                f"{prefix}/formation_control/workspace_adaptive",
                f"{prefix}/formation_control/workspace_barrier_value",
                f"{prefix}/formation_control/workspace_relaxation",
                f"{prefix}/formation_control/workspace_conservative_constraint_values",
                f"{prefix}/formation_control/workspace_physical_constraint_values",
                f"{prefix}/formation_control/workspace_minimum_physical_margin",
            ]
        )
    return topics


class RosbagRecorder:
    def __init__(self, bag_dir: Path, topics: Sequence[str]) -> None:
        self.bag_dir = bag_dir.resolve()
        self.topics = list(topics)
        self.process: subprocess.Popen[str] | None = None

    def start(self) -> None:
        if self.process is not None:
            raise RuntimeError("rosbag recorder is already running")
        self.bag_dir.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ros2",
            "bag",
            "record",
            "-o",
            str(self.bag_dir),
            *self.topics,
        ]
        print(f"Starting rosbag: {self.bag_dir}", flush=True)
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

        print(f"Closing rosbag: {self.bag_dir}", flush=True)
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


class SplitRecorderNode(Node):
    def __init__(self) -> None:
        super().__init__("formation_experiment_split_recorder")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.phase: str | None = None
        self.mission_status: str | None = None

        self.create_subscription(
            String,
            PHASE_TOPIC,
            self._phase_callback,
            qos,
        )
        self.create_subscription(
            String,
            MISSION_STATUS_TOPIC,
            self._mission_status_callback,
            qos,
        )

    def _phase_callback(self, message: String) -> None:
        value = message.data.strip().upper()
        if value != self.phase:
            self.get_logger().info(f"Experiment phase: {value}")
        self.phase = value

    def _mission_status_callback(self, message: String) -> None:
        value = message.data.strip().upper()
        if value != self.mission_status:
            self.get_logger().info(f"Mission status: {value}")
        self.mission_status = value

    def wait_for_phase(self, desired: str) -> None:
        desired = desired.upper()
        while rclpy.ok() and self.phase != desired:
            rclpy.spin_once(self, timeout_sec=0.2)
        if not rclpy.ok():
            raise KeyboardInterrupt

    def wait_for_mission_status(self, desired: set[str]) -> str:
        desired = {value.upper() for value in desired}
        while rclpy.ok() and self.mission_status not in desired:
            rclpy.spin_once(self, timeout_sec=0.2)
        if not rclpy.ok():
            raise KeyboardInterrupt
        assert self.mission_status is not None
        return self.mission_status


def write_manifest(
    run_dir: Path,
    *,
    name: str,
    robots: Sequence[str],
    edges: Sequence[str],
) -> None:
    lines = [
        "schema_version: 2",
        f'name: "{name}"',
        f'created_local: "{datetime.now().astimezone().isoformat(timespec="seconds")}"',
        "recording_layout: split_initialization_mission",
        "robots:",
    ]
    lines.extend(f'  - "{robot}"' for robot in robots)
    lines.append("edges:")
    for edge in edges:
        follower, parent = edge.split(":", 1)
        lines.extend(
            [
                f'  - observer: "{follower}"',
                f'    target: "{parent}"',
            ]
        )
    (run_dir / "run_manifest.yaml").write_text("\n".join(lines) + "\n")


def run_postprocessing(run_dir: Path, *, mission_exists: bool) -> None:
    """Export and plot every phase available in the completed run."""
    scripts_dir = Path(__file__).resolve().parent
    export_script = scripts_dir / "export_experiment.py"
    plot_script = scripts_dir / "plot_experiment.py"

    def call(command: list[str], label: str) -> None:
        print(f"\n=== {label} ===", flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            print(
                f"WARNING: {label} returned exit code {result.returncode}.",
                file=sys.stderr,
            )

    # `export_experiment.py` and `plot_experiment.py` both skip unavailable
    # phases. This means a failed initialization run still produces useful
    # initialization output without requiring a mission bag.
    call(
        [sys.executable, str(export_script), str(run_dir)],
        "Export experiment",
    )
    call(
        [sys.executable, str(plot_script), str(run_dir)],
        "Plot experiment",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--robots",
        required=True,
        help="comma-separated robot names",
    )
    parser.add_argument(
        "--edge",
        action="append",
        default=[],
        help="directed sensing edge follower:parent; repeat as needed",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/experiments"),
    )
    parser.add_argument(
        "--no-postprocess",
        action="store_true",
        help="do not automatically export/plot after recording",
    )
    args = parser.parse_args()

    robots = [item.strip() for item in args.robots.split(",") if item.strip()]
    if not robots:
        parser.error("--robots must contain at least one robot")
    if len(set(robots)) != len(robots):
        parser.error("--robots contains duplicate names")

    for edge in args.edge:
        if edge.count(":") != 1:
            parser.error(f"invalid --edge {edge!r}; expected follower:parent")
        follower, parent = edge.split(":", 1)
        if follower not in robots or parent not in robots:
            parser.error(
                f"edge {edge!r} references a robot not listed in --robots"
            )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = args.output_root.expanduser().resolve()
    run_dir = (output_root / f"{stamp}_{args.name}").resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    write_manifest(
        run_dir,
        name=args.name,
        robots=robots,
        edges=args.edge,
    )

    topics = recorder_topics(robots)
    initialization = RosbagRecorder(
        run_dir / "initialization" / "bag",
        topics,
    )
    mission = RosbagRecorder(
        run_dir / "mission" / "bag",
        topics,
    )

    print(f"Run directory: {run_dir}")
    print(f"Initialization folder: {(run_dir / 'initialization').resolve()}")
    print(f"Mission folder: {(run_dir / 'mission').resolve()}")
    print(f"Robots: {', '.join(robots)}")
    print(f"Edges: {', '.join(args.edge) if args.edge else '(none)'}")
    print(f"Recording {len(topics)} explicit topics.")
    print("Start this recorder before arming. Ctrl-C is always safe.")

    rclpy.init()
    node = SplitRecorderNode()
    mission_started = False

    try:
        initialization.start()
        node.get_logger().info(
            "Recording initialization until experiment phase FORMATION."
        )
        node.wait_for_phase("FORMATION")
        initialization.stop()

        node.get_logger().info(
            "Initialization complete. Waiting for mission status RUNNING."
        )
        status = node.wait_for_mission_status(
            {"RUNNING", "ABORTED", "COMPLETE"}
        )

        if status == "RUNNING":
            mission.start()
            mission_started = True
            node.get_logger().info(
                "Recording mission until mission status COMPLETE or ABORTED."
            )
            final_status = node.wait_for_mission_status(
                {"COMPLETE", "ABORTED"}
            )
            mission.stop()
            node.get_logger().info(f"Mission recording closed on {final_status}.")
        else:
            node.get_logger().warn(
                f"Mission never entered RUNNING; current status is {status}."
            )

    except KeyboardInterrupt:
        node.get_logger().warn("Recorder interrupted; closing active bag cleanly.")
    finally:
        initialization.stop()
        mission.stop()
        node.destroy_node()
        rclpy.shutdown()

    if not args.no_postprocess:
        run_postprocessing(run_dir, mission_exists=mission_started)

    print("\nRecording workflow complete.")
    print(f"Absolute run folder: {run_dir}")


if __name__ == "__main__":
    main()
