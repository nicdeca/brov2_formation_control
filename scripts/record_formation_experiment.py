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


def _expand_topic_template(template: str, robot: str) -> str:
    return (
        template.replace("{robot}", robot)
        .replace("{robot_lower}", robot.lower())
    )


def _unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def recorder_topics(
    robots: Sequence[str],
    *,
    state_topic_template: str,
    ekf_topic_template: str,
    mocap_pose_topic_template: str,
    mocap_core_pose_topic_template: str,
    imu_topic_template: str,
) -> list[str]:
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
                # Always record PX4, raw MoCap, and the in-repository EKF so
                # estimator comparisons are available regardless of which
                # source drives the controller.
                f"{prefix}/fmu/out/vehicle_odometry",
                _expand_topic_template(mocap_pose_topic_template, robot),
                _expand_topic_template(mocap_core_pose_topic_template, robot),
                _expand_topic_template(ekf_topic_template, robot),
                _expand_topic_template(state_topic_template, robot),
                _expand_topic_template(imu_topic_template, robot),
                # Also retain both common gyro sources when present so a run
                # can diagnose estimator-rate problems after the fact.
                f"/{robot}/mavros/imu/data",
                f"/mocap/{robot}/imu",
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
    return _unique(topics)


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
            "--topics",
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
    state_source: str,
    state_topic_template: str,
    mocap_world_frame: str,
    odom_twist_frame: str,
    comparison_ekf_topic_template: str,
    mocap_pose_topic_template: str,
    mocap_core_pose_topic_template: str,
    imu_topic_template: str,
) -> None:
    lines = [
        "schema_version: 4",
        f'name: "{name}"',
        f'state_source: "{state_source}"',
        f'state_topic_template: "{state_topic_template}"',
        f'mocap_world_frame: "{mocap_world_frame}"',
        f'odom_twist_frame: "{odom_twist_frame}"',
        (
            'comparison_ekf_topic_template: '
            f'"{comparison_ekf_topic_template}"'
        ),
        (
            'mocap_pose_topic_template: '
            f'"{mocap_pose_topic_template}"'
        ),
        (
            'mocap_core_pose_topic_template: '
            f'"{mocap_core_pose_topic_template}"'
        ),
        (
            'comparison_imu_topic_template: '
            f'"{imu_topic_template}"'
        ),
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
    """Export and plot every experiment phase that is available."""
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

    # The run-level exporter and plotter discover which split phases are
    # present. A partial run therefore still produces useful output, while a
    # complete run processes initialization and mission in the same commands.
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
        "--state-source",
        choices=("px4", "nav_msgs"),
        default="px4",
        help=(
            "state source used by the controller; stored in the run manifest "
            "so exported canonical trajectories match the online controller"
        ),
    )
    parser.add_argument(
        "--state-topic-template",
        default="",
        help=(
            "per-robot controller state topic template using {robot} or "
            "{robot_lower}; empty selects the source default"
        ),
    )
    parser.add_argument(
        "--mocap-world-frame",
        default="core_nwu",
        help="generic-Odometry world-frame convention stored in the manifest",
    )
    parser.add_argument(
        "--odom-twist-frame",
        default="body",
        help="generic-Odometry twist-frame convention stored in the manifest",
    )
    parser.add_argument(
        "--ekf-topic-template",
        default="/mocap/{robot}/odom_ekf",
        help="MoCap-EKF topic recorded for offline estimator comparison",
    )
    parser.add_argument(
        "--mocap-pose-topic-template",
        default="/mocap/{robot}/pose",
        help="raw MoCap PoseStamped topic recorded for estimator diagnostics",
    )
    parser.add_argument(
        "--mocap-core-pose-topic-template",
        default="/mocap/{robot}/pose_core",
        help=(
            "transformed raw MoCap pose in core NWU/FLU, recorded for "
            "three-way estimator comparison"
        ),
    )
    parser.add_argument(
        "--imu-topic-template",
        default="/{robot}/mavros/imu/data",
        help=(
            "gyro/IMU topic associated with the MoCap estimator; the recorder "
            "also retains the standard MAVROS and simulated-MoCap gyro topics"
        ),
    )
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

    state_topic_template = args.state_topic_template.strip()
    if not state_topic_template:
        state_topic_template = (
            "/{robot}/fmu/out/vehicle_odometry"
            if args.state_source == "px4"
            else "/mocap/{robot}/odom_ekf"
        )

    templates = {
        "state topic": state_topic_template,
        "EKF topic": args.ekf_topic_template.strip(),
        "MoCap pose topic": args.mocap_pose_topic_template.strip(),
        "core MoCap pose topic": (
            args.mocap_core_pose_topic_template.strip()
        ),
        "IMU topic": args.imu_topic_template.strip(),
    }
    if len(robots) > 1:
        for label, template in templates.items():
            if "{robot}" not in template and "{robot_lower}" not in template:
                parser.error(
                    f"{label} template must contain '{{robot}}' or "
                    "'{robot_lower}' for a multi-robot recording"
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
        state_source=args.state_source,
        state_topic_template=state_topic_template,
        mocap_world_frame=args.mocap_world_frame,
        odom_twist_frame=args.odom_twist_frame,
        comparison_ekf_topic_template=args.ekf_topic_template.strip(),
        mocap_pose_topic_template=args.mocap_pose_topic_template.strip(),
        mocap_core_pose_topic_template=(
            args.mocap_core_pose_topic_template.strip()
        ),
        imu_topic_template=args.imu_topic_template.strip(),
    )

    topics = recorder_topics(
        robots,
        state_topic_template=state_topic_template,
        ekf_topic_template=args.ekf_topic_template.strip(),
        mocap_pose_topic_template=args.mocap_pose_topic_template.strip(),
        mocap_core_pose_topic_template=(
            args.mocap_core_pose_topic_template.strip()
        ),
        imu_topic_template=args.imu_topic_template.strip(),
    )
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
    print(f"Controller state source: {args.state_source}")
    print(f"Controller state topic: {state_topic_template}")
    print(f"Comparison EKF topic: {args.ekf_topic_template.strip()}")
    print(
        "Core MoCap pose topic: "
        f"{args.mocap_core_pose_topic_template.strip()}"
    )
    print(f"IMU topic: {args.imu_topic_template.strip()}")
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
        if rclpy.ok():
            rclpy.shutdown()

    if not args.no_postprocess:
        run_postprocessing(run_dir, mission_exists=mission_started)

    print("\nRecording workflow complete.")
    print(f"Absolute run folder: {run_dir}")


if __name__ == "__main__":
    main()
