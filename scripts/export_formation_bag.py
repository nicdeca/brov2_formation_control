#!/usr/bin/env python3
"""Export a formation-control rosbag to a portable ROS-independent NPZ.

Run from the ROS environment, e.g.::

    source setup_ros2.sh
    python scripts/export_formation_bag.py outputs/experiments/<run>

The resulting ``formation_history.npz`` is the only input needed by
``plot_formation_experiment.py``.
"""

from __future__ import annotations

import argparse
import bisect
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from formation_control.experiment import (
    FIELD_WIDTHS,
    FormationExperimentHistory,
    unpack_snapshot,
)

try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except ImportError as error:  # pragma: no cover - ROS-only script
    raise SystemExit(
        "ROS Python packages are unavailable. Run `source setup_ros2.sh` first."
    ) from error


S_NED_TO_NWU = np.diag([1.0, -1.0, -1.0])


@dataclass(frozen=True)
class TopicSeries:
    records: list[tuple[int, Any]]
    times_ns: list[int]

    @classmethod
    def from_records(cls, records: list[tuple[int, Any]]) -> "TopicSeries":
        return cls(records=records, times_ns=[item[0] for item in records])

    def nearest(self, timestamp_ns: int, max_delta_ns: int) -> Any | None:
        if not self.records:
            return None
        index = bisect.bisect_left(self.times_ns, timestamp_ns)
        candidates: list[tuple[int, Any]] = []
        if index < len(self.records):
            candidates.append(self.records[index])
        if index > 0:
            candidates.append(self.records[index - 1])
        stamp, message = min(
            candidates,
            key=lambda item: abs(item[0] - timestamp_ns),
        )
        if abs(stamp - timestamp_ns) > max_delta_ns:
            return None
        return message


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    norm = float(np.linalg.norm(q))
    if norm <= 0.0:
        raise ValueError("zero-norm quaternion in PX4 odometry")
    w, x, y, z = q / norm
    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=float,
    )


def matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Return a normalized scalar-first quaternion for a rotation matrix."""
    r = np.asarray(rotation, dtype=float)
    trace = float(np.trace(r))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        q = np.array(
            [
                0.25 * scale,
                (r[2, 1] - r[1, 2]) / scale,
                (r[0, 2] - r[2, 0]) / scale,
                (r[1, 0] - r[0, 1]) / scale,
            ]
        )
    else:
        diagonal_index = int(np.argmax(np.diag(r)))
        if diagonal_index == 0:
            scale = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            q = np.array(
                [
                    (r[2, 1] - r[1, 2]) / scale,
                    0.25 * scale,
                    (r[0, 1] + r[1, 0]) / scale,
                    (r[0, 2] + r[2, 0]) / scale,
                ]
            )
        elif diagonal_index == 1:
            scale = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            q = np.array(
                [
                    (r[0, 2] - r[2, 0]) / scale,
                    (r[0, 1] + r[1, 0]) / scale,
                    0.25 * scale,
                    (r[1, 2] + r[2, 1]) / scale,
                ]
            )
        else:
            scale = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            q = np.array(
                [
                    (r[1, 0] - r[0, 1]) / scale,
                    (r[0, 2] + r[2, 0]) / scale,
                    (r[1, 2] + r[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    q /= np.linalg.norm(q)
    if q[0] < 0.0:
        q = -q
    return q


def px4_odometry_to_core(message) -> dict[str, np.ndarray]:
    """Convert PX4 NED/FRD VehicleOdometry to core NWU/FLU."""
    if int(message.pose_frame) != 1:
        raise ValueError(
            f"expected PX4 NED pose_frame=1, got {message.pose_frame}"
        )
    if int(message.velocity_frame) != 1:
        raise ValueError(
            f"expected PX4 NED velocity_frame=1, got {message.velocity_frame}"
        )

    p_ned = np.asarray(message.position, dtype=float)
    v_ned = np.asarray(message.velocity, dtype=float)
    q_frd_to_ned = np.asarray(message.q, dtype=float)
    omega_frd = np.asarray(message.angular_velocity, dtype=float)

    r_ned_frd = quaternion_to_matrix(q_frd_to_ned)
    r_nwu_flu = S_NED_TO_NWU @ r_ned_frd @ S_NED_TO_NWU

    p_nwu = S_NED_TO_NWU @ p_ned
    v_nwu = S_NED_TO_NWU @ v_ned
    v_body_flu = r_nwu_flu.T @ v_nwu
    omega_flu = S_NED_TO_NWU @ omega_frd

    return {
        "position": p_nwu,
        "quaternion": matrix_to_quaternion(r_nwu_flu),
        "linear_velocity_body": v_body_flu,
        "angular_velocity_body": omega_flu,
    }


def px4_time_seconds(message) -> float | None:
    """Prefer PX4 simulation/sample time when available."""
    for field in ("timestamp_sample", "timestamp"):
        value = int(getattr(message, field, 0))
        if value > 0:
            return 1e-6 * value
    return None


def storage_identifier(bag_dir: Path) -> str:
    with (bag_dir / "metadata.yaml").open("r", encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream)
    info = metadata.get("rosbag2_bagfile_information", metadata)
    return str(info["storage_identifier"])


def read_bag(bag_dir: Path) -> dict[str, TopicSeries]:
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(
            uri=str(bag_dir),
            storage_id=storage_identifier(bag_dir),
        ),
        rosbag2_py.ConverterOptions("", ""),
    )

    topic_types = {
        item.name: item.type for item in reader.get_all_topics_and_types()
    }
    message_classes = {
        topic: get_message(type_name) for topic, type_name in topic_types.items()
    }

    raw: dict[str, list[tuple[int, Any]]] = defaultdict(list)
    while reader.has_next():
        topic, data, timestamp_ns = reader.read_next()
        raw[topic].append(
            (
                int(timestamp_ns),
                deserialize_message(data, message_classes[topic]),
            )
        )
    return {topic: TopicSeries.from_records(records) for topic, records in raw.items()}


def load_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_manifest.yaml"
    if not path.exists():
        raise FileNotFoundError(f"missing run manifest: {path}")
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _snapshot_array_shape(field: str, n_samples: int, n_agents: int) -> tuple[int, ...]:
    width = int(FIELD_WIDTHS[field])
    if width == 1:
        return (n_samples, n_agents)
    return (n_samples, n_agents, width)


def _message_xyz(message) -> np.ndarray:
    value = getattr(message, "xyz", None)
    if value is None:
        return np.full(3, np.nan)
    return np.asarray(value, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="default: <run_dir>/formation_history.npz",
    )
    parser.add_argument(
        "--reference-robot",
        default=None,
        help="robot whose diagnostic snapshots define the common control grid",
    )
    parser.add_argument(
        "--max-sync-ms",
        type=float,
        default=30.0,
        help="nearest-neighbor synchronization tolerance (default: 30 ms)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    manifest = load_manifest(run_dir)
    robots = [str(robot) for robot in manifest["robots"]]
    edges = list(manifest.get("edges", []))
    bag = read_bag(run_dir / "bag")

    diagnostic_topics = {
        robot: f"/{robot}/formation_control/diagnostic_snapshot" for robot in robots
    }
    odometry_topics = {
        robot: f"/{robot}/fmu/out/vehicle_odometry" for robot in robots
    }
    thrust_topics = {
        robot: f"/{robot}/fmu/in/vehicle_thrust_setpoint" for robot in robots
    }
    torque_topics = {
        robot: f"/{robot}/fmu/in/vehicle_torque_setpoint" for robot in robots
    }
    mode_topics = {
        robot: f"/{robot}/fmu/out/vehicle_control_mode" for robot in robots
    }

    reference_robot = args.reference_robot
    if reference_robot is None:
        follower_names = [str(edge["observer"]) for edge in edges]
        candidates = follower_names if follower_names else robots
        reference_robot = max(
            candidates,
            key=lambda robot: len(
                bag.get(
                    diagnostic_topics[robot],
                    TopicSeries([], []),
                ).records
            ),
        )
    if reference_robot not in robots:
        raise SystemExit(f"unknown reference robot {reference_robot!r}")

    reference_diagnostics = bag.get(diagnostic_topics[reference_robot])
    if reference_diagnostics is None or not reference_diagnostics.records:
        raise SystemExit(
            f"No diagnostic snapshots found for {reference_robot}. "
            "The online snapshot publisher must be integrated before recording."
        )

    max_delta_ns = int(args.max_sync_ms * 1e6)
    required_diag_robots = {str(edge["observer"]) for edge in edges}

    valid_bag_stamps: list[int] = []
    for bag_stamp, _ in reference_diagnostics.records:
        if any(
            odometry_topics[robot] not in bag
            or bag[odometry_topics[robot]].nearest(bag_stamp, max_delta_ns) is None
            for robot in robots
        ):
            continue
        if any(
            diagnostic_topics[robot] not in bag
            or bag[diagnostic_topics[robot]].nearest(bag_stamp, max_delta_ns) is None
            for robot in required_diag_robots
        ):
            continue
        valid_bag_stamps.append(bag_stamp)

    if len(valid_bag_stamps) < 2:
        raise SystemExit("Too few synchronized controller/odometry samples were found.")

    n_samples = len(valid_bag_stamps)
    n_agents = len(robots)

    positions = np.full((n_samples, n_agents, 3), np.nan)
    quaternions = np.full((n_samples, n_agents, 4), np.nan)
    linear_velocity_body = np.full((n_samples, n_agents, 3), np.nan)
    angular_velocity_body = np.full((n_samples, n_agents, 3), np.nan)
    px4_thrust_setpoint = np.full((n_samples, n_agents, 3), np.nan)
    px4_torque_setpoint = np.full((n_samples, n_agents, 3), np.nan)
    px4_armed = np.full((n_samples, n_agents), np.nan)
    px4_offboard_enabled = np.full((n_samples, n_agents), np.nan)

    snapshot_fields = [name for name in FIELD_WIDTHS if name != "schema_version"]
    snapshots = {
        field: np.full(
            _snapshot_array_shape(field, n_samples, n_agents),
            np.nan,
        )
        for field in snapshot_fields
    }

    reference_px4_times = np.full(n_samples, np.nan)

    for step, bag_stamp in enumerate(valid_bag_stamps):
        for agent, robot in enumerate(robots):
            odom_msg = bag[odometry_topics[robot]].nearest(
                bag_stamp,
                max_delta_ns,
            )
            assert odom_msg is not None
            state = px4_odometry_to_core(odom_msg)
            positions[step, agent] = state["position"]
            quaternions[step, agent] = state["quaternion"]
            linear_velocity_body[step, agent] = state["linear_velocity_body"]
            angular_velocity_body[step, agent] = state["angular_velocity_body"]

            if robot == reference_robot:
                px4_time = px4_time_seconds(odom_msg)
                if px4_time is not None:
                    reference_px4_times[step] = px4_time

            diag_series = bag.get(diagnostic_topics[robot])
            if diag_series is not None:
                diag_msg = diag_series.nearest(bag_stamp, max_delta_ns)
                if diag_msg is not None:
                    decoded = unpack_snapshot(diag_msg.data)
                    for field in snapshots:
                        snapshots[field][step, agent] = decoded[field]

            thrust_series = bag.get(thrust_topics[robot])
            if thrust_series is not None:
                message = thrust_series.nearest(bag_stamp, max_delta_ns)
                if message is not None:
                    px4_thrust_setpoint[step, agent] = _message_xyz(message)

            torque_series = bag.get(torque_topics[robot])
            if torque_series is not None:
                message = torque_series.nearest(bag_stamp, max_delta_ns)
                if message is not None:
                    px4_torque_setpoint[step, agent] = _message_xyz(message)

            mode_series = bag.get(mode_topics[robot])
            if mode_series is not None:
                message = mode_series.nearest(bag_stamp, max_delta_ns)
                if message is not None:
                    px4_armed[step, agent] = float(
                        bool(getattr(message, "flag_armed", False))
                    )
                    px4_offboard_enabled[step, agent] = float(
                        bool(
                            getattr(
                                message,
                                "flag_control_offboard_enabled",
                                False,
                            )
                        )
                    )

    # Prefer PX4's simulation/sample clock. Fall back to rosbag receive time if
    # PX4 timestamps are unavailable or non-monotone.
    if np.all(np.isfinite(reference_px4_times)) and np.all(
        np.diff(reference_px4_times) > 0.0
    ):
        times = reference_px4_times - reference_px4_times[0]
        time_source = "px4_timestamp"
    else:
        stamps = np.asarray(valid_bag_stamps, dtype=np.int64)
        times = (stamps - stamps[0]) * 1e-9
        time_source = "rosbag_receive_timestamp"

    arrays = {
        "times": np.asarray(times, dtype=float),
        "bag_timestamps_ns": np.asarray(valid_bag_stamps, dtype=np.int64),
        "positions": positions,
        "quaternions": quaternions,
        "linear_velocity_body": linear_velocity_body,
        "angular_velocity_body": angular_velocity_body,
        "px4_thrust_setpoint": px4_thrust_setpoint,
        "px4_torque_setpoint": px4_torque_setpoint,
        "px4_armed": px4_armed,
        "px4_offboard_enabled": px4_offboard_enabled,
        **snapshots,
    }
    metadata = {
        "schema_version": 1,
        "robots": robots,
        "edges": edges,
        "reference_robot": reference_robot,
        "source_run_dir": str(run_dir),
        "max_sync_ms": float(args.max_sync_ms),
        "time_source": time_source,
    }

    output = args.output or run_dir / "formation_history.npz"
    FormationExperimentHistory(arrays=arrays, metadata=metadata).save(output)

    print(f"Exported {n_samples} synchronized controller samples")
    print(f"Duration: {times[-1]:.3f} s")
    print(f"Time source: {time_source}")
    print(f"History: {output}")


if __name__ == "__main__":
    main()
