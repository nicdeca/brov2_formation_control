#!/usr/bin/env python3
"""Export an initialization rosbag without requiring valid controller outputs.

Synchronized PX4 odometry is sufficient. Controller diagnostics are decoded
when available. Explicit initialization targets and workspace bounds can be
supplied as fallbacks for older bags that were recorded before those data were
published correctly.
"""

from __future__ import annotations

import argparse
import bisect
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except ImportError as error:
    raise SystemExit(
        "ROS Python packages are unavailable. Run `source setup_ros2.sh` first."
    ) from error

try:
    from formation_control.experiment import unpack_snapshot
except ImportError:
    unpack_snapshot = None


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
    return {
        topic: TopicSeries.from_records(records)
        for topic, records in raw.items()
    }


def load_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_manifest.yaml"
    if not path.exists():
        raise FileNotFoundError(f"missing run manifest: {path}")
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _message_numeric_array(message: Any, width: int) -> np.ndarray | None:
    """Extract a finite numeric vector from common ROS message layouts."""
    for attribute in ("data", "values", "value", "xyz"):
        if not hasattr(message, attribute):
            continue
        try:
            value = np.asarray(getattr(message, attribute), dtype=float).reshape(-1)
        except (TypeError, ValueError):
            continue
        if value.size != width:
            continue
        if not np.all(np.isfinite(value)):
            continue
        return value
    return None


def _message_numeric_scalar(message: Any) -> float | None:
    """Extract one finite scalar from common ROS message layouts."""
    for attribute in ("data", "value"):
        if not hasattr(message, attribute):
            continue
        try:
            value = np.asarray(getattr(message, attribute), dtype=float).reshape(-1)
        except (TypeError, ValueError):
            continue
        if value.size != 1 or not np.isfinite(value[0]):
            continue
        return float(value[0])
    return None


def _nearest_vector(
    bag: dict[str, TopicSeries],
    topic: str,
    timestamp_ns: int,
    max_delta_ns: int,
    width: int,
) -> np.ndarray | None:
    series = bag.get(topic)
    if series is None:
        return None
    message = series.nearest(timestamp_ns, max_delta_ns)
    if message is None:
        return None
    return _message_numeric_array(message, width)


def _nearest_scalar(
    bag: dict[str, TopicSeries],
    topic: str,
    timestamp_ns: int,
    max_delta_ns: int,
) -> float | None:
    series = bag.get(topic)
    if series is None:
        return None
    message = series.nearest(timestamp_ns, max_delta_ns)
    if message is None:
        return None
    return _message_numeric_scalar(message)


def _parse_initial_target(spec: str) -> tuple[str, np.ndarray]:
    if ":" not in spec:
        raise argparse.ArgumentTypeError(
            "expected ROBOT:x,y,z, for example glub:4.475,0.750,-0.775"
        )
    robot, coordinates = spec.split(":", 1)
    values = np.fromstring(coordinates, sep=",", dtype=float)
    if not robot.strip() or values.shape != (3,) or not np.all(np.isfinite(values)):
        raise argparse.ArgumentTypeError(
            "expected ROBOT:x,y,z with three finite coordinates"
        )
    return robot.strip(), values


def _workspace_margins_from_bounds(
    positions: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Compute [x_min,x_max,y_min,y_max,z_min,z_max] wall margins."""
    result = np.full((positions.shape[0], positions.shape[1], 6), np.nan)
    for axis in range(3):
        result[:, :, 2 * axis] = positions[:, :, axis] - lower[axis]
        result[:, :, 2 * axis + 1] = upper[axis] - positions[:, :, axis]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="default: <run_dir>/initialization/initialization_history.npz",
    )
    parser.add_argument(
        "--reference-robot",
        default=None,
        help="robot whose PX4 odometry defines the synchronization grid",
    )
    parser.add_argument(
        "--max-sync-ms",
        type=float,
        default=40.0,
    )
    parser.add_argument(
        "--initial-target",
        action="append",
        default=[],
        type=_parse_initial_target,
        metavar="ROBOT:X,Y,Z",
        help=(
            "absolute core-NWU initialization target for an older bag whose "
            "controller snapshot did not contain the target; repeat per robot"
        ),
    )
    parser.add_argument(
        "--workspace-physical-lower",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=None,
    )
    parser.add_argument(
        "--workspace-physical-upper",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=None,
    )
    parser.add_argument(
        "--workspace-conservative-lower",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=None,
    )
    parser.add_argument(
        "--workspace-conservative-upper",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=None,
    )
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    bag_dir = run_dir / "initialization" / "bag"
    if not bag_dir.exists():
        raise FileNotFoundError(f"missing initialization bag: {bag_dir}")

    manifest = load_manifest(run_dir)
    robots = [str(robot) for robot in manifest["robots"]]
    robot_index = {robot: index for index, robot in enumerate(robots)}
    bag = read_bag(bag_dir)

    odometry_topics = {
        robot: f"/{robot}/fmu/out/vehicle_odometry" for robot in robots
    }
    mode_topics = {
        robot: f"/{robot}/fmu/out/vehicle_control_mode" for robot in robots
    }
    diagnostic_topics = {
        robot: f"/{robot}/formation_control/diagnostic_snapshot"
        for robot in robots
    }
    workspace_relaxation_topics = {
        robot: f"/{robot}/formation_control/workspace_relaxation"
        for robot in robots
    }
    workspace_physical_topics = {
        robot: (
            f"/{robot}/formation_control/"
            "workspace_physical_constraint_values"
        )
        for robot in robots
    }
    workspace_conservative_topics = {
        robot: (
            f"/{robot}/formation_control/"
            "workspace_conservative_constraint_values"
        )
        for robot in robots
    }
    required_slack_topics = {
        robot: f"/{robot}/formation_control/required_slack"
        for robot in robots
    }
    thruster_utilization_topics = {
        robot: f"/{robot}/formation_control/thruster_utilization"
        for robot in robots
    }

    reference_robot = args.reference_robot or robots[0]
    if reference_robot not in robots:
        raise SystemExit(f"unknown reference robot {reference_robot!r}")
    reference_series = bag.get(odometry_topics[reference_robot])
    if reference_series is None or not reference_series.records:
        raise SystemExit(
            f"No PX4 odometry found for reference robot {reference_robot}."
        )

    max_delta_ns = int(args.max_sync_ms * 1e6)
    stamps: list[int] = []
    for stamp, _ in reference_series.records:
        if all(
            odometry_topics[robot] in bag
            and bag[odometry_topics[robot]].nearest(stamp, max_delta_ns)
            is not None
            for robot in robots
        ):
            stamps.append(stamp)

    if len(stamps) < 2:
        raise SystemExit("Too few synchronized initialization odometry samples.")

    n = len(stamps)
    m = len(robots)

    positions = np.full((n, m, 3), np.nan)
    quaternions = np.full((n, m, 4), np.nan)
    linear_velocity_body = np.full((n, m, 3), np.nan)
    angular_velocity_body = np.full((n, m, 3), np.nan)
    px4_armed = np.full((n, m), np.nan)
    px4_offboard_enabled = np.full((n, m), np.nan)

    reference_position = np.full((n, m, 3), np.nan)
    desired_relative_position = np.full((n, m, 3), np.nan)
    required_slack = np.full((n, m), np.nan)
    thruster_utilization = np.full((n, m), np.nan)
    workspace_relaxation = np.full((n, m, 6), np.nan)
    workspace_physical_constraint_values = np.full((n, m, 6), np.nan)
    workspace_conservative_constraint_values = np.full((n, m, 6), np.nan)

    reference_px4_times = np.full(n, np.nan)

    for step, stamp in enumerate(stamps):
        for agent, robot in enumerate(robots):
            odom = bag[odometry_topics[robot]].nearest(stamp, max_delta_ns)
            assert odom is not None
            state = px4_odometry_to_core(odom)
            positions[step, agent] = state["position"]
            quaternions[step, agent] = state["quaternion"]
            linear_velocity_body[step, agent] = state["linear_velocity_body"]
            angular_velocity_body[step, agent] = state["angular_velocity_body"]

            if robot == reference_robot:
                t_px4 = px4_time_seconds(odom)
                if t_px4 is not None:
                    reference_px4_times[step] = t_px4

            mode_series = bag.get(mode_topics[robot])
            if mode_series is not None:
                mode = mode_series.nearest(stamp, max_delta_ns)
                if mode is not None:
                    px4_armed[step, agent] = float(
                        bool(getattr(mode, "flag_armed", False))
                    )
                    px4_offboard_enabled[step, agent] = float(
                        bool(
                            getattr(
                                mode,
                                "flag_control_offboard_enabled",
                                False,
                            )
                        )
                    )

            if unpack_snapshot is not None:
                diag_series = bag.get(diagnostic_topics[robot])
                if diag_series is not None:
                    diag = diag_series.nearest(stamp, max_delta_ns)
                    if diag is not None:
                        try:
                            decoded = unpack_snapshot(diag.data)
                        except Exception:
                            decoded = {}
                        if "reference_position" in decoded:
                            reference_position[step, agent] = decoded[
                                "reference_position"
                            ]
                        if "desired_relative_position" in decoded:
                            desired_relative_position[step, agent] = decoded[
                                "desired_relative_position"
                            ]
                        if "required_slack" in decoded:
                            required_slack[step, agent] = decoded[
                                "required_slack"
                            ]
                        if "thruster_utilization" in decoded:
                            thruster_utilization[step, agent] = decoded[
                                "thruster_utilization"
                            ]
                        if "workspace_relaxation" in decoded:
                            workspace_relaxation[step, agent] = decoded[
                                "workspace_relaxation"
                            ]
                        if "workspace_physical_constraint_values" in decoded:
                            workspace_physical_constraint_values[
                                step, agent
                            ] = decoded[
                                "workspace_physical_constraint_values"
                            ]
                        if "workspace_conservative_constraint_values" in decoded:
                            workspace_conservative_constraint_values[
                                step, agent
                            ] = decoded[
                                "workspace_conservative_constraint_values"
                            ]

            # Prefer dedicated workspace topics when present.  These are
            # particularly useful during INITIALIZE, where the follower runs
            # an absolute-position LeaderCoreRuntime.
            value = _nearest_vector(
                bag,
                workspace_relaxation_topics[robot],
                stamp,
                max_delta_ns,
                6,
            )
            if value is not None:
                workspace_relaxation[step, agent] = value

            value = _nearest_vector(
                bag,
                workspace_physical_topics[robot],
                stamp,
                max_delta_ns,
                6,
            )
            if value is not None:
                workspace_physical_constraint_values[step, agent] = value

            value = _nearest_vector(
                bag,
                workspace_conservative_topics[robot],
                stamp,
                max_delta_ns,
                6,
            )
            if value is not None:
                workspace_conservative_constraint_values[step, agent] = value

            scalar = _nearest_scalar(
                bag,
                required_slack_topics[robot],
                stamp,
                max_delta_ns,
            )
            if scalar is not None:
                required_slack[step, agent] = scalar

            scalar = _nearest_scalar(
                bag,
                thruster_utilization_topics[robot],
                stamp,
                max_delta_ns,
            )
            if scalar is not None:
                thruster_utilization[step, agent] = scalar

    explicit_targets: dict[str, list[float]] = {}
    for robot, target in args.initial_target:
        if robot not in robot_index:
            raise SystemExit(
                f"--initial-target refers to unknown robot {robot!r}; "
                f"known robots: {tuple(robots)}"
            )
        reference_position[:, robot_index[robot], :] = target
        explicit_targets[robot] = target.tolist()
        print(f"Using explicit initialization target for {robot}: {target.tolist()}")

    workspace_args = (
        args.workspace_physical_lower,
        args.workspace_physical_upper,
        args.workspace_conservative_lower,
        args.workspace_conservative_upper,
    )
    if any(value is not None for value in workspace_args):
        if not all(value is not None for value in workspace_args):
            raise SystemExit(
                "When supplying workspace fallback bounds, provide all four: "
                "physical lower/upper and conservative lower/upper."
            )
        physical_lower = np.asarray(args.workspace_physical_lower, dtype=float)
        physical_upper = np.asarray(args.workspace_physical_upper, dtype=float)
        conservative_lower = np.asarray(
            args.workspace_conservative_lower, dtype=float
        )
        conservative_upper = np.asarray(
            args.workspace_conservative_upper, dtype=float
        )
        if not (
            np.all(physical_lower < physical_upper)
            and np.all(conservative_lower < conservative_upper)
            and np.all(physical_lower <= conservative_lower)
            and np.all(conservative_upper <= physical_upper)
        ):
            raise SystemExit("Invalid workspace fallback bounds.")

        fallback_physical = _workspace_margins_from_bounds(
            positions, physical_lower, physical_upper
        )
        fallback_conservative = _workspace_margins_from_bounds(
            positions, conservative_lower, conservative_upper
        )
        missing = ~np.isfinite(workspace_physical_constraint_values)
        workspace_physical_constraint_values[missing] = fallback_physical[missing]
        missing = ~np.isfinite(workspace_conservative_constraint_values)
        workspace_conservative_constraint_values[missing] = (
            fallback_conservative[missing]
        )
        workspace_bounds_metadata = {
            "physical_lower": physical_lower.tolist(),
            "physical_upper": physical_upper.tolist(),
            "conservative_lower": conservative_lower.tolist(),
            "conservative_upper": conservative_upper.tolist(),
            "source": "explicit_export_fallback",
        }
        print("Filled missing workspace margins from explicit workspace bounds.")
    else:
        workspace_bounds_metadata = None

    if np.all(np.isfinite(reference_px4_times)) and np.all(
        np.diff(reference_px4_times) > 0.0
    ):
        times = reference_px4_times - reference_px4_times[0]
        time_source = "px4_timestamp"
    else:
        stamp_array = np.asarray(stamps, dtype=np.int64)
        times = (stamp_array - stamp_array[0]) * 1e-9
        time_source = "rosbag_receive_timestamp"

    metadata = {
        "schema_version": 3,
        "kind": "initialization_history",
        "robots": robots,
        "edges": list(manifest.get("edges", [])),
        "reference_robot": reference_robot,
        "source_run_dir": str(run_dir),
        "time_source": time_source,
        "max_sync_ms": float(args.max_sync_ms),
        "explicit_initialization_targets": explicit_targets,
        "workspace_bounds": workspace_bounds_metadata,
    }

    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else (run_dir / "initialization" / "initialization_history.npz").resolve()
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        times=np.asarray(times, dtype=float),
        bag_timestamps_ns=np.asarray(stamps, dtype=np.int64),
        positions=positions,
        quaternions=quaternions,
        linear_velocity_body=linear_velocity_body,
        angular_velocity_body=angular_velocity_body,
        px4_armed=px4_armed,
        px4_offboard_enabled=px4_offboard_enabled,
        reference_position=reference_position,
        desired_relative_position=desired_relative_position,
        required_slack=required_slack,
        thruster_utilization=thruster_utilization,
        workspace_relaxation=workspace_relaxation,
        workspace_physical_constraint_values=workspace_physical_constraint_values,
        workspace_conservative_constraint_values=(
            workspace_conservative_constraint_values
        ),
        metadata_json=np.asarray(json.dumps(metadata)),
    )

    print(f"Exported {n} synchronized initialization samples")
    print(f"Duration: {times[-1]:.3f} s")
    print(f"Time source: {time_source}")
    print(f"History: {output}")
    print(f"Absolute output folder: {output.parent}")


if __name__ == "__main__":
    main()
