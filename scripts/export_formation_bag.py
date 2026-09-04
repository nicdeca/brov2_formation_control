#!/usr/bin/env python3
"""Export a formation-control rosbag to a portable ROS-independent NPZ.

Run from the ROS environment, e.g.::

    source setup_ros2.sh
    python scripts/export_formation_bag.py outputs/experiments/<run> --phase mission

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



def _expand_topic_template(template: str, robot: str) -> str:
    return (
        str(template)
        .replace("{robot}", robot)
        .replace("{robot_lower}", robot.lower())
    )


def _state_configuration_from_manifest(
    manifest: dict[str, Any],
) -> tuple[str, str, str, str, str, str]:
    """Return selected source/topic/frame settings with legacy PX4 defaults."""
    state_source = str(manifest.get("state_source", "px4")).strip().lower()
    if state_source not in ("px4", "nav_msgs"):
        raise ValueError(
            "run manifest state_source must be 'px4' or 'nav_msgs', "
            f"got {state_source!r}"
        )

    state_topic_template = str(
        manifest.get("state_topic_template", "")
    ).strip()
    if not state_topic_template:
        state_topic_template = (
            "/{robot}/fmu/out/vehicle_odometry"
            if state_source == "px4"
            else "/mocap/{robot}/odom_ekf"
        )

    mocap_world_frame = str(
        manifest.get("mocap_world_frame", "core_nwu")
    ).strip()
    odom_twist_frame = str(
        manifest.get("odom_twist_frame", "body")
    ).strip()
    ekf_topic_template = str(
        manifest.get(
            "comparison_ekf_topic_template",
            "/mocap/{robot}/odom_ekf",
        )
    ).strip()
    mocap_core_pose_topic_template = str(
        manifest.get(
            "mocap_core_pose_topic_template",
            "/mocap/{robot}/pose_core",
        )
    ).strip()

    return (
        state_source,
        state_topic_template,
        mocap_world_frame,
        odom_twist_frame,
        ekf_topic_template,
        mocap_core_pose_topic_template,
    )


def nav_odometry_to_core(
    message,
    *,
    world_frame: str,
    twist_frame: str,
) -> dict[str, np.ndarray]:
    """Convert generic Odometry exactly through the online ROS adapter."""
    try:
        from formation_control_ros.frame_conventions import FrameConvention
        from formation_control_ros.state_adapter import odometry_to_core_state
    except ImportError as error:
        raise SystemExit(
            "formation_control_ros is unavailable. Source the ROS workspace "
            "before exporting a run that uses nav_msgs/Odometry."
        ) from error

    state = np.asarray(
        odometry_to_core_state(
            message,
            FrameConvention(
                world_frame=world_frame,
                twist_frame=twist_frame,
            ),
        ),
        dtype=float,
    )
    if state.shape != (13,):
        raise ValueError(
            "odometry_to_core_state returned unexpected state shape "
            f"{state.shape}; expected (13,)"
        )
    return {
        "position": state[0:3].copy(),
        "quaternion": state[3:7].copy(),
        "linear_velocity_body": state[7:10].copy(),
        "angular_velocity_body": state[10:13].copy(),
    }




def core_pose_message_to_core(message) -> dict[str, np.ndarray]:
    """Convert transformed core-NWU/FLU PoseStamped to canonical arrays."""
    position = np.array(
        [
            message.pose.position.x,
            message.pose.position.y,
            message.pose.position.z,
        ],
        dtype=float,
    )
    q_xyzw = np.array(
        [
            message.pose.orientation.x,
            message.pose.orientation.y,
            message.pose.orientation.z,
            message.pose.orientation.w,
        ],
        dtype=float,
    )
    norm = float(np.linalg.norm(q_xyzw))
    if (
        not np.all(np.isfinite(position))
        or not np.all(np.isfinite(q_xyzw))
        or norm <= 1e-12
    ):
        raise ValueError("invalid transformed raw MoCap pose")
    q_xyzw /= norm
    # Exported/core quaternion convention is scalar-first [w,x,y,z].
    quaternion = np.array(
        [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]],
        dtype=float,
    )
    if quaternion[0] < 0.0:
        quaternion = -quaternion
    return {
        "position": position,
        "quaternion": quaternion,
    }


def _rotation_log_vector(rotation: np.ndarray) -> np.ndarray:
    """SO(3) logarithm as a rotation vector."""
    r = np.asarray(rotation, dtype=float)
    cosine = float(np.clip((np.trace(r) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    vee = np.array(
        [
            r[2, 1] - r[1, 2],
            r[0, 2] - r[2, 0],
            r[1, 0] - r[0, 1],
        ],
        dtype=float,
    )
    if angle < 1e-8:
        return 0.5 * vee
    sine = float(np.sin(angle))
    if abs(sine) < 1e-8:
        # Rare near-pi case; eigenvector is more stable.
        values, vectors = np.linalg.eig(r)
        index = int(np.argmin(np.abs(values - 1.0)))
        axis = np.real(vectors[:, index])
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        return axis * angle
    return (0.5 * angle / sine) * vee


def finite_difference_mocap_twist(
    times: np.ndarray,
    positions: np.ndarray,
    quaternions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Diagnostic body-FLU twist from raw MoCap pose finite differences.

    This is intentionally labeled as a finite-difference diagnostic. It is not
    treated as a measured twist and is not used by the controller.
    """
    times = np.asarray(times, dtype=float)
    positions = np.asarray(positions, dtype=float)
    quaternions = np.asarray(quaternions, dtype=float)
    linear_body = np.full(positions.shape, np.nan)
    angular_body = np.full(positions.shape, np.nan)

    n_samples, n_agents, _ = positions.shape
    for agent in range(n_agents):
        for index in range(n_samples):
            left = max(0, index - 1)
            right = min(n_samples - 1, index + 1)
            if left == right:
                continue
            dt = float(times[right] - times[left])
            if not np.isfinite(dt) or dt <= 1e-6:
                continue

            required = (
                np.all(np.isfinite(positions[[left, right], agent]))
                and np.all(
                    np.isfinite(
                        quaternions[[left, index, right], agent]
                    )
                )
            )
            if not required:
                continue

            q_current = quaternions[index, agent]
            r_current = quaternion_to_matrix(q_current)
            velocity_world = (
                positions[right, agent] - positions[left, agent]
            ) / dt
            linear_body[index, agent] = (
                r_current.T @ velocity_world
            )

            r_left = quaternion_to_matrix(
                quaternions[left, agent]
            )
            r_right = quaternion_to_matrix(
                quaternions[right, agent]
            )
            # World-frame increment, then express omega in current FLU body.
            rotation_increment = r_right @ r_left.T
            omega_world = (
                _rotation_log_vector(rotation_increment) / dt
            )
            angular_body[index, agent] = (
                r_current.T @ omega_world
            )

    return linear_body, angular_body

def _convert_selected_state(
    message,
    *,
    state_source: str,
    world_frame: str,
    twist_frame: str,
) -> dict[str, np.ndarray]:
    if state_source == "px4":
        return px4_odometry_to_core(message)
    return nav_odometry_to_core(
        message,
        world_frame=world_frame,
        twist_frame=twist_frame,
    )


def _fill_state_arrays(
    state: dict[str, np.ndarray],
    *,
    step: int,
    agent: int,
    positions: np.ndarray,
    quaternions: np.ndarray,
    linear_velocity_body: np.ndarray,
    angular_velocity_body: np.ndarray,
) -> None:
    positions[step, agent] = state["position"]
    quaternions[step, agent] = state["quaternion"]
    linear_velocity_body[step, agent] = state["linear_velocity_body"]
    angular_velocity_body[step, agent] = state["angular_velocity_body"]


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
        "--phase",
        choices=("initialization", "mission"),
        default=None,
        help=(
            "phase to export for the split recording layout; when omitted, "
            "legacy <run_dir>/bag is used if present, otherwise mission"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "output NPZ path; default is <run_dir>/<phase>/formation_history.npz "
            "for split recordings or <run_dir>/formation_history.npz for legacy runs"
        ),
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
    (
        state_source,
        state_topic_template,
        mocap_world_frame,
        odom_twist_frame,
        ekf_topic_template,
        mocap_core_pose_topic_template,
    ) = _state_configuration_from_manifest(manifest)

    if args.phase is not None:
        phase_name = args.phase
        phase_dir = run_dir / phase_name
    elif (run_dir / "bag").exists():
        phase_name = "legacy"
        phase_dir = run_dir
    elif (run_dir / "mission" / "bag").exists():
        phase_name = "mission"
        phase_dir = run_dir / "mission"
    else:
        raise FileNotFoundError(
            "No rosbag found. Expected <run_dir>/bag or "
            "<run_dir>/<initialization|mission>/bag."
        )

    bag_dir = phase_dir / "bag"
    if not bag_dir.exists():
        raise FileNotFoundError(f"missing rosbag directory: {bag_dir}")
    bag = read_bag(bag_dir)

    diagnostic_topics = {
        robot: f"/{robot}/formation_control/diagnostic_snapshot" for robot in robots
    }
    selected_odometry_topics = {
        robot: _expand_topic_template(state_topic_template, robot)
        for robot in robots
    }
    px4_odometry_topics = {
        robot: f"/{robot}/fmu/out/vehicle_odometry" for robot in robots
    }
    ekf_odometry_topics = {
        robot: _expand_topic_template(ekf_topic_template, robot)
        for robot in robots
    }
    mocap_core_pose_topics = {
        robot: _expand_topic_template(
            mocap_core_pose_topic_template,
            robot,
        )
        for robot in robots
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
    required_diag_robots = set(robots)

    valid_bag_stamps: list[int] = []
    for bag_stamp, _ in reference_diagnostics.records:
        if any(
            selected_odometry_topics[robot] not in bag
            or bag[selected_odometry_topics[robot]].nearest(
                bag_stamp, max_delta_ns
            ) is None
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

    # Parallel estimator histories on the same controller synchronization grid.
    # Standard plots use the canonical arrays above, which correspond to the
    # state source selected in the run manifest.
    px4_positions = np.full((n_samples, n_agents, 3), np.nan)
    px4_quaternions = np.full((n_samples, n_agents, 4), np.nan)
    px4_linear_velocity_body = np.full((n_samples, n_agents, 3), np.nan)
    px4_angular_velocity_body = np.full((n_samples, n_agents, 3), np.nan)

    ekf_positions = np.full((n_samples, n_agents, 3), np.nan)
    ekf_quaternions = np.full((n_samples, n_agents, 4), np.nan)
    ekf_linear_velocity_body = np.full((n_samples, n_agents, 3), np.nan)
    ekf_angular_velocity_body = np.full((n_samples, n_agents, 3), np.nan)

    # Raw MoCap after the estimator's explicit input-frame transform. These
    # arrays are direct pose measurements in core NWU / FLU.
    mocap_positions = np.full((n_samples, n_agents, 3), np.nan)
    mocap_quaternions = np.full((n_samples, n_agents, 4), np.nan)

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
            selected_msg = bag[selected_odometry_topics[robot]].nearest(
                bag_stamp,
                max_delta_ns,
            )
            assert selected_msg is not None
            selected_state = _convert_selected_state(
                selected_msg,
                state_source=state_source,
                world_frame=mocap_world_frame,
                twist_frame=odom_twist_frame,
            )
            _fill_state_arrays(
                selected_state,
                step=step,
                agent=agent,
                positions=positions,
                quaternions=quaternions,
                linear_velocity_body=linear_velocity_body,
                angular_velocity_body=angular_velocity_body,
            )

            px4_series = bag.get(px4_odometry_topics[robot])
            if px4_series is not None:
                px4_msg = px4_series.nearest(bag_stamp, max_delta_ns)
                if px4_msg is not None:
                    px4_state = px4_odometry_to_core(px4_msg)
                    _fill_state_arrays(
                        px4_state,
                        step=step,
                        agent=agent,
                        positions=px4_positions,
                        quaternions=px4_quaternions,
                        linear_velocity_body=px4_linear_velocity_body,
                        angular_velocity_body=px4_angular_velocity_body,
                    )
                    if robot == reference_robot and state_source == "px4":
                        px4_time = px4_time_seconds(px4_msg)
                        if px4_time is not None:
                            reference_px4_times[step] = px4_time

            ekf_series = bag.get(ekf_odometry_topics[robot])
            if ekf_series is not None:
                ekf_msg = ekf_series.nearest(bag_stamp, max_delta_ns)
                if ekf_msg is not None:
                    ekf_state = nav_odometry_to_core(
                        ekf_msg,
                        world_frame=mocap_world_frame,
                        twist_frame=odom_twist_frame,
                    )
                    _fill_state_arrays(
                        ekf_state,
                        step=step,
                        agent=agent,
                        positions=ekf_positions,
                        quaternions=ekf_quaternions,
                        linear_velocity_body=ekf_linear_velocity_body,
                        angular_velocity_body=ekf_angular_velocity_body,
                    )

            mocap_series = bag.get(
                mocap_core_pose_topics[robot]
            )
            if mocap_series is not None:
                mocap_msg = mocap_series.nearest(
                    bag_stamp,
                    max_delta_ns,
                )
                if mocap_msg is not None:
                    try:
                        mocap_state = core_pose_message_to_core(
                            mocap_msg
                        )
                    except ValueError:
                        pass
                    else:
                        mocap_positions[step, agent] = (
                            mocap_state["position"]
                        )
                        mocap_quaternions[step, agent] = (
                            mocap_state["quaternion"]
                        )

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

    # A synchronized diagnostic message can still correspond to an invalid
    # controller sample (for example, a one-tick fallback with NaN thruster
    # forces). FormationTrajectory requires finite controls for every robot.
    # Drop such timestamps rather than fabricating zero commands.
    finite_control_mask = np.all(
        np.isfinite(snapshots["thruster_forces"]),
        axis=(1, 2),
    )

    n_invalid_controls = int(np.count_nonzero(~finite_control_mask))
    if n_invalid_controls:
        print(
            f"Dropping {n_invalid_controls} synchronized sample(s) with "
            "non-finite thruster forces."
        )

    valid_bag_stamps = [
        stamp
        for stamp, keep in zip(valid_bag_stamps, finite_control_mask)
        if keep
    ]

    positions = positions[finite_control_mask]
    quaternions = quaternions[finite_control_mask]
    linear_velocity_body = linear_velocity_body[finite_control_mask]
    angular_velocity_body = angular_velocity_body[finite_control_mask]

    px4_positions = px4_positions[finite_control_mask]
    px4_quaternions = px4_quaternions[finite_control_mask]
    px4_linear_velocity_body = px4_linear_velocity_body[finite_control_mask]
    px4_angular_velocity_body = px4_angular_velocity_body[finite_control_mask]
    ekf_positions = ekf_positions[finite_control_mask]
    ekf_quaternions = ekf_quaternions[finite_control_mask]
    ekf_linear_velocity_body = ekf_linear_velocity_body[finite_control_mask]
    ekf_angular_velocity_body = ekf_angular_velocity_body[finite_control_mask]
    mocap_positions = mocap_positions[finite_control_mask]
    mocap_quaternions = mocap_quaternions[finite_control_mask]

    px4_thrust_setpoint = px4_thrust_setpoint[finite_control_mask]
    px4_torque_setpoint = px4_torque_setpoint[finite_control_mask]
    px4_armed = px4_armed[finite_control_mask]
    px4_offboard_enabled = px4_offboard_enabled[finite_control_mask]

    reference_px4_times = reference_px4_times[finite_control_mask]

    for field in snapshots:
        snapshots[field] = snapshots[field][finite_control_mask]

    n_samples = int(np.count_nonzero(finite_control_mask))
    if n_samples < 2:
        raise SystemExit(
            "Too few samples remain after rejecting non-finite controller outputs."
        )

    # Prefer PX4's simulation/sample clock. Fall back to rosbag receive time if
    # PX4 timestamps are unavailable or non-monotone.
    if (
        state_source == "px4"
        and np.all(np.isfinite(reference_px4_times))
        and np.all(np.diff(reference_px4_times) > 0.0)
    ):
        times = reference_px4_times - reference_px4_times[0]
        time_source = "px4_timestamp"
    else:
        stamps = np.asarray(valid_bag_stamps, dtype=np.int64)
        times = (stamps - stamps[0]) * 1e-9
        time_source = "rosbag_receive_timestamp"

    mocap_linear_velocity_body, mocap_angular_velocity_body = (
        finite_difference_mocap_twist(
            np.asarray(times, dtype=float),
            mocap_positions,
            mocap_quaternions,
        )
    )

    arrays = {
        "times": np.asarray(times, dtype=float),
        "bag_timestamps_ns": np.asarray(valid_bag_stamps, dtype=np.int64),
        "positions": positions,
        "quaternions": quaternions,
        "linear_velocity_body": linear_velocity_body,
        "angular_velocity_body": angular_velocity_body,
        "px4_positions": px4_positions,
        "px4_quaternions": px4_quaternions,
        "px4_linear_velocity_body": px4_linear_velocity_body,
        "px4_angular_velocity_body": px4_angular_velocity_body,
        "ekf_positions": ekf_positions,
        "ekf_quaternions": ekf_quaternions,
        "ekf_linear_velocity_body": ekf_linear_velocity_body,
        "ekf_angular_velocity_body": ekf_angular_velocity_body,
        "mocap_positions": mocap_positions,
        "mocap_quaternions": mocap_quaternions,
        "mocap_linear_velocity_body_fd": mocap_linear_velocity_body,
        "mocap_angular_velocity_body_fd": mocap_angular_velocity_body,
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
        "recording_phase": phase_name,
        "max_sync_ms": float(args.max_sync_ms),
        "time_source": time_source,
        "state_source": state_source,
        "state_topic_template": state_topic_template,
        "mocap_world_frame": mocap_world_frame,
        "odom_twist_frame": odom_twist_frame,
        "comparison_ekf_topic_template": ekf_topic_template,
        "mocap_core_pose_topic_template": (
            mocap_core_pose_topic_template
        ),
        "mocap_twist_source": "finite_difference_of_core_pose",
    }

    output = (args.output or phase_dir / "formation_history.npz").expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    FormationExperimentHistory(arrays=arrays, metadata=metadata).save(output)

    print(f"Exported {n_samples} synchronized controller samples")
    print(f"Recording phase: {phase_name}")
    print(f"Duration: {times[-1]:.3f} s")
    print(f"Time source: {time_source}")
    print(f"Controller state source: {state_source}")
    print(f"Controller state topic: {state_topic_template}")
    print(f"History: {output}")
    print(f"Absolute output folder: {output.parent}")


if __name__ == "__main__":
    main()
