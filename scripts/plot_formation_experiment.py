#!/usr/bin/env python3
"""Plot a ROS/SITL/hardware formation run with the existing BlueROV2 plotters.

The rosbag exporter writes a ROS-independent ``formation_history.npz``.  This
script adapts that history back to ``FormationScenario``, ``FormationTrajectory``
and ``CLFDiagnosticHistory`` and then calls the same plotting functions used by
``examples/04_bluerov2_fov_clf_qp.py``.

Typical usage::

    uv run python scripts/plot_formation_experiment.py RUN/formation_history.npz \
        --paper --paper-quality --save

    uv run python scripts/plot_formation_experiment.py RUN/formation_history.npz \
        --trajectory --leader-tracking --distance --fov --thrusters \
        --controller-time --show

    uv run python scripts/plot_formation_experiment.py RUN/formation_history.npz \
        --all --save
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType

import matplotlib.pyplot as plt
import numpy as np

from formation_control.actuation import (
    BlueROV2HeavyThrusterAllocation,
    BlueROV2HeavyThrusterConfiguration,
    T200ForceLimits,
)
from formation_control.constraints import DistanceDomain, FieldOfViewDomain
from formation_control.experiment import FormationExperimentHistory
from formation_control.geometry import rotation_matrix_from_quaternion
from formation_control.graphs import DirectedSensingGraph
from formation_control.simulation import (
    FormationReference,
    FormationScenario,
    FormationTrajectory,
)
from formation_control.visualization import (
    BlueROV2HeavyVisualGeometry,
    apply_visualization_style,
    plot_formation_3d,
    save_figure,
)
from formation_control.visualization.domain_relaxation_plot import (
    plot_vertical_fov_relaxation_from_histories,
)


PAPER_PLOTS = {
    "trajectory",
    "leader_tracking",
    "distance",
    "fov",
    "adaptive_fov",
    "slack",
    "actuation",
    "domain",
    "thrusters",
    "controller_time",
    "clf_value",
}

ALL_PLOTS = PAPER_PLOTS | {
    "relaxation_rates",
    "clf_balance",
    "clf_drift",
    "backstepping",
    "peak_debug",
}


def _load_legacy_module(repo_root: Path) -> ModuleType:
    """Load example 04 as a module so its exact plotters are reused."""
    path = repo_root / "examples" / "04_bluerov2_fov_clf_qp.py"
    if not path.exists():
        raise FileNotFoundError(
            "Cannot find the existing BlueROV2 diagnostic plotters at "
            f"{path}. Pass --repo-root if needed."
        )

    module_name = "_formation_control_legacy_bluerov2_example"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses inspect sys.modules while the module is executing.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _first_finite_vector(array: np.ndarray, agent: int) -> np.ndarray:
    values = np.asarray(array[:, agent], dtype=float)
    if values.ndim != 2:
        raise ValueError("expected a vector-valued agent history")
    mask = np.all(np.isfinite(values), axis=1)
    if not np.any(mask):
        raise ValueError("required logged vector is entirely NaN")
    return values[np.flatnonzero(mask)[0]].copy()


def _first_finite_scalar(array: np.ndarray, agent: int) -> float:
    values = np.asarray(array[:, agent], dtype=float).reshape(-1)
    mask = np.isfinite(values)
    if not np.any(mask):
        raise ValueError("required logged scalar is entirely NaN")
    return float(values[np.flatnonzero(mask)[0]])


def _graph_from_metadata(
    metadata: dict,
    robots: list[str],
) -> tuple[DirectedSensingGraph, dict[str, int]]:
    index = {robot: agent for agent, robot in enumerate(robots)}
    adjacency = np.zeros((len(robots), len(robots)), dtype=int)
    for edge in metadata.get("edges", []):
        observer = str(edge["observer"])
        target = str(edge["target"])
        if observer not in index or target not in index:
            raise ValueError(
                f"manifest edge {observer}->{target} uses an unknown robot"
            )
        adjacency[index[observer], index[target]] = 1
    return DirectedSensingGraph.from_adjacency(adjacency), index


def _reference_offsets(
    desired_history: np.ndarray,
    graph: DirectedSensingGraph,
) -> np.ndarray:
    """Recover formation offsets from desired parent-minus-follower vectors."""
    n_agents = desired_history.shape[1]
    offsets = np.full((n_agents, 3), np.nan)
    offsets[graph.root] = 0.0
    unresolved = set(range(n_agents)) - {graph.root}

    while unresolved:
        progress = False
        for edge in graph:
            observer = edge.observer
            target = edge.target
            if observer not in unresolved or not np.all(np.isfinite(offsets[target])):
                continue

            values = desired_history[:, observer]
            finite = np.all(np.isfinite(values), axis=1)
            if not np.any(finite):
                continue
            desired_relative = values[np.flatnonzero(finite)[0]]
            # d_ij^d = p_target^d - p_observer^d.
            offsets[observer] = offsets[target] - desired_relative
            unresolved.remove(observer)
            progress = True

        if not progress:
            raise ValueError(
                "Could not reconstruct the formation reference. Every follower "
                "must log desired_relative_position at least once."
            )

    return offsets


def _world_linear_velocity(
    quaternions: np.ndarray,
    body_velocity: np.ndarray,
) -> np.ndarray:
    result = np.full_like(body_velocity, np.nan)
    for step in range(quaternions.shape[0]):
        for agent in range(quaternions.shape[1]):
            q = quaternions[step, agent]
            v_body = body_velocity[step, agent]
            if not (np.all(np.isfinite(q)) and np.all(np.isfinite(v_body))):
                continue
            result[step, agent] = rotation_matrix_from_quaternion(q) @ v_body
    return result


def _make_allocation(history: FormationExperimentHistory, follower: int):
    """Rebuild the exact logged T200 force bounds with standard Heavy geometry."""
    limits = _first_finite_vector(
        history.arrays["thruster_force_limits"],
        follower,
    )
    reverse, forward = (float(value) for value in limits)
    base = BlueROV2HeavyThrusterConfiguration.default_45deg()
    configuration = BlueROV2HeavyThrusterConfiguration(
        positions_body=base.positions_body,
        directions_body=base.directions_body,
        force_limits=T200ForceLimits(forward=forward, reverse=reverse),
        names=base.names,
    )
    return BlueROV2HeavyThrusterAllocation(configuration)


def _make_camera(history: FormationExperimentHistory, follower: int):
    """Reconstruct the camera if the installed core exposes CameraExtrinsics."""
    half_angles = _first_finite_vector(
        history.arrays["camera_half_angles"],
        follower,
    )
    position = _first_finite_vector(
        history.arrays["camera_position_body"],
        follower,
    )
    rotation = _first_finite_vector(
        history.arrays["camera_rotation_camera_from_body"],
        follower,
    ).reshape(3, 3)

    try:
        from formation_control.geometry import CameraExtrinsics, PinholeCamera

        extrinsics = CameraExtrinsics(
            rotation_camera_from_body=rotation,
            position_camera_in_body=position,
        )
        return PinholeCamera.from_degrees(
            horizontal_half_angle=float(np.rad2deg(half_angles[0])),
            vertical_half_angle=float(np.rad2deg(half_angles[1])),
            extrinsics=extrinsics,
        )
    except (ImportError, TypeError):
        # Camera visualization is optional; FoV diagnostics use the logged
        # normalized coordinates and remain exact even if this API changed.
        return None


def _extend_state_history(array: np.ndarray) -> np.ndarray:
    return np.concatenate((array, array[-1:]), axis=0)


def _build_legacy_data(
    history: FormationExperimentHistory,
    legacy: ModuleType,
) -> dict[str, object]:
    arrays = history.arrays
    robots = [str(robot) for robot in history.metadata["robots"]]
    graph, robot_index = _graph_from_metadata(history.metadata, robots)
    edges = tuple(graph.edges)
    if not edges:
        raise ValueError("plotting requires at least one follower edge")
    first_follower = edges[0].observer

    offsets = _reference_offsets(arrays["desired_relative_position"], graph)
    scenario = FormationScenario(
        graph=graph,
        reference=FormationReference(offsets=offsets, root=graph.root),
        initial_positions=np.asarray(arrays["positions"][0], dtype=float),
    )

    control_times = np.asarray(arrays["times"], dtype=float)
    if control_times.size < 2:
        raise ValueError("at least two controller samples are required")
    dt = float(np.median(np.diff(control_times)))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("logged controller timestamps are not strictly increasing")

    trajectory_times = np.concatenate(
        (control_times, np.array([control_times[-1] + dt]))
    )
    positions = _extend_state_history(np.asarray(arrays["positions"], dtype=float))
    quaternions = _extend_state_history(
        np.asarray(arrays["quaternions"], dtype=float)
    )
    velocity_world = _world_linear_velocity(
        np.asarray(arrays["quaternions"], dtype=float),
        np.asarray(arrays["linear_velocity_body"], dtype=float),
    )
    velocity_world = _extend_state_history(velocity_world)
    controls = np.asarray(arrays["thruster_forces"], dtype=float)

    trajectory = FormationTrajectory(
        times=trajectory_times,
        positions=positions,
        velocities=velocity_world,
        controls=controls,
        quaternions=quaternions,
    )

    distance_values = _first_finite_vector(
        arrays["distance_domain"],
        first_follower,
    )
    fov_values = _first_finite_vector(
        arrays["fov_domain"],
        first_follower,
    )
    distance_domain = DistanceDomain(
        d_min=float(distance_values[0]),
        d_max=float(distance_values[1]),
        d_min_conservative=float(distance_values[2]),
        d_max_conservative=float(distance_values[3]),
    )
    fov_domain = FieldOfViewDomain(
        alpha_h_conservative=float(fov_values[0]),
        alpha_v_conservative=float(fov_values[1]),
    )

    maxima = np.array(
        [
            distance_domain.collision_enlargement_max,
            distance_domain.range_enlargement_max,
            fov_domain.horizontal_enlargement_max,
            fov_domain.vertical_enlargement_max,
        ],
        dtype=float,
    )
    s_history = np.nan_to_num(
        np.asarray(arrays["relaxation_state"], dtype=float),
        nan=0.0,
    )
    rho_history = _extend_state_history(s_history * maxima)
    image_history = _extend_state_history(
        np.asarray(arrays["image_coordinates"], dtype=float)
    )
    s_history_extended = _extend_state_history(s_history)

    constraints = _first_finite_vector(
        arrays["constraints_enabled"],
        first_follower,
    ) > 0.5
    adaptive = _first_finite_scalar(
        arrays["adaptive_enabled"],
        first_follower,
    ) > 0.5
    domain_margin_ratio = _first_finite_scalar(
        arrays["domain_margin_ratio"],
        first_follower,
    )

    clf_kwargs = {
        "value": np.asarray(arrays["clf_value"], dtype=float),
        "decay": np.asarray(arrays["clf_decay"], dtype=float),
        "drift": np.asarray(arrays["clf_drift"], dtype=float),
        "configuration_local_rate": np.asarray(
            arrays["clf_configuration_local_rate"], dtype=float
        ),
        "parent_rate": np.asarray(arrays["clf_parent_rate"], dtype=float),
        "velocity_backstepping_rate": np.asarray(
            arrays["clf_velocity_backstepping_rate"], dtype=float
        ),
        "dynamics_bias_rate": np.asarray(
            arrays["clf_dynamics_bias_rate"], dtype=float
        ),
        "dynamics_bias_linear_rate": np.asarray(
            arrays["clf_dynamics_bias_linear_rate"], dtype=float
        ),
        "dynamics_bias_angular_rate": np.asarray(
            arrays["clf_dynamics_bias_angular_rate"], dtype=float
        ),
        "command_acceleration_rate": np.asarray(
            arrays["clf_command_acceleration_rate"], dtype=float
        ),
        "command_acceleration_linear_rate": np.asarray(
            arrays["clf_command_acceleration_linear_rate"], dtype=float
        ),
        "command_acceleration_angular_rate": np.asarray(
            arrays["clf_command_acceleration_angular_rate"], dtype=float
        ),
        "velocity_error_norm": np.asarray(
            arrays["clf_velocity_error_norm"], dtype=float
        ),
        "velocity_error_linear_norm": np.asarray(
            arrays["clf_velocity_error_linear_norm"], dtype=float
        ),
        "velocity_error_angular_norm": np.asarray(
            arrays["clf_velocity_error_angular_norm"], dtype=float
        ),
        "filtered_velocity_derivative_norm": np.asarray(
            arrays["clf_filtered_velocity_derivative_norm"], dtype=float
        ),
        "filtered_linear_acceleration_norm": np.asarray(
            arrays["clf_filtered_linear_acceleration_norm"], dtype=float
        ),
        "filtered_angular_acceleration_norm": np.asarray(
            arrays["clf_filtered_angular_acceleration_norm"], dtype=float
        ),
        "generalized_velocity": np.asarray(
            arrays["generalized_velocity"], dtype=float
        ),
        "filtered_velocity": np.asarray(
            arrays["filtered_velocity"], dtype=float
        ),
        "desired_velocity": np.asarray(
            arrays["desired_velocity"], dtype=float
        ),
        "filtered_velocity_derivative": np.asarray(
            arrays["filtered_velocity_derivative"], dtype=float
        ),
        "dynamics_bias": np.asarray(arrays["dynamics_bias"], dtype=float),
        "best_actuator_contribution": np.asarray(
            arrays["clf_best_actuator_contribution"], dtype=float
        ),
        "minimum_modeled_derivative": np.asarray(
            arrays["clf_minimum_modeled_derivative"], dtype=float
        ),
        "hard_clf_residual": np.asarray(
            arrays["clf_hard_residual"], dtype=float
        ),
    }
    # The current pure-Python example also stores the pre-saturation virtual
    # command, but schema v1 intentionally does not because no existing ROS
    # plot requires it.  Supply explicit NaNs only when that legacy dataclass
    # version requires the field.
    if "unlimited_desired_velocity" in inspect.signature(
        legacy.CLFDiagnosticHistory
    ).parameters:
        clf_kwargs["unlimited_desired_velocity"] = np.full_like(
            np.asarray(arrays["desired_velocity"], dtype=float),
            np.nan,
        )

    clf = legacy.CLFDiagnosticHistory(**clf_kwargs)

    allocation = _make_allocation(history, first_follower)
    camera = _make_camera(history, first_follower)

    return {
        "arrays": arrays,
        "robots": robots,
        "robot_index": robot_index,
        "graph": graph,
        "first_follower": first_follower,
        "scenario": scenario,
        "trajectory": trajectory,
        "distance_domain": distance_domain,
        "fov_domain": fov_domain,
        "rho_history": rho_history,
        "s_history": s_history_extended,
        "relaxation_rate_history": np.asarray(
            arrays["relaxation_rate"], dtype=float
        ),
        "image_history": image_history,
        "slacks": np.asarray(arrays["slack"], dtype=float),
        "required_slacks": np.asarray(arrays["required_slack"], dtype=float),
        "actuation_margins": np.asarray(arrays["actuation_margin"], dtype=float),
        "controller_times": np.asarray(arrays["controller_time_s"], dtype=float),
        "clf": clf,
        "allocation": allocation,
        "camera": camera,
        "constraints_enabled": constraints,
        "adaptive": adaptive,
        "domain_margin_ratio": domain_margin_ratio,
    }


def _selected_plots(args: argparse.Namespace) -> set[str]:
    selected = {
        name for name in ALL_PLOTS if bool(getattr(args, name, False))
    }
    if args.paper:
        selected |= PAPER_PLOTS
    if args.all:
        selected = set(ALL_PLOTS)
    if not selected:
        selected = set(PAPER_PLOTS)
    return selected


def _desired_positions(data: dict[str, object]) -> np.ndarray:
    arrays = data["arrays"]
    scenario = data["scenario"]
    graph = data["graph"]
    root = graph.root
    reference = np.asarray(arrays["reference_position"][:, root], dtype=float)
    finite = np.all(np.isfinite(reference), axis=1)
    if np.any(finite):
        root_position = reference[np.flatnonzero(finite)[-1]]
    else:
        root_position = scenario.initial_positions[root]
    return root_position + scenario.reference.offsets


def _trajectory_figure(data: dict[str, object], paper_quality: bool):
    scenario = data["scenario"]
    trajectory = data["trajectory"]
    camera = data["camera"]
    distance_domain = data["distance_domain"]
    fov_domain = data["fov_domain"]
    image_history = data["image_history"]
    constraints = data["constraints_enabled"]
    legacy = data["legacy"]

    kwargs = {
        "desired_positions": _desired_positions(data),
        "title": "Formation-control experiment",
        "paper_quality": paper_quality,
        "vehicle_geometry": BlueROV2HeavyVisualGeometry().wireframe(),
        "show_body_forward": False,
    }

    try:
        edge_quality = legacy.connection_quality_history(
            scenario,
            trajectory,
            distance_domain,
            fov_domain,
            image_history,
            distance_constraints=bool(constraints[0] or constraints[1]),
            fov_constraints=bool(constraints[2] or constraints[3]),
        )
        kwargs["edge_quality"] = edge_quality
    except (AttributeError, ValueError):
        pass

    if camera is not None:
        kwargs["camera"] = camera
        kwargs["camera_agents"] = tuple(
            agent for agent in range(scenario.n_agents) if agent != scenario.graph.root
        )
        kwargs["camera_depth"] = 0.75

    # Keep compatibility if plot_formation_3d's optional keyword surface evolves.
    signature = inspect.signature(plot_formation_3d)
    kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    result = plot_formation_3d(
        trajectory,
        scenario.graph,
        **kwargs,
    )
    return result[0] if isinstance(result, tuple) else result


def _leader_tracking_figure(data: dict[str, object], paper_quality: bool):
    arrays = data["arrays"]
    graph = data["graph"]
    times = np.asarray(arrays["times"], dtype=float)
    root = graph.root

    p = np.asarray(arrays["positions"][:, root], dtype=float)
    p_ref = np.asarray(arrays["reference_position"][:, root], dtype=float)
    v_ref = np.asarray(arrays["reference_velocity"][:, root], dtype=float)
    v_world = _world_linear_velocity(
        np.asarray(arrays["quaternions"][:, root : root + 1], dtype=float),
        np.asarray(arrays["linear_velocity_body"][:, root : root + 1], dtype=float),
    )[:, 0]

    valid_p = np.all(np.isfinite(p_ref), axis=1)
    valid_v = np.all(np.isfinite(v_ref), axis=1)
    if np.count_nonzero(valid_p) < 2:
        return None

    apply_visualization_style(paper_quality=paper_quality)
    figure, (ax_p, ax_v) = plt.subplots(2, 1, sharex=True)
    ax_p.plot(times[valid_p], np.linalg.norm(p[valid_p] - p_ref[valid_p], axis=1))
    ax_p.set_ylabel(r"$\|p_0-p_r\|$ [m]")
    ax_p.set_title("Leader tracking error")
    ax_p.grid(True, alpha=0.3)

    if np.count_nonzero(valid_v) >= 2:
        ax_v.plot(
            times[valid_v],
            np.linalg.norm(v_world[valid_v] - v_ref[valid_v], axis=1),
        )
    ax_v.set_xlabel(r"$t$ [s]")
    ax_v.set_ylabel(r"$\|v_0-v_r\|$ [m/s]")
    ax_v.grid(True, alpha=0.3)
    figure.tight_layout()
    return figure


def _print_summary(data: dict[str, object]) -> None:
    arrays = data["arrays"]
    graph = data["graph"]
    robots = data["robots"]
    followers = [edge.observer for edge in graph]
    times = np.asarray(arrays["times"], dtype=float)

    print(f"Run duration: {times[-1]:.3f} s")
    print(f"Robots: {', '.join(robots)}")
    print(
        "Edges: "
        + ", ".join(
            f"{robots[edge.observer]}->{robots[edge.target]}" for edge in graph
        )
    )

    if followers:
        controller = np.asarray(arrays["controller_time_s"][:, followers], dtype=float)
        finite = controller[np.isfinite(controller)]
        if finite.size:
            print(
                "Controller time: "
                f"mean {1e3 * np.mean(finite):.3f} ms, "
                f"max {1e3 * np.max(finite):.3f} ms"
            )

        utilization = np.asarray(
            arrays["thruster_utilization"][:, followers], dtype=float
        )
        finite = utilization[np.isfinite(utilization)]
        if finite.size:
            print(f"Maximum thruster utilization: {np.max(finite):.3f}")

        required = np.asarray(arrays["required_slack"][:, followers], dtype=float)
        finite = required[np.isfinite(required)]
        if finite.size:
            print(
                "Actuation feasibility: "
                f"max required slack {np.max(finite):.4g}, "
                f"infeasible samples {100.0 * np.mean(finite > 1e-10):.2f}%"
            )

        margin = np.asarray(arrays["actuation_margin"][:, followers], dtype=float)
        finite = margin[np.isfinite(margin)]
        if finite.size:
            print(f"Minimum zero-slack actuation margin: {np.min(finite):.4g}")

        physical = np.asarray(
            arrays["minimum_physical_margin"][:, followers], dtype=float
        )
        finite = physical[np.isfinite(physical)]
        if finite.size:
            print(f"Minimum logged physical sensing margin: {np.min(finite):.4g}")

        s = np.asarray(arrays["relaxation_state"][:, followers], dtype=float)
        finite = s[np.isfinite(s)]
        if finite.size:
            print(f"Maximum normalized domain relaxation: {np.max(finite):.3f}")

        fallback = np.asarray(arrays["fallback"][:, followers], dtype=float)
        finite = fallback[np.isfinite(fallback)]
        if finite.size:
            print(f"Fallback samples: {100.0 * np.mean(finite > 0.5):.2f}%")

    root = graph.root
    p_ref = np.asarray(arrays["reference_position"][:, root], dtype=float)
    v_ref = np.asarray(arrays["reference_velocity"][:, root], dtype=float)
    p = np.asarray(arrays["positions"][:, root], dtype=float)
    v = _world_linear_velocity(
        np.asarray(arrays["quaternions"][:, root : root + 1], dtype=float),
        np.asarray(arrays["linear_velocity_body"][:, root : root + 1], dtype=float),
    )[:, 0]
    valid_p = np.all(np.isfinite(p_ref), axis=1)
    valid_v = np.all(np.isfinite(v_ref), axis=1)
    if np.any(valid_p):
        error = np.linalg.norm(p[valid_p] - p_ref[valid_p], axis=1)
        print(
            "Leader position tracking: "
            f"RMS {np.sqrt(np.mean(error**2)):.4f} m, "
            f"max {np.max(error):.4f} m"
        )
    if np.any(valid_v):
        error = np.linalg.norm(v[valid_v] - v_ref[valid_v], axis=1)
        print(
            "Leader velocity tracking: "
            f"RMS {np.sqrt(np.mean(error**2)):.4f} m/s"
        )


def _save_figures(
    figures: dict[str, plt.Figure | None],
    output_dir: Path,
    figure_format: str,
    paper_quality: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for name, figure in figures.items():
        if figure is None:
            continue
        save_figure(
            figure,
            output_dir / f"{name}.{figure_format}",
            paper_quality=paper_quality,
        )
        count += 1
    print(f"Saved {count} figure(s) to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root containing examples/04_bluerov2_fov_clf_qp.py",
    )
    parser.add_argument(
        "--observer",
        default=None,
        help="robot name used for the representative adaptive-FoV paper plot",
    )
    parser.add_argument("--paper", action="store_true", help="paper-oriented preset")
    parser.add_argument("--all", action="store_true", help="all available diagnostics")
    parser.add_argument("--trajectory", action="store_true")
    parser.add_argument("--leader-tracking", dest="leader_tracking", action="store_true")
    parser.add_argument("--distance", action="store_true")
    parser.add_argument("--fov", action="store_true")
    parser.add_argument("--adaptive-fov", dest="adaptive_fov", action="store_true")
    parser.add_argument("--slack", action="store_true")
    parser.add_argument("--actuation", action="store_true")
    parser.add_argument("--domain", action="store_true")
    parser.add_argument(
        "--relaxation-rates",
        dest="relaxation_rates",
        action="store_true",
    )
    parser.add_argument("--thrusters", action="store_true")
    parser.add_argument(
        "--controller-time",
        dest="controller_time",
        action="store_true",
    )
    parser.add_argument("--clf-value", dest="clf_value", action="store_true")
    parser.add_argument("--clf-balance", dest="clf_balance", action="store_true")
    parser.add_argument("--clf-drift", dest="clf_drift", action="store_true")
    parser.add_argument("--backstepping", action="store_true")
    parser.add_argument("--peak-debug", dest="peak_debug", action="store_true")
    parser.add_argument("--paper-quality", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--format",
        choices=("pdf", "png", "svg"),
        default="pdf",
    )
    args = parser.parse_args()

    history = FormationExperimentHistory.load(args.history)
    legacy = _load_legacy_module(args.repo_root.resolve())
    data = _build_legacy_data(history, legacy)
    data["legacy"] = legacy
    selected = _selected_plots(args)

    scenario = data["scenario"]
    trajectory = data["trajectory"]
    distance_domain = data["distance_domain"]
    fov_domain = data["fov_domain"]
    rho_history = data["rho_history"]
    image_history = data["image_history"]
    clf = data["clf"]
    adaptive = bool(data["adaptive"])

    _print_summary(data)
    figures: dict[str, plt.Figure | None] = {}

    if "trajectory" in selected:
        figures["trajectory_3d"] = _trajectory_figure(data, args.paper_quality)

    if "leader_tracking" in selected:
        figures["leader_tracking"] = _leader_tracking_figure(
            data,
            args.paper_quality,
        )
        if figures["leader_tracking"] is None:
            print(
                "Skipping leader-tracking plot: leader reference_position was "
                "not present in the snapshot."
            )

    if "distance" in selected:
        figures["distance"] = legacy.plot_distance_diagnostics(
            scenario,
            trajectory,
            distance_domain,
            rho_history,
            adaptive=adaptive,
        )

    if "fov" in selected:
        horizontal, vertical = legacy.plot_fov_diagnostics(
            scenario,
            trajectory,
            fov_domain,
            rho_history,
            image_history,
            adaptive=adaptive,
        )
        figures["fov_horizontal"] = horizontal
        figures["fov_vertical"] = vertical

    if "adaptive_fov" in selected and adaptive:
        observer = data["first_follower"]
        if args.observer is not None:
            if args.observer not in data["robot_index"]:
                raise SystemExit(
                    f"unknown --observer {args.observer!r}; choose one of "
                    f"{tuple(data['robots'])}"
                )
            observer = data["robot_index"][args.observer]
        figure, _ = plot_vertical_fov_relaxation_from_histories(
            trajectory.times,
            image_history,
            data["s_history"],
            observer=observer,
            alpha_conservative=fov_domain.alpha_v_conservative,
            alpha_physical=1.0,
            domain_margin_ratio=float(data["domain_margin_ratio"]),
            paper_quality=args.paper_quality,
        )
        figures["adaptive_fov_vertical"] = figure

    if "slack" in selected:
        figures["slack"] = legacy.plot_slack(
            scenario,
            trajectory,
            data["slacks"],
            data["required_slacks"],
        )

    if "actuation" in selected:
        figures["actuation_margin"] = legacy.plot_actuation_margin(
            scenario,
            trajectory,
            data["actuation_margins"],
        )

    if "domain" in selected and adaptive:
        figures["domain_enlargement"] = legacy.plot_domain_enlargement(
            scenario,
            trajectory,
            distance_domain,
            fov_domain,
            rho_history,
        )

    if "relaxation_rates" in selected and adaptive:
        figures["relaxation_rates"] = legacy.plot_relaxation_rates(
            scenario,
            trajectory,
            data["relaxation_rate_history"],
        )

    if "thrusters" in selected:
        figures["thruster_forces"] = legacy.plot_thruster_forces(
            scenario,
            trajectory,
            data["allocation"],
            control_space="thruster",
        )

    if "controller_time" in selected:
        figures["controller_time"] = legacy.plot_controller_times(
            scenario,
            trajectory,
            data["controller_times"],
            control_space="thruster",
        )

    if "clf_value" in selected:
        figures["clf_value_decay"] = legacy.plot_clf_value_and_decay(
            scenario,
            trajectory,
            clf,
        )

    if "clf_balance" in selected:
        figures["clf_feasibility_balance"] = legacy.plot_clf_feasibility_balance(
            scenario,
            trajectory,
            clf,
        )

    if "clf_drift" in selected:
        figures["clf_drift_components"] = legacy.plot_clf_drift_components(
            scenario,
            trajectory,
            clf,
        )

    if "backstepping" in selected:
        figures["velocity_backstepping_split"] = legacy.plot_velocity_backstepping_split(
            scenario,
            trajectory,
            clf,
        )

    if "peak_debug" in selected:
        required = data["required_slacks"]
        figures["velocity_backstepping_peak"] = (
            legacy.plot_velocity_backstepping_peak_detail(
                trajectory,
                required,
                clf,
            )
        )
        figures["command_velocity_peak"] = legacy.plot_command_velocity_peak_detail(
            trajectory,
            required,
            clf,
        )
        figures["command_acceleration_peak"] = (
            legacy.plot_command_acceleration_peak_detail(
                trajectory,
                required,
                clf,
            )
        )
        figures["clf_peak_detail"] = legacy.plot_clf_peak_detail(
            scenario,
            trajectory,
            required,
            clf,
        )

    if args.save:
        output_dir = args.output_dir or args.history.parent / "plots"
        _save_figures(
            figures,
            output_dir,
            args.format,
            args.paper_quality,
        )

    if args.show:
        plt.show()
    else:
        for figure in figures.values():
            if figure is not None:
                plt.close(figure)


if __name__ == "__main__":
    main()
