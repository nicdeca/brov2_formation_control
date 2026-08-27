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

    uv run python scripts/plot_formation_experiment.py RUN/formation_history.npz \
        --animation --save --animation-format mp4 --frame-stride 2
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
from matplotlib.animation import FuncAnimation

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
    save_animation,
    save_figure,
)
from formation_control.visualization.formation_3d import (
    DesiredVehicleStyle3D,
    animate_formation_3d,
)
from formation_control.visualization.domain_relaxation_plot import (
    plot_vertical_fov_relaxation_from_histories,
)


PAPER_PLOTS = {
    "trajectory",
    "leader_tracking",
    "leader_position",
    "formation_error",
    "formation_tracking",
    "workspace",
    "workspace_relaxation",
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


def _apply_paper_quality_style() -> None:
    """Use large fonts suitable for figures embedded in an ICRA paper."""
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 17,
            "axes.titlesize": 17,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 14,
            "figure.titlesize": 17,
            "lines.linewidth": 2.0,
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.major.size": 5.0,
            "ytick.major.size": 5.0,
        }
    )


def _apply_plot_style(paper_quality: bool) -> None:
    """Apply the shared style and then enforce larger paper-quality fonts."""
    apply_visualization_style(paper_quality=paper_quality)
    if paper_quality:
        _apply_paper_quality_style()


def _apply_paper_quality_to_figure(figure: plt.Figure) -> None:
    """Enlarge text and line elements of an already-created figure."""
    for axis in figure.axes:
        axis.title.set_fontsize(17)
        axis.xaxis.label.set_fontsize(17)
        axis.yaxis.label.set_fontsize(17)

        axis.tick_params(
            axis="both",
            which="major",
            labelsize=15,
            width=1.2,
            length=5.0,
        )

        legend = axis.get_legend()
        if legend is not None:
            for legend_text in legend.get_texts():
                legend_text.set_fontsize(14)

        for line in axis.get_lines():
            line.set_linewidth(max(line.get_linewidth(), 2.0))


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


def _desired_position_history(data: dict[str, object]) -> np.ndarray:
    """Reconstruct the time-varying absolute position reference of every robot.

    The root uses its logged absolute reference. Each follower reference is
    reconstructed recursively from its parent's desired absolute position and
    its logged parent-minus-follower formation reference.
    """
    arrays = data["arrays"]
    graph = data["graph"]
    trajectory = data["trajectory"]

    times = np.asarray(arrays["times"], dtype=float)
    positions = np.asarray(arrays["positions"], dtype=float)
    root_reference = np.asarray(
        arrays["reference_position"][:, graph.root],
        dtype=float,
    )
    relative_reference = np.asarray(
        arrays["desired_relative_position"],
        dtype=float,
    )

    n_samples = times.size
    n_agents = graph.n_agents
    desired = np.full((n_samples, n_agents, 3), np.nan, dtype=float)

    # Root: use the logged reference whenever available. During initialization
    # a missing reference is intentionally left invisible rather than replaced
    # by a fictitious desired point.
    finite_root = np.all(np.isfinite(root_reference), axis=1)
    desired[finite_root, graph.root] = root_reference[finite_root]

    # Forward-fill references because they are piecewise constant between
    # command updates and some exported samples may not repeat the value.
    for k in range(1, n_samples):
        if not np.all(np.isfinite(desired[k, graph.root])):
            desired[k, graph.root] = desired[k - 1, graph.root]

    # If the very first root reference is missing but later values exist,
    # back-fill only up to the first valid sample.
    valid_root_idx = np.flatnonzero(
        np.all(np.isfinite(desired[:, graph.root]), axis=1)
    )
    if valid_root_idx.size:
        first = int(valid_root_idx[0])
        desired[:first, graph.root] = desired[first, graph.root]

    # Reconstruct each follower recursively along the directed tree:
    # d_ij^d = p_j^d - p_i^d  =>  p_i^d = p_j^d - d_ij^d.
    unresolved = set(range(n_agents)) - {graph.root}
    while unresolved:
        progress = False
        for edge in graph:
            observer = edge.observer
            target = edge.target
            if observer not in unresolved:
                continue

            rel = np.asarray(relative_reference[:, observer, :], dtype=float).copy()

            # Forward-fill the piecewise-constant formation reference.
            for k in range(1, n_samples):
                if not np.all(np.isfinite(rel[k])):
                    rel[k] = rel[k - 1]

            valid_rel_idx = np.flatnonzero(np.all(np.isfinite(rel), axis=1))
            if valid_rel_idx.size:
                first = int(valid_rel_idx[0])
                rel[:first] = rel[first]

            parent = desired[:, target, :]
            valid = (
                np.all(np.isfinite(parent), axis=1)
                & np.all(np.isfinite(rel), axis=1)
            )
            if not np.any(valid):
                continue

            desired[valid, observer, :] = parent[valid] - rel[valid]
            unresolved.remove(observer)
            progress = True

        if not progress:
            break

    # FormationTrajectory carries one repeated terminal state beyond the
    # controller-history arrays, so mirror that convention here.
    if trajectory.n_samples == n_samples + 1:
        desired = np.concatenate((desired, desired[-1:]), axis=0)
    elif trajectory.n_samples != n_samples:
        raise ValueError(
            "desired-position history cannot be aligned with trajectory samples"
        )

    return desired


def _animation(
    data: dict[str, object],
    *,
    frame_stride: int,
    elevation: float,
    azimuth: float,
):
    """Build a video-friendly 3-D animation from the exported ROS history."""
    scenario = data["scenario"]
    trajectory = data["trajectory"]
    camera = data["camera"]
    distance_domain = data["distance_domain"]
    fov_domain = data["fov_domain"]
    image_history = data["image_history"]
    constraints = data["constraints_enabled"]
    legacy = data["legacy"]

    desired_history = _desired_position_history(data)
    vehicle_geometry = BlueROV2HeavyVisualGeometry().wireframe()
    followers = tuple(
        agent
        for agent in range(scenario.n_agents)
        if agent != scenario.graph.root
    )

    edge_quality = None
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
    except (AttributeError, ValueError):
        pass

    kwargs = {
        "desired_position_history": desired_history,
        "desired_vehicle_style": DesiredVehicleStyle3D(
            color="0.45",
            alpha=0.24,
            linestyle="--",
            linewidth_scale=0.85,
        ),
        "camera": camera,
        "camera_agents": followers,
        "camera_depth": 0.75,
        "vehicle_geometry": vehicle_geometry,
        "edge_quality": edge_quality,
        "paper_quality": False,
        "show_body_forward": False,
        "trail_length": 250,
        "frame_stride": frame_stride,
        "elevation": elevation,
        "azimuth": azimuth,
        "title": "BlueROV2 formation-control experiment",
    }

    # Keep this script usable while formation_3d.py is being rolled out.
    signature = inspect.signature(animate_formation_3d)
    kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }

    control_times = np.asarray(data["arrays"]["times"], dtype=float)
    dt = float(np.median(np.diff(control_times)))
    kwargs["interval_ms"] = max(1, int(round(1000.0 * dt * frame_stride)))

    return animate_formation_3d(
        trajectory,
        scenario.graph,
        **kwargs,
    ), dt


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




def _edge_subscript(edge) -> str:
    """Return the paper's 1-based edge subscript ``ij``."""
    return f"{edge.observer + 1}{edge.target + 1}"



def _animation_frame_indices(n_samples: int, frame_stride: int) -> list[int]:
    indices = list(range(0, n_samples, frame_stride))
    if indices[-1] != n_samples - 1:
        indices.append(n_samples - 1)
    return indices


def _finite_ylim(
    series: list[np.ndarray],
    *,
    padding_fraction: float = 0.08,
) -> tuple[float, float]:
    values = np.concatenate(
        [
            np.asarray(value, dtype=float).reshape(-1)
            for value in series
            if np.asarray(value).size
        ]
    )
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -1.0, 1.0
    lower = float(np.min(values))
    upper = float(np.max(values))
    span = upper - lower
    if span <= 1e-12:
        span = max(abs(lower), 1.0)
    padding = padding_fraction * span
    return lower - padding, upper + padding


def _formation_error_animation(
    data: dict[str, object],
    *,
    paper_quality: bool,
    frame_stride: int,
    show_legends: bool,
):
    """Animate formation-error norms with a fixed time axis and moving cursor."""
    arrays = data["arrays"]
    graph = data["graph"]
    robots = data["robots"]

    times = np.asarray(arrays["times"], dtype=float)
    positions = np.asarray(arrays["positions"], dtype=float)
    desired = np.asarray(arrays["desired_relative_position"], dtype=float)

    _apply_plot_style(paper_quality)
    figure, axis = plt.subplots()

    line_specs = []
    all_values: list[np.ndarray] = []
    for edge in graph:
        observer = edge.observer
        target = edge.target
        error_vector = (
            positions[:, target, :]
            - positions[:, observer, :]
            - desired[:, observer, :]
        )
        error_norm = np.linalg.norm(error_vector, axis=1)
        valid = np.all(np.isfinite(error_vector), axis=1) & np.isfinite(times)
        values = np.full_like(times, np.nan, dtype=float)
        values[valid] = error_norm[valid]
        ij = _edge_subscript(edge)
        line = axis.plot([], [], label=rf"$\|\tilde{{\boldsymbol{{p}}}}_{{{ij}}}\|$")[0]
        line_specs.append((line, values))
        all_values.append(values)

    axis.set_xlim(float(times[0]), float(times[-1]))
    lower, upper = _finite_ylim(all_values)
    axis.set_ylim(max(0.0, lower), upper)
    axis.set_xlabel(r"$t$ [s]")
    axis.set_ylabel(r"$\|\tilde{\boldsymbol{p}}_{ij}\|$ [m]")
    if not paper_quality:
        axis.set_title("Formation error")
    axis.grid(True, alpha=0.3)
    if show_legends:
        axis.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=min(4, max(1, len(line_specs))),
            frameon=False,
        )
    cursor = axis.axvline(times[0], color="0.35", linewidth=1.2, alpha=0.8)
    time_text = axis.text(
        0.985,
        0.04,
        "",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
    )
    figure.tight_layout()

    frames = _animation_frame_indices(len(times), frame_stride)

    def update(frame: int):
        for line, values in line_specs:
            line.set_data(times[: frame + 1], values[: frame + 1])
        cursor.set_xdata([times[frame], times[frame]])
        time_text.set_text(rf"$t={times[frame]:.1f}\,\mathrm{{s}}$")
        return [
            *(line for line, _ in line_specs),
            cursor,
            time_text,
        ]

    animation = FuncAnimation(
        figure,
        update,
        frames=frames,
        interval=30,
        blit=False,
        repeat=False,
    )
    return figure, animation


def _distance_diagnostics_animation(
    data: dict[str, object],
    *,
    paper_quality: bool,
    frame_stride: int,
    show_legends: bool,
):
    """Animate measured distances and adaptive distance boundaries."""
    scenario = data["scenario"]
    trajectory = data["trajectory"]
    distance_domain = data["distance_domain"]
    rho_history = np.asarray(data["rho_history"], dtype=float)
    adaptive = bool(data["adaptive"])
    times = np.asarray(trajectory.times, dtype=float)

    _apply_plot_style(paper_quality)
    figure, axis = plt.subplots()

    animated_lines = []
    all_values: list[np.ndarray] = []

    for edge in scenario.graph:
        observer = edge.observer
        target = edge.target
        ij = _edge_subscript(edge)

        relative = (
            trajectory.positions[:, target, :]
            - trajectory.positions[:, observer, :]
        )
        distance = np.linalg.norm(relative, axis=1)
        measured = axis.plot([], [], linestyle="-", label=rf"$d_{{{ij}}}$")[0]
        color = measured.get_color()
        animated_lines.append((measured, distance))
        all_values.append(distance)

        if adaptive:
            adaptive_minimum = np.array(
                [
                    distance_domain.effective_minimum_distance(value)
                    for value in rho_history[:, observer, 0]
                ],
                dtype=float,
            )
            adaptive_maximum = np.array(
                [
                    distance_domain.effective_maximum_distance(value)
                    for value in rho_history[:, observer, 1]
                ],
                dtype=float,
            )
            line_min = axis.plot(
                [], [], "--", color=color, alpha=0.85,
                label=rf"$d_{{\min,{ij}}}^a$",
            )[0]
            line_max = axis.plot(
                [], [], "--", color=color, alpha=0.85,
                label=rf"$d_{{\max,{ij}}}^a$",
            )[0]
            animated_lines.extend(
                [(line_min, adaptive_minimum), (line_max, adaptive_maximum)]
            )
            all_values.extend([adaptive_minimum, adaptive_maximum])

    for value, label, linestyle, color in (
        (distance_domain.d_min, r"$d_{\min}$", ":", "0.15"),
        (distance_domain.d_max, r"$d_{\max}$", ":", "0.15"),
        (
            distance_domain.d_min_conservative,
            r"$d_{\min}^{c}$",
            "-.",
            "0.50",
        ),
        (
            distance_domain.d_max_conservative,
            r"$d_{\max}^{c}$",
            "-.",
            "0.50",
        ),
    ):
        axis.axhline(
            value,
            linestyle=linestyle,
            color=color,
            linewidth=1.8,
            label=label,
        )
        all_values.append(np.array([value]))

    axis.set_xlim(float(times[0]), float(times[-1]))
    axis.set_ylim(*_finite_ylim(all_values))
    axis.set_xlabel(r"$t$ [s]")
    axis.set_ylabel(r"$d_{ij}$ [m]")
    if not paper_quality:
        axis.set_title("Inter-robot distance")
    axis.grid(True, alpha=0.3)
    if show_legends:
        axis.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=4,
            frameon=False,
        )
    cursor = axis.axvline(times[0], color="0.35", linewidth=1.2, alpha=0.8)
    time_text = axis.text(
        0.985, 0.04, "", transform=axis.transAxes, ha="right", va="bottom"
    )
    figure.tight_layout()

    frames = _animation_frame_indices(len(times), frame_stride)

    def update(frame: int):
        for line, values in animated_lines:
            line.set_data(times[: frame + 1], values[: frame + 1])
        cursor.set_xdata([times[frame], times[frame]])
        time_text.set_text(rf"$t={times[frame]:.1f}\,\mathrm{{s}}$")
        return [
            *(line for line, _ in animated_lines),
            cursor,
            time_text,
        ]

    animation = FuncAnimation(
        figure,
        update,
        frames=frames,
        interval=30,
        blit=False,
        repeat=False,
    )
    return figure, animation


def _fov_diagnostics_animations(
    data: dict[str, object],
    *,
    paper_quality: bool,
    frame_stride: int,
    show_legends: bool,
):
    """Animate horizontal and vertical FoV measurements/adaptive bounds."""
    scenario = data["scenario"]
    trajectory = data["trajectory"]
    fov_domain = data["fov_domain"]
    rho_history = np.asarray(data["rho_history"], dtype=float)
    image_history = np.asarray(data["image_history"], dtype=float)
    adaptive = bool(data["adaptive"])
    times = np.asarray(trajectory.times, dtype=float)

    def build(
        *,
        component: int,
        channel: int,
        symbol: str,
        conservative_limit: float,
        title: str,
    ):
        _apply_plot_style(paper_quality)
        figure, axis = plt.subplots()
        animated_lines = []
        all_values: list[np.ndarray] = [
            np.array([-1.0, 1.0, -conservative_limit, conservative_limit])
        ]

        for edge in scenario.graph:
            observer = edge.observer
            ij = _edge_subscript(edge)
            measured_values = image_history[:, observer, component]
            measured = axis.plot(
                [],
                [],
                "-",
                label=rf"$\alpha_{{{symbol},{ij}}}$",
            )[0]
            color = measured.get_color()
            animated_lines.append((measured, measured_values))
            all_values.append(measured_values)

            if adaptive:
                if component == 0:
                    adaptive_limit = np.array(
                        [
                            fov_domain.effective_horizontal_limit(value)
                            for value in rho_history[:, observer, channel]
                        ],
                        dtype=float,
                    )
                else:
                    adaptive_limit = np.array(
                        [
                            fov_domain.effective_vertical_limit(value)
                            for value in rho_history[:, observer, channel]
                        ],
                        dtype=float,
                    )
                line_pos = axis.plot(
                    [],
                    [],
                    "--",
                    color=color,
                    alpha=0.85,
                    label=rf"$+\alpha_{{{symbol},{ij}}}^{{a}}$",
                )[0]
                line_neg = axis.plot(
                    [],
                    [],
                    "--",
                    color=color,
                    alpha=0.85,
                    label=rf"$-\alpha_{{{symbol},{ij}}}^{{a}}$",
                )[0]
                animated_lines.extend(
                    [(line_pos, adaptive_limit), (line_neg, -adaptive_limit)]
                )
                all_values.extend([adaptive_limit, -adaptive_limit])

        axis.axhline(
            1.0,
            linestyle=":",
            color="0.15",
            linewidth=1.8,
            label=rf"$+\alpha_{{{symbol}}}$",
        )
        axis.axhline(
            -1.0,
            linestyle=":",
            color="0.15",
            linewidth=1.8,
            label=rf"$-\alpha_{{{symbol}}}$",
        )
        axis.axhline(
            conservative_limit,
            linestyle="-.",
            color="0.50",
            linewidth=1.8,
            label=rf"$+\alpha_{{{symbol}}}^c$",
        )
        axis.axhline(
            -conservative_limit,
            linestyle="-.",
            color="0.50",
            linewidth=1.8,
            label=rf"$-\alpha_{{{symbol}}}^c$",
        )

        axis.set_xlim(float(times[0]), float(times[-1]))
        axis.set_ylim(*_finite_ylim(all_values))
        axis.set_xlabel(r"$t$ [s]")
        axis.set_ylabel(rf"$\alpha_{{{symbol},ij}}$")
        if not paper_quality:
            axis.set_title(title)
        axis.grid(True, alpha=0.3)
        if show_legends:
            axis.legend(
                loc="lower center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=4,
                frameon=False,
            )
        cursor = axis.axvline(
            times[0], color="0.35", linewidth=1.2, alpha=0.8
        )
        time_text = axis.text(
            0.985,
            0.04,
            "",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
        )
        figure.tight_layout()

        frames = _animation_frame_indices(len(times), frame_stride)

        def update(frame: int):
            for line, values in animated_lines:
                line.set_data(times[: frame + 1], values[: frame + 1])
            cursor.set_xdata([times[frame], times[frame]])
            time_text.set_text(rf"$t={times[frame]:.1f}\,\mathrm{{s}}$")
            return [
                *(line for line, _ in animated_lines),
                cursor,
                time_text,
            ]

        animation = FuncAnimation(
            figure,
            update,
            frames=frames,
            interval=30,
            blit=False,
            repeat=False,
        )
        return figure, animation

    return {
        "fov_horizontal": build(
            component=0,
            channel=2,
            symbol="h",
            conservative_limit=fov_domain.alpha_h_conservative,
            title="Horizontal field of view",
        ),
        "fov_vertical": build(
            component=1,
            channel=3,
            symbol="v",
            conservative_limit=fov_domain.alpha_v_conservative,
            title="Vertical field of view",
        ),
    }


def _distance_diagnostics_figure(
    data: dict[str, object],
    paper_quality: bool,
    show_legends: bool,
) -> plt.Figure:
    """Plot inter-robot distances and adaptive distance-domain boundaries.

    Each sensing edge receives one color. The measured distance and both
    adaptive boundaries for that edge use the same color, while line style
    distinguishes the measured quantity from the adaptive limits.
    """
    scenario = data["scenario"]
    trajectory = data["trajectory"]
    distance_domain = data["distance_domain"]
    rho_history = np.asarray(data["rho_history"], dtype=float)
    adaptive = bool(data["adaptive"])

    _apply_plot_style(paper_quality)
    figure, axis = plt.subplots()

    for edge in scenario.graph:
        observer = edge.observer
        target = edge.target
        ij = _edge_subscript(edge)

        relative = (
            trajectory.positions[:, target, :]
            - trajectory.positions[:, observer, :]
        )
        distance = np.linalg.norm(relative, axis=1)

        measured_line = axis.plot(
            trajectory.times,
            distance,
            linestyle="-",
            label=rf"$d_{{{ij}}}$",
        )[0]
        edge_color = measured_line.get_color()

        if adaptive:
            adaptive_minimum = np.array(
                [
                    distance_domain.effective_minimum_distance(value)
                    for value in rho_history[:, observer, 0]
                ],
                dtype=float,
            )
            adaptive_maximum = np.array(
                [
                    distance_domain.effective_maximum_distance(value)
                    for value in rho_history[:, observer, 1]
                ],
                dtype=float,
            )

            axis.plot(
                trajectory.times,
                adaptive_minimum,
                linestyle="--",
                color=edge_color,
                alpha=0.85,
                label=rf"$d_{{\min,{ij}}}^a$",
            )
            axis.plot(
                trajectory.times,
                adaptive_maximum,
                linestyle="--",
                color=edge_color,
                alpha=0.85,
                label=rf"$d_{{\max,{ij}}}^a$",
            )

    # Global physical and conservative bounds are edge independent.
    axis.axhline(
        distance_domain.d_min,
        linestyle=":",
        color="0.15",
        linewidth=1.8,
        label=r"$d_{\min}$",
    )
    axis.axhline(
        distance_domain.d_max,
        linestyle=":",
        color="0.15",
        linewidth=1.8,
        label=r"$d_{\max}$",
    )
    axis.axhline(
        distance_domain.d_min_conservative,
        linestyle="-.",
        color="0.50",
        linewidth=1.8,
        label=r"$d_{\min}^{c}$",
    )
    axis.axhline(
        distance_domain.d_max_conservative,
        linestyle="-.",
        color="0.50",
        linewidth=1.8,
        label=r"$d_{\max}^{c}$",
    )

    axis.set_xlabel(r"$t$ [s]")
    axis.set_ylabel(r"$d_{ij}$ [m]")
    if not paper_quality:
        axis.set_title("Inter-robot distance")
    axis.grid(True, alpha=0.3)
    if show_legends:
        axis.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=4,
            frameon=False,
        )
    figure.tight_layout()
    return figure


def _fov_diagnostics_figures(
    data: dict[str, object],
    paper_quality: bool,
    show_legends: bool,
) -> tuple[plt.Figure, plt.Figure]:
    """Plot horizontal and vertical normalized-image FoV constraints.

    Each edge has one color. The measured normalized image coordinate and its
    adaptive positive/negative FoV boundaries share that edge color.
    """
    scenario = data["scenario"]
    trajectory = data["trajectory"]
    fov_domain = data["fov_domain"]
    rho_history = np.asarray(data["rho_history"], dtype=float)
    image_history = np.asarray(data["image_history"], dtype=float)
    adaptive = bool(data["adaptive"])

    def make_component(
        *,
        component: int,
        channel: int,
        symbol: str,
        conservative_limit: float,
        title: str,
    ) -> plt.Figure:
        _apply_plot_style(paper_quality)
        figure, axis = plt.subplots()

        for edge in scenario.graph:
            observer = edge.observer
            ij = _edge_subscript(edge)

            measured_line = axis.plot(
                trajectory.times,
                image_history[:, observer, component],
                linestyle="-",
                label=rf"$\alpha_{{{symbol},{ij}}}$",
            )[0]
            edge_color = measured_line.get_color()

            if adaptive:
                if component == 0:
                    adaptive_limit = np.array(
                        [
                            fov_domain.effective_horizontal_limit(value)
                            for value in rho_history[:, observer, channel]
                        ],
                        dtype=float,
                    )
                else:
                    adaptive_limit = np.array(
                        [
                            fov_domain.effective_vertical_limit(value)
                            for value in rho_history[:, observer, channel]
                        ],
                        dtype=float,
                    )

                axis.plot(
                    trajectory.times,
                    adaptive_limit,
                    linestyle="--",
                    color=edge_color,
                    alpha=0.85,
                    label=rf"$+\alpha_{{{symbol},{ij}}}^{{a}}$",
                )
                axis.plot(
                    trajectory.times,
                    -adaptive_limit,
                    linestyle="--",
                    color=edge_color,
                    alpha=0.85,
                    label=rf"$-\alpha_{{{symbol},{ij}}}^{{a}}$",
                )

        # Normalized image coordinates have physical FoV limits at +/- 1.
        axis.axhline(
            1.0,
            linestyle=":",
            color="0.15",
            linewidth=1.8,
            label=rf"$+\alpha_{{{symbol}}}$",
        )
        axis.axhline(
            -1.0,
            linestyle=":",
            color="0.15",
            linewidth=1.8,
            label=rf"$-\alpha_{{{symbol}}}$",
        )
        axis.axhline(
            conservative_limit,
            linestyle="-.",
            color="0.50",
            linewidth=1.8,
            label=rf"$+\alpha_{{{symbol}}}^c$",
        )
        axis.axhline(
            -conservative_limit,
            linestyle="-.",
            color="0.50",
            linewidth=1.8,
            label=rf"$-\alpha_{{{symbol}}}^c$",
        )

        axis.set_xlabel(r"$t$ [s]")
        axis.set_ylabel(rf"$\alpha_{{{symbol},ij}}$")
        if not paper_quality:
            axis.set_title(title)
        axis.grid(True, alpha=0.3)
        if show_legends:
            axis.legend(
                loc="lower center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=4,
                frameon=False,
            )
        figure.tight_layout()
        return figure

    horizontal = make_component(
        component=0,
        channel=2,
        symbol="h",
        conservative_limit=fov_domain.alpha_h_conservative,
        title="Horizontal field of view",
    )
    vertical = make_component(
        component=1,
        channel=3,
        symbol="v",
        conservative_limit=fov_domain.alpha_v_conservative,
        title="Vertical field of view",
    )
    return horizontal, vertical


def _formation_error_figure(data: dict[str, object], paper_quality: bool, show_legends: bool):
    """Plot ||(p_target-p_observer)-d_des|| for every directed follower edge."""
    arrays = data["arrays"]
    graph = data["graph"]
    robots = data["robots"]

    times = np.asarray(arrays["times"], dtype=float)
    positions = np.asarray(arrays["positions"], dtype=float)
    desired = np.asarray(arrays["desired_relative_position"], dtype=float)

    _apply_plot_style(paper_quality)
    figure, axis = plt.subplots()

    plotted = False
    for edge in graph:
        observer = edge.observer
        target = edge.target

        actual_relative = (
            positions[:, target, :] - positions[:, observer, :]
        )
        desired_relative = desired[:, observer, :]
        error_vector = actual_relative - desired_relative

        valid = (
            np.all(np.isfinite(error_vector), axis=1)
            & np.isfinite(times)
        )
        if np.count_nonzero(valid) < 2:
            continue

        error_norm = np.linalg.norm(error_vector[valid], axis=1)
        axis.plot(
            times[valid],
            error_norm,
            label=rf"$\|\tilde{{\boldsymbol{{p}}}}_{{{observer + 1}{target + 1}}}\|$",
        )
        plotted = True

    if not plotted:
        plt.close(figure)
        return None

    axis.set_xlabel(r"$t$ [s]")
    axis.set_ylabel(r"$\|\tilde{\boldsymbol{p}}_{ij}\|$ [m]")
    axis.set_title("Formation error")
    axis.grid(True, alpha=0.3)
    if show_legends and len(tuple(graph.edges)) > 1:
        axis.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=min(4, len(tuple(graph.edges))),
            frameon=False,
        )
    figure.tight_layout()
    return figure




def _thruster_force_figures(
    data: dict[str, object],
    paper_quality: bool,
) -> dict[str, plt.Figure]:
    """Plot all logged thruster forces and force limits, one figure per robot."""
    arrays = data["arrays"]
    robots = data["robots"]
    times = np.asarray(arrays["times"], dtype=float)
    forces = np.asarray(arrays["thruster_forces"], dtype=float)
    force_limits = np.asarray(arrays["thruster_force_limits"], dtype=float)

    figures: dict[str, plt.Figure] = {}

    for agent, robot in enumerate(robots):
        values = forces[:, agent, :]
        if values.ndim != 2:
            continue

        valid = np.any(np.isfinite(values), axis=1) & np.isfinite(times)
        if np.count_nonzero(valid) < 2:
            continue

        # The logged pair is [reverse_magnitude, forward_magnitude].
        # Both are positive configuration values; the signed admissible force
        # interval is [-reverse_magnitude, +forward_magnitude].
        # Use the first finite sample because these are fixed parameters.
        limits_history = force_limits[:, agent, :]
        finite_limits = np.all(np.isfinite(limits_history), axis=1)
        if np.any(finite_limits):
            reverse_limit, forward_limit = limits_history[
                np.flatnonzero(finite_limits)[0]
            ]
        else:
            reverse_limit = np.nan
            forward_limit = np.nan

        _apply_plot_style(paper_quality)
        figure, axis = plt.subplots()

        for thruster in range(values.shape[1]):
            axis.plot(
                times[valid],
                values[valid, thruster],
                label=f"T{thruster + 1}",
            )

        if np.isfinite(forward_limit):
            axis.axhline(
                forward_limit,
                linestyle="--",
                linewidth=1.2,
                label="forward limit",
            )
        if np.isfinite(reverse_limit):
            # The logged reverse limit is a positive force magnitude.
            # The admissible signed thruster-force lower bound is therefore
            # -reverse_limit.
            axis.axhline(
                -reverse_limit,
                linestyle="--",
                linewidth=1.2,
                label="reverse limit",
            )

        axis.set_xlabel(r"$t$ [s]")
        axis.set_ylabel("Thruster force [N]")
        axis.set_title(f"Thruster forces: {robot}")
        axis.grid(True, alpha=0.3)
        axis.legend(ncol=2)
        figure.tight_layout()

        figures[f"thruster_forces_{robot}"] = figure

    return figures


def _leader_position_figure(data: dict[str, object], paper_quality: bool):
    """Plot actual and desired leader position component by component."""
    arrays = data["arrays"]
    graph = data["graph"]
    times = np.asarray(arrays["times"], dtype=float)
    root = graph.root

    position = np.asarray(arrays["positions"][:, root], dtype=float)
    reference = np.asarray(
        arrays["reference_position"][:, root],
        dtype=float,
    )
    valid = (
        np.all(np.isfinite(position), axis=1)
        & np.all(np.isfinite(reference), axis=1)
        & np.isfinite(times)
    )
    if np.count_nonzero(valid) < 2:
        return None

    _apply_plot_style(paper_quality)
    figure, axes = plt.subplots(3, 1, sharex=True)
    labels = ("x", "y", "z")

    for axis_index, axis in enumerate(axes):
        axis.plot(
            times[valid],
            position[valid, axis_index],
            label="actual",
        )
        axis.plot(
            times[valid],
            reference[valid, axis_index],
            "--",
            label="reference",
        )
        axis.set_ylabel(rf"${labels[axis_index]}$ [m]")
        axis.grid(True, alpha=0.3)

    axes[0].set_title("Leader position tracking")
    axes[0].legend()
    axes[-1].set_xlabel(r"$t$ [s]")
    figure.tight_layout()
    return figure


def _formation_tracking_figures(
    data: dict[str, object],
    paper_quality: bool,
) -> dict[str, plt.Figure]:
    """Actual versus desired parent-minus-follower vector for each edge."""
    arrays = data["arrays"]
    graph = data["graph"]
    robots = data["robots"]
    times = np.asarray(arrays["times"], dtype=float)
    positions = np.asarray(arrays["positions"], dtype=float)
    desired = np.asarray(
        arrays["desired_relative_position"],
        dtype=float,
    )

    figures: dict[str, plt.Figure] = {}
    labels = ("x", "y", "z")

    for edge in graph:
        observer = edge.observer
        target = edge.target
        actual = positions[:, target, :] - positions[:, observer, :]
        reference = desired[:, observer, :]

        valid = (
            np.all(np.isfinite(actual), axis=1)
            & np.all(np.isfinite(reference), axis=1)
            & np.isfinite(times)
        )
        if np.count_nonzero(valid) < 2:
            continue

        _apply_plot_style(paper_quality)
        figure, axes = plt.subplots(3, 1, sharex=True)

        for axis_index, axis in enumerate(axes):
            axis.plot(
                times[valid],
                actual[valid, axis_index],
                label="actual",
            )
            axis.plot(
                times[valid],
                reference[valid, axis_index],
                "--",
                label="desired",
            )
            axis.set_ylabel(rf"$d_{labels[axis_index]}$ [m]")
            axis.grid(True, alpha=0.3)

        axes[0].set_title(
            "Formation tracking: "
            f"{robots[observer]}->{robots[target]}"
        )
        axes[0].legend()
        axes[-1].set_xlabel(r"$t$ [s]")
        figure.tight_layout()
        figures[
            f"formation_tracking_{robots[observer]}_to_{robots[target]}"
        ] = figure

    return figures


_WORKSPACE_NAMES = (
    "x_min",
    "x_max",
    "y_min",
    "y_max",
    "z_min",
    "z_max",
)


def _workspace_available(arrays: dict[str, np.ndarray]) -> bool:
    required = (
        "workspace_relaxation",
        "workspace_conservative_constraint_values",
        "workspace_physical_constraint_values",
    )
    return all(name in arrays for name in required)


def _workspace_bounds(
    position: np.ndarray,
    conservative_values: np.ndarray,
    physical_values: np.ndarray,
    relaxation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover [lower, upper] wall positions from logged constraint margins.

    Constraint channel order:
      [x_min, x_max, y_min, y_max, z_min, z_max].

    For each axis h_min = p-lower and h_max = upper-p.  The adaptive
    constraint margin is reconstructed by interpolating from conservative
    margin to physical margin using the normalized relaxation state.
    """
    n = position.shape[0]
    conservative_bounds = np.full((n, 3, 2), np.nan)
    physical_bounds = np.full((n, 3, 2), np.nan)
    adaptive_bounds = np.full((n, 3, 2), np.nan)

    for axis in range(3):
        lower_channel = 2 * axis
        upper_channel = lower_channel + 1
        p = position[:, axis]

        hc_lower = conservative_values[:, lower_channel]
        hc_upper = conservative_values[:, upper_channel]
        hp_lower = physical_values[:, lower_channel]
        hp_upper = physical_values[:, upper_channel]

        conservative_bounds[:, axis, 0] = p - hc_lower
        conservative_bounds[:, axis, 1] = p + hc_upper
        physical_bounds[:, axis, 0] = p - hp_lower
        physical_bounds[:, axis, 1] = p + hp_upper

        s_lower = relaxation[:, lower_channel]
        s_upper = relaxation[:, upper_channel]
        ha_lower = hc_lower + s_lower * (hp_lower - hc_lower)
        ha_upper = hc_upper + s_upper * (hp_upper - hc_upper)
        adaptive_bounds[:, axis, 0] = p - ha_lower
        adaptive_bounds[:, axis, 1] = p + ha_upper

    return conservative_bounds, adaptive_bounds, physical_bounds


def _workspace_figures(
    data: dict[str, object],
    paper_quality: bool,
) -> dict[str, plt.Figure]:
    arrays = data["arrays"]
    robots = data["robots"]
    if not _workspace_available(arrays):
        return {}

    times = np.asarray(arrays["times"], dtype=float)
    positions = np.asarray(arrays["positions"], dtype=float)
    relaxation = np.asarray(
        arrays["workspace_relaxation"],
        dtype=float,
    )
    conservative = np.asarray(
        arrays["workspace_conservative_constraint_values"],
        dtype=float,
    )
    physical = np.asarray(
        arrays["workspace_physical_constraint_values"],
        dtype=float,
    )

    figures: dict[str, plt.Figure] = {}
    labels = ("x", "y", "z")

    for agent, robot in enumerate(robots):
        p = positions[:, agent, :]
        s = relaxation[:, agent, :]
        hc = conservative[:, agent, :]
        hp = physical[:, agent, :]

        valid = (
            np.all(np.isfinite(p), axis=1)
            & np.all(np.isfinite(s), axis=1)
            & np.all(np.isfinite(hc), axis=1)
            & np.all(np.isfinite(hp), axis=1)
            & np.isfinite(times)
        )
        if np.count_nonzero(valid) < 2:
            continue

        cons_b, adaptive_b, physical_b = _workspace_bounds(
            p,
            hc,
            hp,
            s,
        )

        _apply_plot_style(paper_quality)
        figure, axes = plt.subplots(3, 1, sharex=True)

        for axis_index, axis in enumerate(axes):
            axis.plot(
                times[valid],
                p[valid, axis_index],
                label="robot position",
            )
            axis.plot(
                times[valid],
                cons_b[valid, axis_index, 0],
                ":",
                label="conservative bounds" if axis_index == 0 else None,
            )
            axis.plot(
                times[valid],
                cons_b[valid, axis_index, 1],
                ":",
            )
            axis.plot(
                times[valid],
                adaptive_b[valid, axis_index, 0],
                "--",
                label="adaptive bounds" if axis_index == 0 else None,
            )
            axis.plot(
                times[valid],
                adaptive_b[valid, axis_index, 1],
                "--",
            )
            axis.plot(
                times[valid],
                physical_b[valid, axis_index, 0],
                "-.",
                label="physical bounds" if axis_index == 0 else None,
            )
            axis.plot(
                times[valid],
                physical_b[valid, axis_index, 1],
                "-.",
            )
            axis.set_ylabel(rf"${labels[axis_index]}$ [m]")
            axis.grid(True, alpha=0.3)

        axes[0].set_title(f"Workspace constraints: {robot}")
        axes[0].legend()
        axes[-1].set_xlabel(r"$t$ [s]")
        figure.tight_layout()
        figures[f"workspace_{robot}"] = figure

    return figures


def _workspace_relaxation_figures(
    data: dict[str, object],
    paper_quality: bool,
) -> dict[str, plt.Figure]:
    arrays = data["arrays"]
    robots = data["robots"]
    if "workspace_relaxation" not in arrays:
        return {}

    times = np.asarray(arrays["times"], dtype=float)
    relaxation = np.asarray(
        arrays["workspace_relaxation"],
        dtype=float,
    )
    figures: dict[str, plt.Figure] = {}

    for agent, robot in enumerate(robots):
        values = relaxation[:, agent, :]
        valid = np.any(np.isfinite(values), axis=1) & np.isfinite(times)
        if np.count_nonzero(valid) < 2:
            continue

        _apply_plot_style(paper_quality)
        figure, axis = plt.subplots()
        for channel, name in enumerate(_WORKSPACE_NAMES):
            axis.plot(
                times[valid],
                values[valid, channel],
                label=name,
            )
        axis.set_xlabel(r"$t$ [s]")
        axis.set_ylabel(r"$s_W$")
        axis.set_ylim(-0.02, 1.02)
        axis.set_title(f"Workspace relaxation: {robot}")
        axis.grid(True, alpha=0.3)
        axis.legend(ncol=2)
        figure.tight_layout()
        figures[f"workspace_relaxation_{robot}"] = figure

    return figures


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

    _apply_plot_style(paper_quality)
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

        positions = np.asarray(arrays["positions"], dtype=float)
        desired = np.asarray(
            arrays["desired_relative_position"],
            dtype=float,
        )
        for edge in graph:
            observer = edge.observer
            target = edge.target
            error_vector = (
                positions[:, target, :]
                - positions[:, observer, :]
                - desired[:, observer, :]
            )
            valid = np.all(np.isfinite(error_vector), axis=1)
            if not np.any(valid):
                continue
            error_norm = np.linalg.norm(error_vector[valid], axis=1)
            print(
                f"Formation error {robots[observer]}->{robots[target]}: "
                f"RMS {np.sqrt(np.mean(error_norm**2)):.4f} m, "
                f"max {np.max(error_norm):.4f} m"
            )

    if _workspace_available(arrays):
        workspace_s = np.asarray(
            arrays["workspace_relaxation"],
            dtype=float,
        )
        workspace_physical = np.asarray(
            arrays["workspace_physical_constraint_values"],
            dtype=float,
        )
        workspace_margin = np.asarray(
            arrays.get(
                "workspace_minimum_physical_margin",
                np.full((len(times), len(robots)), np.nan),
            ),
            dtype=float,
        )
        for agent, robot in enumerate(robots):
            finite_s = workspace_s[:, agent][
                np.isfinite(workspace_s[:, agent])
            ]
            finite_margin = workspace_margin[:, agent][
                np.isfinite(workspace_margin[:, agent])
            ]
            values = workspace_physical[:, agent, :]
            if np.any(np.isfinite(values)):
                flat_index = np.nanargmin(values)
                _, channel = np.unravel_index(
                    flat_index,
                    values.shape,
                )
                closest_name = _WORKSPACE_NAMES[channel]
            else:
                closest_name = "unknown"

            if finite_margin.size or finite_s.size:
                pieces = [f"Workspace {robot}:"]
                if finite_margin.size:
                    pieces.append(
                        f"min physical margin {np.min(finite_margin):.4f} m"
                    )
                if finite_s.size:
                    pieces.append(
                        f"max relaxation {np.max(finite_s):.3f}"
                    )
                pieces.append(f"closest wall {closest_name}")
                print(", ".join(pieces))

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

        if paper_quality:
            _apply_paper_quality_to_figure(figure)
            figure.tight_layout()

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
    parser.add_argument(
        "--leader-position",
        dest="leader_position",
        action="store_true",
    )
    parser.add_argument(
        "--formation-tracking",
        dest="formation_tracking",
        action="store_true",
    )
    parser.add_argument("--workspace", action="store_true")
    parser.add_argument(
        "--workspace-relaxation",
        dest="workspace_relaxation",
        action="store_true",
    )
    parser.add_argument(
        "--formation-error",
        dest="formation_error",
        action="store_true",
    )
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
    parser.add_argument(
        "--show-legends",
        action="store_true",
        help=(
            "show legends on formation-error and sensing-constraint plots/"
            "animations; hidden by default for compact paper/video output"
        ),
    )
    parser.add_argument(
        "--animation",
        action="store_true",
        help="create a 3-D formation animation with faint desired vehicles",
    )
    parser.add_argument(
        "--diagnostic-animations",
        action="store_true",
        help=(
            "animate formation error and distance/horizontal-FoV/vertical-FoV "
            "diagnostic plots"
        ),
    )
    parser.add_argument(
        "--animation-format",
        choices=("mp4", "gif"),
        default="mp4",
    )
    parser.add_argument(
        "--animation-fps",
        type=float,
        default=None,
        help=(
            "saved animation FPS; default preserves real experiment time "
            "given --frame-stride"
        ),
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=2,
        help="use every N-th trajectory sample in the animation",
    )
    parser.add_argument("--animation-elevation", type=float, default=25.0)
    parser.add_argument("--animation-azimuth", type=float, default=-60.0)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--format",
        choices=("pdf", "png", "svg"),
        default="pdf",
    )
    args = parser.parse_args()

    if args.paper_quality:
        _apply_paper_quality_style()
    if args.frame_stride <= 0:
        parser.error("--frame-stride must be positive")
    if args.animation_fps is not None and args.animation_fps <= 0.0:
        parser.error("--animation-fps must be positive")

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

    if "leader_position" in selected:
        figures["leader_position"] = _leader_position_figure(
            data,
            args.paper_quality,
        )
        if figures["leader_position"] is None:
            print(
                "Skipping leader-position plot: leader reference position "
                "is unavailable."
            )

    if "formation_tracking" in selected:
        formation_figures = _formation_tracking_figures(
            data,
            args.paper_quality,
        )
        figures.update(formation_figures)
        if not formation_figures:
            print(
                "Skipping formation-tracking plots: no finite desired "
                "relative-position histories were found."
            )

    if "workspace" in selected:
        workspace_figures = _workspace_figures(
            data,
            args.paper_quality,
        )
        figures.update(workspace_figures)
        if not workspace_figures:
            print(
                "Skipping workspace plots: Stage-B workspace diagnostics "
                "are not present in this NPZ."
            )

    if "workspace_relaxation" in selected:
        workspace_relaxation_figures = _workspace_relaxation_figures(
            data,
            args.paper_quality,
        )
        figures.update(workspace_relaxation_figures)
        if not workspace_relaxation_figures:
            print(
                "Skipping workspace-relaxation plots: Stage-B workspace "
                "diagnostics are not present in this NPZ."
            )

    if "formation_error" in selected:
        figures["formation_error"] = _formation_error_figure(
            data,
            args.paper_quality,
            args.show_legends,
        )
        if figures["formation_error"] is None:
            print(
                "Skipping formation-error plot: no edge had at least two "
                "finite desired-relative-position samples."
            )

    if "distance" in selected:
        figures["distance"] = _distance_diagnostics_figure(
            data,
            args.paper_quality,
            args.show_legends,
        )

    if "fov" in selected:
        horizontal, vertical = _fov_diagnostics_figures(
            data,
            args.paper_quality,
            args.show_legends,
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
        thruster_figures = _thruster_force_figures(
            data,
            args.paper_quality,
        )
        figures.update(thruster_figures)
        if not thruster_figures:
            print(
                "Skipping thruster-force plots: no finite per-robot thruster "
                "histories were found."
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

    formation_animation = None
    animation_dt = None
    if args.animation:
        formation_animation, animation_dt = _animation(
            data,
            frame_stride=args.frame_stride,
            elevation=args.animation_elevation,
            azimuth=args.animation_azimuth,
        )

    diagnostic_animations: dict[str, tuple[plt.Figure, FuncAnimation]] = {}
    if args.diagnostic_animations:
        diagnostic_animations["formation_error"] = _formation_error_animation(
            data,
            paper_quality=args.paper_quality,
            frame_stride=args.frame_stride,
            show_legends=args.show_legends,
        )
        diagnostic_animations["distance"] = _distance_diagnostics_animation(
            data,
            paper_quality=args.paper_quality,
            frame_stride=args.frame_stride,
            show_legends=args.show_legends,
        )
        diagnostic_animations.update(
            _fov_diagnostics_animations(
                data,
                paper_quality=args.paper_quality,
                frame_stride=args.frame_stride,
                show_legends=args.show_legends,
            )
        )

        control_times = np.asarray(data["arrays"]["times"], dtype=float)
        animation_dt = float(np.median(np.diff(control_times)))

    if args.save:
        output_dir = args.output_dir or args.history.parent / "plots"
        _save_figures(
            figures,
            output_dir,
            args.format,
            args.paper_quality,
        )

        if formation_animation is not None:
            assert animation_dt is not None
            fps = args.animation_fps
            if fps is None:
                fps = 1.0 / (animation_dt * args.frame_stride)
            save_animation(
                formation_animation.animation,
                output_dir / f"formation_animation.{args.animation_format}",
                paper_quality=False,
                fps=fps,
            )
            print(
                "Saved formation animation to "
                f"{output_dir / f'formation_animation.{args.animation_format}'} "
                f"at {fps:.3f} fps"
            )

        if diagnostic_animations:
            assert animation_dt is not None
            fps = args.animation_fps
            if fps is None:
                fps = 1.0 / (animation_dt * args.frame_stride)
            for name, (_, animation) in diagnostic_animations.items():
                path = output_dir / f"{name}_animation.{args.animation_format}"
                save_animation(
                    animation,
                    path,
                    paper_quality=args.paper_quality,
                    fps=fps,
                )
                print(f"Saved {name} animation to {path} at {fps:.3f} fps")

    if args.show:
        plt.show()
    else:
        for figure in figures.values():
            if figure is not None:
                plt.close(figure)
        if formation_animation is not None:
            plt.close(formation_animation.figure)
        for figure, _ in diagnostic_animations.values():
            plt.close(figure)


if __name__ == "__main__":
    main()
