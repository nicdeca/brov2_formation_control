#!/usr/bin/env python3
"""Challenging pure-Python demonstration of sensing-funnel relaxation.

The followers start inside the conservative sensing set with outward linear
velocity and angular-rate disturbances.  The secondary funnel dynamics must
temporarily enlarge the conservative range/FoV domain toward the physical
domain.  The selected demonstration edge crosses its conservative range limit
while remaining strictly inside the adaptive funnel and physical sensor limit.
The funnel then recovers to the robust conservative set after the transient.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import matplotlib.pyplot as plt
import numpy as np

from formation_control.actuation import T200ForceLimits
from formation_control.visualization import (
    BlueROV2HeavyVisualGeometry,
    animate_formation_3d,
    apply_visualization_style,
    plot_formation_3d,
    save_animation,
    save_figure,
)
from formation_control.visualization.domain_relaxation_plot import (
    plot_vertical_fov_relaxation_from_histories,
)

# Endpoint values of the force polynomials in Gazebo's BlueROV2 Heavy SDF.
GAZEBO_THRUSTER_FORCE_LIMITS = T200ForceLimits(
    forward=66.47563091,
    reverse=49.691034942,
)


@dataclass(frozen=True)
class RelaxationRun:
    label: str
    adaptive: bool
    scenario: object
    trajectory: object
    camera: object
    distance_domain: object
    fov_domain: object
    slacks: np.ndarray
    required_slacks: np.ndarray
    actuation_margins: np.ndarray
    rho_history: np.ndarray
    relaxation_rates: np.ndarray
    image_history: np.ndarray
    controller_times: np.ndarray
    clf: object
    allocation: object
    status: object


def load_controller_example() -> ModuleType:
    """Load the shared stress simulation despite its numeric filename."""

    path = Path(__file__).with_name("04_bluerov2_fov_clf_qp.py")
    module_name = "_funnel_relaxation_demo_controller"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def simulate_relaxation_case(
    example: ModuleType,
    *,
    duration: float,
    dt: float,
    stress_scale: float,
) -> RelaxationRun:
    result = example.simulate(
        duration=duration,
        dt=dt,
        thruster_voltage=16,
        thrust_derating=1.0,
        control_space="thruster",
        distance_constraints=True,
        fov_constraints=True,
        adaptive=True,
        stress_test=True,
        stress_scale=stress_scale,
        relaxation_recovery_gain=0.8,
        relaxation_domain_margin_ratio=0.1,
        thruster_force_limits=GAZEBO_THRUSTER_FORCE_LIMITS,
    )
    return RelaxationRun(
        label="adaptive relaxed funnel",
        adaptive=True,
        scenario=result[0],
        trajectory=result[1],
        camera=result[2],
        distance_domain=result[3],
        fov_domain=result[4],
        slacks=result[5],
        required_slacks=result[6],
        actuation_margins=result[7],
        rho_history=result[8],
        relaxation_rates=result[9],
        image_history=result[10],
        controller_times=result[11],
        clf=result[12],
        allocation=result[13],
        status=result[14],
    )


def simulate_demo(
    *,
    duration: float = 10.0,
    dt: float = 0.01,
    stress_scale: float = 1.0,
) -> RelaxationRun:
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration must be finite and positive")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    if not np.isfinite(stress_scale) or stress_scale < 0.0:
        raise ValueError("stress_scale must be finite and nonnegative")

    example = load_controller_example()
    return simulate_relaxation_case(
        example,
        duration=duration,
        dt=dt,
        stress_scale=stress_scale,
    )


def normalized_relaxation(run: RelaxationRun) -> np.ndarray:
    maxima = np.array(
        (
            run.distance_domain.collision_enlargement_max,
            run.distance_domain.range_enlargement_max,
            run.fov_domain.horizontal_enlargement_max,
            run.fov_domain.vertical_enlargement_max,
        )
    )
    return np.divide(
        run.rho_history,
        maxima,
        out=np.zeros_like(run.rho_history),
        where=maxima > 0.0,
    )


def sensing_history_for_edge(
    run: RelaxationRun,
    edge: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    relative = (
        run.trajectory.positions[:, edge.target]
        - run.trajectory.positions[:, edge.observer]
    )
    return (
        np.linalg.norm(relative, axis=1),
        np.abs(run.image_history[:, edge.observer, 0]),
        np.abs(run.image_history[:, edge.observer, 1]),
    )


def select_conservative_violation(
    run: RelaxationRun,
) -> tuple[object, str, np.ndarray, np.ndarray, float, float]:
    """Select the edge/channel with the largest conservative excursion."""

    candidates = []
    for edge in run.scenario.graph:
        distance, horizontal, vertical = sensing_history_for_edge(run, edge)
        channel_data = (
            (
                "range",
                distance,
                run.rho_history[:, edge.observer, 1],
                run.distance_domain.d_max_conservative,
                run.distance_domain.d_max,
            ),
            (
                "horizontal FoV",
                horizontal,
                run.rho_history[:, edge.observer, 2],
                run.fov_domain.alpha_h_conservative,
                1.0,
            ),
            (
                "vertical FoV",
                vertical,
                run.rho_history[:, edge.observer, 3],
                run.fov_domain.alpha_v_conservative,
                1.0,
            ),
        )
        for name, values, rho, conservative, physical in channel_data:
            normalized_excursion = (np.nanmax(values) - conservative) / (
                physical - conservative
            )
            candidates.append(
                (
                    normalized_excursion,
                    edge,
                    name,
                    values,
                    rho,
                    conservative,
                    physical,
                )
            )

    _, edge, name, values, rho, conservative, physical = max(
        candidates,
        key=lambda candidate: candidate[0],
    )
    return edge, name, values, rho, conservative, physical


def effective_upper_limit(
    run: RelaxationRun,
    channel: str,
    rho: np.ndarray,
) -> np.ndarray:
    if channel == "range":
        return np.array(
            [run.distance_domain.effective_maximum_distance(value) for value in rho]
        )
    if channel == "horizontal FoV":
        return np.array(
            [run.fov_domain.effective_horizontal_limit(value) for value in rho]
        )
    return np.array(
        [run.fov_domain.effective_vertical_limit(value) for value in rho]
    )


def maximum_formation_error(run: RelaxationRun) -> np.ndarray:
    histories = []
    for edge in run.scenario.graph:
        actual = (
            run.trajectory.positions[:, edge.target]
            - run.trajectory.positions[:, edge.observer]
        )
        desired = run.scenario.desired_relative_position(edge.observer, edge.target)
        histories.append(np.linalg.norm(actual - desired, axis=1))
    return np.max(np.vstack(histories), axis=0)


def maximum_thruster_utilization(run: RelaxationRun) -> np.ndarray:
    controls = run.trajectory.controls
    utilization = np.empty(controls.shape[:2])
    for step in range(controls.shape[0]):
        for agent in range(controls.shape[1]):
            utilization[step, agent] = np.max(
                run.allocation.utilization(controls[step, agent])
            )
    return np.max(utilization, axis=1)


def _mark_fallbacks(axes: plt.Axes, run: RelaxationRun) -> None:
    """Compatibility helper for the former comparison plotter."""

    first = run.status.first_fallback_time
    if first is not None:
        axes.axvline(first, color="tab:red", linestyle=":")


def _run_diagnostic_plots(
    run: RelaxationRun,
    example: ModuleType,
    *,
    prefix: str,
    paper_quality: bool,
) -> dict[str, plt.Figure | None]:
    """Build the full diagnostic set used by the other pure-Python run."""

    figures: dict[str, plt.Figure | None] = {}
    trajectory = run.trajectory
    scenario = run.scenario
    times = trajectory.times
    leader = scenario.graph.root

    quality = example.connection_quality_history(
        scenario,
        trajectory,
        run.distance_domain,
        run.fov_domain,
        run.image_history,
        distance_constraints=True,
        fov_constraints=True,
    )
    desired_positions = trajectory.positions[-1, leader] + scenario.reference.offsets
    trajectory_plot = plot_formation_3d(
        trajectory,
        scenario.graph,
        desired_positions=desired_positions,
        camera=run.camera,
        camera_agents=tuple(edge.observer for edge in scenario.graph),
        camera_depth=0.75,
        vehicle_geometry=BlueROV2HeavyVisualGeometry(
            thruster_configuration=run.allocation.configuration
        ).wireframe(),
        edge_quality=quality,
        paper_quality=paper_quality,
        show_body_forward=False,
        title=run.label,
    )
    figures[f"{prefix}_trajectory_3d"] = (
        trajectory_plot[0] if isinstance(trajectory_plot, tuple) else trajectory_plot
    )

    leader_reference = np.broadcast_to(
        trajectory.positions[0, leader],
        trajectory.positions[:, leader].shape,
    )
    leader_tracking, tracking_axes = plt.subplots(2, 1, sharex=True)
    tracking_axes[0].plot(
        times,
        np.linalg.norm(trajectory.positions[:, leader] - leader_reference, axis=1),
    )
    tracking_axes[0].set(ylabel="position error [m]", title=f"Leader hold: {run.label}")
    tracking_axes[0].grid(True, alpha=0.3)
    tracking_axes[1].plot(
        times,
        np.linalg.norm(trajectory.velocities[:, leader], axis=1),
    )
    tracking_axes[1].set(xlabel="time [s]", ylabel="velocity norm [m/s]")
    tracking_axes[1].grid(True, alpha=0.3)
    leader_tracking.tight_layout()
    figures[f"{prefix}_leader_tracking"] = leader_tracking

    leader_position, position_axes = plt.subplots(3, 1, sharex=True)
    labels = ("x", "y", "z")
    for component, axes in enumerate(position_axes):
        axes.plot(times, trajectory.positions[:, leader, component], label="actual")
        axes.plot(times, leader_reference[:, component], "--", label="reference")
        axes.set_ylabel(f"{labels[component]} [m]")
        axes.grid(True, alpha=0.3)
    position_axes[0].set_title(f"Leader position: {run.label}")
    position_axes[0].legend()
    position_axes[-1].set_xlabel("time [s]")
    leader_position.tight_layout()
    figures[f"{prefix}_leader_position"] = leader_position

    formation_error, error_axes = plt.subplots()
    for edge in scenario.graph:
        actual = trajectory.positions[:, edge.target] - trajectory.positions[:, edge.observer]
        desired = scenario.desired_relative_position(edge.observer, edge.target)
        error_axes.plot(
            times,
            np.linalg.norm(actual - desired, axis=1),
            label=f"agent {edge.observer}→{edge.target}",
        )
    error_axes.set(
        xlabel="time [s]",
        ylabel="formation error [m]",
        title=f"Formation error: {run.label}",
    )
    error_axes.grid(True, alpha=0.3)
    error_axes.legend()
    formation_error.tight_layout()
    figures[f"{prefix}_formation_error"] = formation_error

    for edge in scenario.graph:
        actual = trajectory.positions[:, edge.target] - trajectory.positions[:, edge.observer]
        desired = scenario.desired_relative_position(edge.observer, edge.target)
        edge_figure, edge_axes = plt.subplots(3, 1, sharex=True)
        for component, axes in enumerate(edge_axes):
            axes.plot(times, actual[:, component], label="actual")
            axes.axhline(desired[component], linestyle="--", label="desired")
            axes.set_ylabel(f"d{labels[component]} [m]")
            axes.grid(True, alpha=0.3)
        edge_axes[0].set_title(
            f"Formation tracking {edge.observer}→{edge.target}: {run.label}"
        )
        edge_axes[0].legend()
        edge_axes[-1].set_xlabel("time [s]")
        edge_figure.tight_layout()
        figures[f"{prefix}_formation_tracking_{edge.observer}_to_{edge.target}"] = edge_figure

    figures[f"{prefix}_distance"] = example.plot_distance_diagnostics(
        scenario,
        trajectory,
        run.distance_domain,
        run.rho_history,
        adaptive=run.adaptive,
    )
    horizontal, vertical = example.plot_fov_diagnostics(
        scenario,
        trajectory,
        run.fov_domain,
        run.rho_history,
        run.image_history,
        adaptive=run.adaptive,
    )
    figures[f"{prefix}_fov_horizontal"] = horizontal
    figures[f"{prefix}_fov_vertical"] = vertical

    if run.adaptive:
        s_history = normalized_relaxation(run)
        adaptive_fov, _ = plot_vertical_fov_relaxation_from_histories(
            times,
            run.image_history,
            s_history,
            observer=scenario.graph.edges[0].observer,
            alpha_conservative=run.fov_domain.alpha_v_conservative,
            alpha_physical=1.0,
            domain_margin_ratio=0.1,
            paper_quality=paper_quality,
        )
        figures[f"{prefix}_adaptive_fov_vertical"] = adaptive_fov
        figures[f"{prefix}_domain_enlargement"] = example.plot_domain_enlargement(
            scenario,
            trajectory,
            run.distance_domain,
            run.fov_domain,
            run.rho_history,
        )
        figures[f"{prefix}_relaxation_rates"] = example.plot_relaxation_rates(
            scenario,
            trajectory,
            run.relaxation_rates,
        )

    figures[f"{prefix}_slack"] = example.plot_slack(
        scenario,
        trajectory,
        run.slacks,
        run.required_slacks,
    )
    figures[f"{prefix}_actuation_margin"] = example.plot_actuation_margin(
        scenario,
        trajectory,
        run.actuation_margins,
    )

    control_times = times[:-1]
    for agent in range(trajectory.n_agents):
        thruster_figure, thruster_axes = plt.subplots()
        for thruster, name in enumerate(run.allocation.configuration.names):
            thruster_axes.plot(
                control_times,
                trajectory.controls[:, agent, thruster],
                label=name,
            )
        thruster_axes.axhline(
            run.allocation.configuration.force_limits.forward,
            linestyle="--",
            label="forward limit",
        )
        thruster_axes.axhline(
            -run.allocation.configuration.force_limits.reverse,
            linestyle="--",
            label="reverse limit",
        )
        thruster_axes.set(
            xlabel="time [s]",
            ylabel="thruster force [N]",
            title=f"Thruster forces, agent {agent}: {run.label}",
        )
        thruster_axes.grid(True, alpha=0.3)
        thruster_axes.legend(ncol=2)
        thruster_figure.tight_layout()
        figures[f"{prefix}_thruster_forces_agent_{agent}"] = thruster_figure

    figures[f"{prefix}_controller_time"] = example.plot_controller_times(
        scenario,
        trajectory,
        run.controller_times,
        control_space="thruster",
    )
    figures[f"{prefix}_clf_value_decay"] = example.plot_clf_value_and_decay(
        scenario, trajectory, run.clf
    )
    figures[f"{prefix}_clf_feasibility_balance"] = example.plot_clf_feasibility_balance(
        scenario, trajectory, run.clf
    )
    figures[f"{prefix}_clf_drift_components"] = example.plot_clf_drift_components(
        scenario, trajectory, run.clf
    )
    figures[f"{prefix}_velocity_backstepping_split"] = (
        example.plot_velocity_backstepping_split(scenario, trajectory, run.clf)
    )
    figures[f"{prefix}_velocity_backstepping_peak"] = (
        example.plot_velocity_backstepping_peak_detail(
            trajectory, run.required_slacks, run.clf
        )
    )
    figures[f"{prefix}_command_velocity_peak"] = example.plot_command_velocity_peak_detail(
        trajectory, run.required_slacks, run.clf
    )
    figures[f"{prefix}_command_acceleration_peak"] = (
        example.plot_command_acceleration_peak_detail(
            trajectory, run.required_slacks, run.clf
        )
    )
    figures[f"{prefix}_clf_peak_detail"] = example.plot_clf_peak_detail(
        scenario,
        trajectory,
        run.required_slacks,
        run.clf,
    )
    return figures


def plot_comparison(
    result: RelaxationRun,
    *,
    paper_quality: bool = False,
) -> dict[str, plt.Figure | None]:
    fixed = result.fixed
    relaxed = result.relaxed
    plt.rcParams["figure.max_open_warning"] = 0
    figures: dict[str, plt.Figure | None] = {}

    trajectories = plt.figure(figsize=(11, 5))
    for panel, run in enumerate((fixed, relaxed), start=1):
        axes = trajectories.add_subplot(1, 2, panel, projection="3d")
        for agent in range(run.trajectory.n_agents):
            axes.plot(*run.trajectory.positions[:, agent].T, label=f"agent {agent}")
            axes.scatter(*run.trajectory.positions[0, agent], s=18)
        axes.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]", title=run.label)
        if panel == 1:
            axes.legend()
    trajectories.suptitle("Identical sensing stress with and without funnel relaxation")
    trajectories.tight_layout()
    figures["trajectory_comparison"] = trajectories

    sensing, sensing_axes = plt.subplots(3, 1, sharex=True)
    labels = (
        (
            "distance [m]",
            fixed.distance_domain.d_max_conservative,
            fixed.distance_domain.d_max,
        ),
        (r"$|\alpha_h|$", fixed.fov_domain.alpha_h_conservative, 1.0),
        (r"$|\alpha_v|$", fixed.fov_domain.alpha_v_conservative, 1.0),
    )
    for channel, axes in enumerate(sensing_axes):
        for edge_index, edge in enumerate(fixed.scenario.graph):
            color = f"C{edge_index}"
            fixed_values = sensing_history_for_edge(fixed, edge)[channel]
            relaxed_values = sensing_history_for_edge(relaxed, edge)[channel]
            edge_label = f"agent {edge.observer}→{edge.target}"
            axes.plot(
                fixed.trajectory.times,
                fixed_values,
                color=color,
                linestyle=":",
                label=f"fixed, {edge_label}",
            )
            axes.plot(
                relaxed.trajectory.times,
                relaxed_values,
                color=color,
                label=f"relaxed, {edge_label}",
            )
        axes.axhline(
            labels[channel][1],
            color="tab:orange",
            linestyle="--",
            label="conservative limit",
        )
        axes.axhline(labels[channel][2], color="tab:red", linestyle=":", label="physical limit")
        axes.set_ylabel(labels[channel][0])
        axes.grid(True, alpha=0.3)
        _mark_fallbacks(axes, fixed)
    sensing_axes[-1].set_xlabel("time [s]")
    sensing.suptitle(
        "Sensing coordinate of each follower→parent edge\n"
        "solid: relaxed funnel; dotted: fixed conservative funnel"
    )
    handles, legend_labels = sensing_axes[0].get_legend_handles_labels()
    sensing.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=4,
    )
    sensing.tight_layout(rect=(0.0, 0.12, 1.0, 1.0))
    figures["sensing_domain_comparison"] = sensing

    edge, channel, relaxed_values, rho, conservative, physical = (
        select_conservative_violation(relaxed)
    )
    fixed_edge = next(
        candidate
        for candidate in fixed.scenario.graph
        if candidate.observer == edge.observer and candidate.target == edge.target
    )
    channel_index = {"range": 0, "horizontal FoV": 1, "vertical FoV": 2}[channel]
    fixed_values = sensing_history_for_edge(fixed, fixed_edge)[channel_index]
    adaptive_limit = effective_upper_limit(relaxed, channel, rho)
    violation, (coordinate_axes, margin_axes) = plt.subplots(2, 1, sharex=True)
    coordinate_axes.plot(
        fixed.trajectory.times,
        fixed_values,
        linestyle=":",
        label="fixed funnel",
    )
    coordinate_axes.plot(
        relaxed.trajectory.times,
        relaxed_values,
        label="relaxed funnel",
    )
    coordinate_axes.plot(
        relaxed.trajectory.times,
        adaptive_limit,
        "--",
        label="adaptive limit",
    )
    coordinate_axes.axhline(
        conservative,
        color="tab:orange",
        linestyle="--",
        label="conservative limit",
    )
    coordinate_axes.axhline(
        physical,
        color="tab:red",
        linestyle=":",
        label="physical limit",
    )
    coordinate_axes.fill_between(
        relaxed.trajectory.times,
        conservative,
        relaxed_values,
        where=relaxed_values > conservative,
        color="tab:orange",
        alpha=0.25,
        label="conservative violation",
    )
    _mark_fallbacks(coordinate_axes, fixed)
    coordinate_axes.set_ylabel("distance [m]" if channel == "range" else r"$|\alpha|$")
    coordinate_axes.grid(True, alpha=0.3)
    coordinate_axes.legend(ncol=2)

    margin_axes.plot(
        relaxed.trajectory.times,
        conservative - relaxed_values,
        label="conservative margin",
    )
    margin_axes.plot(
        relaxed.trajectory.times,
        adaptive_limit - relaxed_values,
        "--",
        label="adaptive margin",
    )
    margin_axes.plot(
        relaxed.trajectory.times,
        physical - relaxed_values,
        ":",
        label="physical margin",
    )
    margin_axes.axhline(0.0, color="black", linewidth=0.8)
    margin_axes.fill_between(
        relaxed.trajectory.times,
        conservative - relaxed_values,
        0.0,
        where=relaxed_values > conservative,
        color="tab:orange",
        alpha=0.25,
    )
    margin_axes.set(xlabel="time [s]", ylabel="signed margin")
    margin_axes.grid(True, alpha=0.3)
    margin_axes.legend(ncol=3)
    violation.suptitle(
        f"Explicit conservative {channel} violation: "
        f"agent {edge.observer}→{edge.target}"
    )
    violation.tight_layout()
    figures["conservative_constraint_violation"] = violation

    relaxation_figure, relaxation_axes = plt.subplots()
    s_history = normalized_relaxation(relaxed)
    channel_names = ("collision", "range", "horizontal FoV", "vertical FoV")
    followers = [edge.observer for edge in relaxed.scenario.graph]
    for channel, channel_name in enumerate(channel_names):
        relaxation_axes.plot(
            relaxed.trajectory.times,
            np.max(s_history[:, followers, channel], axis=1),
            label=channel_name,
        )
    relaxation_axes.set(
        xlabel="time [s]",
        ylabel="maximum normalized relaxation",
        title="Adaptive funnel enlargement and recovery",
        ylim=(-0.02, 1.02),
    )
    relaxation_axes.grid(True, alpha=0.3)
    relaxation_axes.legend()
    figures["funnel_relaxation"] = relaxation_figure

    tracking, tracking_axes = plt.subplots()
    tracking_axes.plot(
        fixed.trajectory.times,
        maximum_formation_error(fixed),
        label=fixed.label,
    )
    tracking_axes.plot(
        relaxed.trajectory.times,
        maximum_formation_error(relaxed),
        label=relaxed.label,
    )
    _mark_fallbacks(tracking_axes, fixed)
    tracking_axes.set(
        xlabel="time [s]",
        ylabel="maximum formation error [m]",
        title="Formation tracking consequence of fixed-funnel fallback",
    )
    tracking_axes.grid(True, alpha=0.3)
    tracking_axes.legend()
    tracking.tight_layout()
    figures["formation_error_comparison"] = tracking

    effort, effort_axes = plt.subplots()
    effort_axes.plot(
        fixed.trajectory.times[:-1],
        maximum_thruster_utilization(fixed),
        label=fixed.label,
    )
    effort_axes.plot(
        relaxed.trajectory.times[:-1],
        maximum_thruster_utilization(relaxed),
        label=relaxed.label,
    )
    effort_axes.axhline(1.0, color="tab:red", linestyle=":", label="thruster limit")
    _mark_fallbacks(effort_axes, fixed)
    effort_axes.set(
        xlabel="time [s]",
        ylabel="maximum normalized thruster utilization",
        title="Control effort during the sensing transient",
    )
    effort_axes.grid(True, alpha=0.3)
    effort_axes.legend()
    effort.tight_layout()
    figures["thruster_utilization_comparison"] = effort
    example = load_controller_example()
    figures.update(
        _run_diagnostic_plots(
            fixed,
            example,
            prefix="fixed",
            paper_quality=paper_quality,
        )
    )
    figures.update(
        _run_diagnostic_plots(
            relaxed,
            example,
            prefix="relaxed",
            paper_quality=paper_quality,
        )
    )
    return figures


def plot_demo(
    run: RelaxationRun,
    *,
    paper_quality: bool = False,
) -> dict[str, plt.Figure | None]:
    """Plot one challenging run without a fixed/no-relaxation comparison."""

    plt.rcParams["figure.max_open_warning"] = 0
    figures: dict[str, plt.Figure | None] = {}
    times = run.trajectory.times

    sensing, sensing_axes = plt.subplots(3, 1, sharex=True)
    labels = (
        ("distance [m]", run.distance_domain.d_max_conservative, run.distance_domain.d_max),
        (r"$|\alpha_h|$", run.fov_domain.alpha_h_conservative, 1.0),
        (r"$|\alpha_v|$", run.fov_domain.alpha_v_conservative, 1.0),
    )
    for channel_index, axes in enumerate(sensing_axes):
        for edge in run.scenario.graph:
            values = sensing_history_for_edge(run, edge)[channel_index]
            axes.plot(times, values, label=f"agent {edge.observer}→{edge.target}")
        axes.axhline(
            labels[channel_index][1],
            color="tab:orange",
            linestyle="--",
            label="conservative limit",
        )
        axes.axhline(
            labels[channel_index][2],
            color="tab:red",
            linestyle=":",
            label="physical limit",
        )
        axes.set_ylabel(labels[channel_index][0])
        axes.grid(True, alpha=0.3)
    sensing_axes[0].legend(ncol=2)
    sensing_axes[-1].set_xlabel("time [s]")
    sensing.suptitle("Sensing coordinate of each follower→parent edge")
    sensing.tight_layout()
    figures["sensing_coordinates"] = sensing

    edge, channel, values, rho, conservative, physical = select_conservative_violation(run)
    adaptive_limit = effective_upper_limit(run, channel, rho)
    violation, (coordinate_axes, margin_axes) = plt.subplots(2, 1, sharex=True)
    coordinate_axes.plot(times, values, label=f"agent {edge.observer}→{edge.target}")
    coordinate_axes.plot(times, adaptive_limit, "--", label="adaptive funnel limit")
    coordinate_axes.axhline(
        conservative,
        color="tab:orange",
        linestyle="--",
        label="conservative limit",
    )
    coordinate_axes.axhline(
        physical,
        color="tab:red",
        linestyle=":",
        label="physical limit",
    )
    coordinate_axes.fill_between(
        times,
        conservative,
        values,
        where=values > conservative,
        color="tab:orange",
        alpha=0.25,
        label="outside conservative set",
    )
    coordinate_axes.set_ylabel("distance [m]" if channel == "range" else r"$|\alpha|$")
    coordinate_axes.grid(True, alpha=0.3)
    coordinate_axes.legend(ncol=2)

    conservative_margin = conservative - values
    adaptive_margin = adaptive_limit - values
    physical_margin = physical - values
    margin_axes.plot(times, conservative_margin, label="conservative margin")
    margin_axes.plot(times, adaptive_margin, "--", label="adaptive-funnel margin")
    margin_axes.plot(times, physical_margin, ":", label="physical margin")
    margin_axes.axhline(0.0, color="black", linewidth=0.8)
    margin_axes.fill_between(
        times,
        conservative_margin,
        0.0,
        where=conservative_margin < 0.0,
        color="tab:orange",
        alpha=0.25,
    )
    margin_axes.set(xlabel="time [s]", ylabel="signed margin")
    margin_axes.grid(True, alpha=0.3)
    margin_axes.legend(ncol=3)
    violation.suptitle(
        f"Conservative {channel} violation inside the adaptive funnel: "
        f"agent {edge.observer}→{edge.target}"
    )
    violation.tight_layout()
    figures["conservative_constraint_violation"] = violation

    relaxation_figure, relaxation_axes = plt.subplots()
    s_history = normalized_relaxation(run)
    followers = [edge.observer for edge in run.scenario.graph]
    channel_names = ("collision", "range", "horizontal FoV", "vertical FoV")
    for channel_index, channel_name in enumerate(channel_names):
        relaxation_axes.plot(
            times,
            np.max(s_history[:, followers, channel_index], axis=1),
            label=channel_name,
        )
    relaxation_axes.set(
        xlabel="time [s]",
        ylabel="maximum normalized relaxation",
        title="Adaptive funnel enlargement and recovery",
        ylim=(-0.02, 1.02),
    )
    relaxation_axes.grid(True, alpha=0.3)
    relaxation_axes.legend()
    relaxation_figure.tight_layout()
    figures["funnel_relaxation"] = relaxation_figure

    figures.update(
        _run_diagnostic_plots(
            run,
            load_controller_example(),
            prefix="adaptive",
            paper_quality=paper_quality,
        )
    )
    return figures


def print_summary(run: RelaxationRun) -> None:
    s_max = float(np.max(normalized_relaxation(run)))
    fallback = (
        f"fallback at {run.status.first_fallback_time:.3f} s"
        if run.status.fallback_occurred
        else "no fallback"
    )
    print(
        f"{run.label}: {fallback}; maximum normalized relaxation {s_max:.3f}; "
        f"peak formation error {np.max(maximum_formation_error(run)):.3f} m"
    )
    edge, channel, values, rho, conservative, physical = select_conservative_violation(run)
    adaptive_limit = effective_upper_limit(run, channel, rho)
    print(
        f"Focused violation: agent {edge.observer}->{edge.target} {channel}, "
        f"peak coordinate {np.max(values):.3f}, conservative limit "
        f"{conservative:.3f}, physical limit {physical:.3f}, minimum adaptive "
        f"margin {np.min(adaptive_limit - values):.3f}"
    )


def save_histories(run: RelaxationRun, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "funnel_relaxation_demo.npz"
    np.savez_compressed(
        output,
        times=run.trajectory.times,
        positions=run.trajectory.positions,
        velocities=run.trajectory.velocities,
        quaternions=run.trajectory.quaternions,
        controls=run.trajectory.controls,
        slacks=run.slacks,
        required_slacks=run.required_slacks,
        actuation_margins=run.actuation_margins,
        rho=run.rho_history,
        relaxation_rates=run.relaxation_rates,
        normalized_relaxation=normalized_relaxation(run),
        image_coordinates=run.image_history,
        controller_times=run.controller_times,
        fallback_times=run.status.fallback_times,
        **{
            f"clf_{name}": value
            for name, value in vars(run.clf).items()
        },
    )
    return output


def build_animations(
    run: RelaxationRun,
    *,
    frame_stride: int,
    paper_quality: bool,
) -> dict[str, object]:
    geometry = BlueROV2HeavyVisualGeometry(
        thruster_configuration=run.allocation.configuration
    ).wireframe()
    return {
        "funnel_relaxation": animate_formation_3d(
            run.trajectory,
            run.scenario.graph,
            camera=run.camera,
            camera_agents=tuple(edge.observer for edge in run.scenario.graph),
            camera_depth=0.75,
            vehicle_geometry=geometry,
            paper_quality=paper_quality,
            show_body_forward=False,
            trail_length=150,
            frame_stride=frame_stride,
            title="Challenging adaptive-funnel relaxation",
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--stress-scale", type=float, default=1.0)
    parser.add_argument("--paper-quality", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--animate", action="store_true")
    parser.add_argument("--save-animation", action="store_true")
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--animation-format", choices=("gif", "mp4"), default="mp4")
    parser.add_argument("--format", choices=("png", "pdf", "svg"), default="png")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/bluerov2_funnel_relaxation_demo"),
    )
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    if args.duration <= 0.0:
        parser.error("--duration must be positive")
    if args.dt <= 0.0:
        parser.error("--dt must be positive")
    if args.stress_scale < 0.0:
        parser.error("--stress-scale must be nonnegative")
    if args.frame_stride <= 0:
        parser.error("--frame-stride must be positive")

    apply_visualization_style(paper_quality=args.paper_quality)
    result = simulate_demo(
        duration=args.duration,
        dt=args.dt,
        stress_scale=args.stress_scale,
    )
    print_summary(result)
    figures = plot_demo(result, paper_quality=args.paper_quality)

    animations = {}
    if args.animate or args.save_animation:
        animations = build_animations(
            result,
            frame_stride=args.frame_stride,
            paper_quality=args.paper_quality,
        )

    if args.save:
        history_path = save_histories(result, args.output_dir)
        saved_figures = 0
        for name, figure in figures.items():
            if figure is None:
                continue
            save_figure(
                figure,
                args.output_dir / f"{name}.{args.format}",
                paper_quality=args.paper_quality,
            )
            saved_figures += 1
        print(f"Saved histories to {history_path} and {saved_figures} figures")

    if args.save_animation:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, animation in animations.items():
            output = args.output_dir / f"{name}.{args.animation_format}"
            save_animation(
                animation.animation,
                output,
                paper_quality=args.paper_quality,
            )
            print(f"Saved animation to {output}")

    if not args.no_show:
        plt.show()
    else:
        for figure in figures.values():
            if figure is not None:
                plt.close(figure)


if __name__ == "__main__":
    main()
