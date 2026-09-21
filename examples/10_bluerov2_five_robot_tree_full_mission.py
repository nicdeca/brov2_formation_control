#!/usr/bin/env python3
"""Pure-Python counterpart of the five-robot ROS/Gazebo full experiment.

This example reproduces the command sequence executed by
``scripts/run_five_robot_tree_experiment.py --leader itrl_rov_1 --profile full``:
the nominal, wide, compact, staggered, and final nominal formations; the four
leader-velocity pulses; and all publication, stop-command, and settling times.

There is deliberately no water-tank or workspace model.  The nominal launch
geometry is translated near the origin, while relative positions, controller
settings, BlueROV2 Gazebo dynamics, sensing constraints, and mission timing
remain the same as in the five-robot simulation launch.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from types import ModuleType

import matplotlib.pyplot as plt
import numpy as np
from moving_formation_validation import (
    build_barrier_templates,
    build_camera,
    build_domains,
    build_edge_potential,
    build_relaxation_policy,
    inertial_linear_velocity,
    look_at_quaternion,
)

from formation_control.actuation import (
    BlueROV2HeavyThrusterAllocation,
    T200ForceLimits,
)
from formation_control.constraints import evaluate_sensing_constraint_kinematics
from formation_control.control import build_bluerov2_controller_design
from formation_control.control.bluerov2_leader import BlueROV2LeaderController
from formation_control.geometry import (
    quaternion_from_roll_pitch_yaw,
    rotation_matrix_from_quaternion,
)
from formation_control.graphs import DirectedSensingGraph
from formation_control.models import BlueROV2Model
from formation_control.simulation import (
    FormationReference,
    FormationScenario,
    FormationTrajectory,
    RK4Integrator,
)
from formation_control.simulation.leader_references import VelocityCommandReferenceFilter
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

ROBOT_NAMES = tuple(f"itrl_rov_{index}" for index in range(1, 6))
EDGE_LABELS = ("2→1", "3→1", "4→2", "5→3")

# Endpoint values of the force polynomials in Gazebo's BlueROV2 Heavy SDF.
GAZEBO_THRUSTER_FORCE_LIMITS = T200ForceLimits(
    forward=66.47563091,
    reverse=49.691034942,
)


@dataclass(frozen=True, slots=True)
class MissionPhase:
    """One constant-command interval in the ROS ``full`` profile."""

    label: str
    duration: float
    formation: str
    velocity_command: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class FiveRobotMissionResult:
    trajectory: FormationTrajectory
    scenario: FormationScenario
    leader_reference_position: np.ndarray
    leader_reference_velocity: np.ndarray
    velocity_commands: np.ndarray
    desired_edge_relative_positions: np.ndarray
    edge_errors: np.ndarray
    normalized_relaxation: np.ndarray
    relaxation_enlargement: np.ndarray
    relaxation_rates: np.ndarray
    image_history: np.ndarray
    slacks: np.ndarray
    required_slacks: np.ndarray
    actuation_margins: np.ndarray
    controller_times: np.ndarray
    clf: CLFDiagnosticHistory
    camera: object
    distance_domain: object
    fov_domain: object
    phase_indices: np.ndarray
    phases: tuple[MissionPhase, ...]
    allocation: BlueROV2HeavyThrusterAllocation


@dataclass(frozen=True)
class CLFDiagnosticHistory:
    """Histories consumed by the common BlueROV2 diagnostic plotters."""

    value: np.ndarray
    decay: np.ndarray
    drift: np.ndarray
    configuration_local_rate: np.ndarray
    parent_rate: np.ndarray
    velocity_backstepping_rate: np.ndarray
    dynamics_bias_rate: np.ndarray
    dynamics_bias_linear_rate: np.ndarray
    dynamics_bias_angular_rate: np.ndarray
    command_acceleration_rate: np.ndarray
    command_acceleration_linear_rate: np.ndarray
    command_acceleration_angular_rate: np.ndarray
    velocity_error_norm: np.ndarray
    velocity_error_linear_norm: np.ndarray
    velocity_error_angular_norm: np.ndarray
    filtered_velocity_derivative_norm: np.ndarray
    filtered_linear_acceleration_norm: np.ndarray
    filtered_angular_acceleration_norm: np.ndarray
    generalized_velocity: np.ndarray
    filtered_velocity: np.ndarray
    desired_velocity: np.ndarray
    unlimited_desired_velocity: np.ndarray
    filtered_velocity_derivative: np.ndarray
    dynamics_bias: np.ndarray
    best_actuator_contribution: np.ndarray
    minimum_modeled_derivative: np.ndarray
    hard_clf_residual: np.ndarray


def formation_offsets() -> dict[str, np.ndarray]:
    """Return root-relative offsets equivalent to the ROS edge references."""

    return {
        "tree_nominal": np.array(
            (
                (0.0, 0.0, 0.0),
                (0.70, 1.65, 0.0),
                (-0.70, 1.65, 0.0),
                (1.20, 3.10, 0.0),
                (-1.20, 3.10, 0.0),
            )
        ),
        "tree_wide": np.array(
            (
                (0.0, 0.0, 0.0),
                (0.80, 1.95, 0.0),
                (-0.80, 1.95, 0.0),
                (1.35, 3.60, 0.0),
                (-1.35, 3.60, 0.0),
            )
        ),
        "tree_compact": np.array(
            (
                (0.0, 0.0, 0.0),
                (0.50, 1.30, 0.0),
                (-0.50, 1.30, 0.0),
                (0.85, 2.40, 0.0),
                (-0.85, 2.40, 0.0),
            )
        ),
        "tree_staggered": np.array(
            (
                (0.0, 0.0, 0.0),
                (0.70, 1.65, -0.10),
                (-0.70, 1.65, 0.10),
                (1.20, 3.10, 0.0),
                (-1.20, 3.10, 0.0),
            )
        ),
    }


def build_five_robot_graph() -> DirectedSensingGraph:
    """Build ``4 -> 2 -> 1`` and ``5 -> 3 -> 1`` (zero-based internally)."""

    return DirectedSensingGraph.from_edges(
        5,
        ((1, 0), (2, 0), (3, 1), (4, 2)),
        root=0,
    ).require_rooted_tree()


def full_mission_phases() -> tuple[MissionPhase, ...]:
    """Return the exact 121.5 s command timeline of the ROS full profile."""

    phases: list[MissionPhase] = []
    formation = "tree_nominal"

    def change_formation(name: str, settle: float) -> None:
        nonlocal formation
        formation = name
        phases.append(MissionPhase(f"command {name}", 2.0, formation))
        phases.append(MissionPhase(f"settle {name}", settle, formation))

    def move(label: str, command: tuple[float, float, float], duration: float) -> None:
        phases.append(MissionPhase(label, duration, formation, command))
        # ``stop_leader`` publishes ten zero commands at 20 Hz.
        phases.append(MissionPhase(f"stop after {label}", 0.5, formation))
        phases.append(MissionPhase(f"settle after {label}", 10.0, formation))

    change_formation("tree_nominal", 8.0)
    change_formation("tree_wide", 12.0)
    change_formation("tree_compact", 10.0)
    move("leader +y", (0.0, 0.25, 0.0), 4.0)
    move("leader +x", (0.18, 0.0, 0.0), 3.5)
    move("leader -x", (-0.18, 0.0, 0.0), 3.5)
    move("leader -y", (0.0, -0.25, 0.0), 4.0)
    change_formation("tree_staggered", 12.0)
    change_formation("tree_nominal", 12.0)
    phases.append(MissionPhase("final stop", 0.5, formation))
    return tuple(phases)


def mission_duration(phases: tuple[MissionPhase, ...]) -> float:
    return float(sum(phase.duration for phase in phases))


def phase_boundaries(phases: tuple[MissionPhase, ...]) -> np.ndarray:
    return np.cumsum([phase.duration for phase in phases], dtype=float)


def phase_index_at(time: float, boundaries: np.ndarray) -> int:
    return min(int(np.searchsorted(boundaries, time, side="right")), len(boundaries) - 1)


def build_scenario(formation: str = "tree_nominal") -> FormationScenario:
    graph = build_five_robot_graph()
    offsets = formation_offsets()[formation]
    root_position = np.array([0.0, 0.0, -1.5])
    initial_positions = root_position + formation_offsets()["tree_nominal"]
    return FormationScenario(
        graph=graph,
        reference=FormationReference(offsets=offsets),
        initial_positions=initial_positions,
    )


def build_initial_states(scenario: FormationScenario) -> np.ndarray:
    states = np.zeros((scenario.n_agents, 13), dtype=float)
    leader = scenario.graph.root
    states[leader, :3] = scenario.initial_positions[leader]
    states[leader, 3:7] = quaternion_from_roll_pitch_yaw(0.0, 0.0, 0.0)

    for edge in scenario.graph:
        observer = edge.observer
        target = edge.target
        states[observer, :3] = scenario.initial_positions[observer]
        states[observer, 3:7] = look_at_quaternion(
            scenario.initial_positions[target] - scenario.initial_positions[observer]
        )
    return states


def scenario_with_formation(scenario: FormationScenario, name: str) -> FormationScenario:
    return FormationScenario(
        graph=scenario.graph,
        reference=FormationReference(offsets=formation_offsets()[name]),
        initial_positions=scenario.initial_positions,
    )


def simulate_full_mission(*, dt: float = 0.02) -> FiveRobotMissionResult:
    """Simulate the full ROS mission with the pure-Python RK4 plant."""

    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")

    phases = full_mission_phases()
    boundaries = phase_boundaries(phases)
    duration = mission_duration(phases)
    # Include every command boundary explicitly even when ``dt`` does not
    # divide a phase duration, so no integration step straddles two commands.
    regular_times = np.arange(0.0, duration, dt, dtype=float)
    times = np.unique(np.concatenate((regular_times, boundaries, (duration,))))
    steps = times.size - 1

    scenario = build_scenario()
    active_formation = phases[0].formation
    states = build_initial_states(scenario)
    model = BlueROV2Model()  # Gazebo parameters, matching the simulation launch.
    default_allocation = BlueROV2HeavyThrusterAllocation.default_45deg(
        voltage=16,
        derating=1.0,
    )
    allocation = BlueROV2HeavyThrusterAllocation(
        configuration=replace(
            default_allocation.configuration,
            force_limits=GAZEBO_THRUSTER_FORCE_LIMITS,
        )
    )

    design = build_bluerov2_controller_design(
        model,
        allocation,
        control_space="thruster",
        virtual_gain=np.diag([1.0, 1.0, 1.0, 1.2, 1.2, 1.2]),
        virtual_velocity_norm_limits=np.array([1.5, 2.0]),
        filter_bandwidth=np.array([3.0, 3.0, 3.0, 4.0, 4.0, 4.0]),
        alpha_gain=1.5,
        slack_linear_penalty=100.0,
        slack_penalty=5e3,
    )
    follower_controller = design.agent_controller

    leader = scenario.graph.root
    leader_controller = BlueROV2LeaderController.from_initial_state(
        follower_controller.dynamics_controller,
        model,
        states[leader],
        position_gain=np.diag([2.0, 2.0, 2.0]),
        attitude_gain=1.0,
    )
    reference_model = VelocityCommandReferenceFilter(bandwidth=0.2)
    reference_state = reference_model.initialize(states[leader, :3])
    initial_reference = reference_model.evaluate(reference_state, np.zeros(3))
    leader_filter = leader_controller.initialize_filter(
        state=states[leader],
        reference=initial_reference,
    )

    camera = build_camera()
    distance_domain, fov_domain = build_domains()
    enabled_relaxation = np.ones(4, dtype=bool)
    relaxation_state = np.zeros((scenario.n_agents, 4), dtype=float)

    def build_templates() -> dict[int, object]:
        return {
            edge.observer: build_barrier_templates(
                scenario, distance_domain, fov_domain, edge.observer, edge.target
            )
            for edge in scenario.graph
        }

    templates = build_templates()

    def build_policies() -> dict[int, object]:
        return {
            edge.observer: build_relaxation_policy(
                distance_domain,
                fov_domain,
                scenario.desired_relative_position(edge.observer, edge.target),
                recovery_gain=0.8,
                barrier_gain=0.20,
                activation_on_ratio=0.10,
                activation_off_ratio=0.30,
                infeasibility_epsilon=1e-3,
            )
            for edge in scenario.graph
        }

    relaxation_policies = build_policies()
    follower_filters: dict[int, np.ndarray] = {}
    for edge in scenario.graph:
        kinematics = evaluate_sensing_constraint_kinematics(
            model,
            camera,
            distance_domain,
            fov_domain,
            states[edge.observer],
            states[edge.target],
        )
        relaxation_state[edge.observer] = relaxation_policies[
            edge.observer
        ].initialize_for_constraint_values(
            kinematics.values,
            enabled=enabled_relaxation,
        )
        potential = build_edge_potential(
            scenario,
            camera,
            distance_domain,
            fov_domain,
            edge.observer,
            edge.target,
            templates[edge.observer],
            relaxation_policies[edge.observer].enlargement(
                relaxation_state[edge.observer]
            ),
            adaptive=True,
            formation_gain=2.0,
        )
        follower_filters[edge.observer] = follower_controller.initialize_filter(
            follower_state=states[edge.observer],
            parent_position=states[edge.target, :3],
            edge_potential=potential,
        )

    positions = np.empty((steps + 1, scenario.n_agents, 3))
    quaternions = np.empty((steps + 1, scenario.n_agents, 4))
    velocities = np.empty((steps + 1, scenario.n_agents, 3))
    controls = np.empty((steps, scenario.n_agents, 8))
    reference_positions = np.empty((steps + 1, 3))
    reference_velocities = np.empty((steps + 1, 3))
    velocity_commands = np.empty((steps + 1, 3))
    desired_edges = np.empty((steps + 1, len(scenario.graph.edges), 3))
    edge_errors = np.empty_like(desired_edges)
    relaxation_history = np.empty((steps + 1, scenario.n_agents, 4))
    relaxation_enlargement = np.empty_like(relaxation_history)
    relaxation_rates = np.zeros((steps, scenario.n_agents, 4))
    image_history = np.full((steps + 1, scenario.n_agents, 2), np.nan)
    slacks = np.full((steps, scenario.n_agents), np.nan)
    required_slacks = np.full((steps, scenario.n_agents), np.nan)
    actuation_margins = np.full((steps, scenario.n_agents), np.nan)
    controller_times = np.full((steps, scenario.n_agents), np.nan)
    phase_indices = np.empty(steps + 1, dtype=int)

    scalar_clf_fields = (
        "value",
        "decay",
        "drift",
        "configuration_local_rate",
        "parent_rate",
        "velocity_backstepping_rate",
        "dynamics_bias_rate",
        "dynamics_bias_linear_rate",
        "dynamics_bias_angular_rate",
        "command_acceleration_rate",
        "command_acceleration_linear_rate",
        "command_acceleration_angular_rate",
        "velocity_error_norm",
        "velocity_error_linear_norm",
        "velocity_error_angular_norm",
        "filtered_velocity_derivative_norm",
        "filtered_linear_acceleration_norm",
        "filtered_angular_acceleration_norm",
        "best_actuator_contribution",
        "minimum_modeled_derivative",
        "hard_clf_residual",
    )
    vector_clf_fields = (
        "generalized_velocity",
        "filtered_velocity",
        "desired_velocity",
        "unlimited_desired_velocity",
        "filtered_velocity_derivative",
        "dynamics_bias",
    )
    clf_histories = {
        name: np.full((steps, scenario.n_agents), np.nan)
        for name in scalar_clf_fields
    }
    clf_histories.update(
        {
            name: np.full((steps, scenario.n_agents, 6), np.nan)
            for name in vector_clf_fields
        }
    )

    def record(sample: int) -> None:
        phase_index = phase_index_at(times[sample], boundaries)
        phase = phases[phase_index]
        sample_scenario = scenario_with_formation(scenario, phase.formation)
        command = np.asarray(phase.velocity_command)
        reference = reference_model.evaluate(reference_state, command)

        phase_indices[sample] = phase_index
        velocity_commands[sample] = command
        reference_positions[sample] = reference.position
        reference_velocities[sample] = reference.velocity
        relaxation_history[sample] = relaxation_state
        for agent in range(scenario.n_agents):
            if agent in relaxation_policies:
                relaxation_enlargement[sample, agent] = relaxation_policies[
                    agent
                ].enlargement(relaxation_state[agent])
            else:
                relaxation_enlargement[sample, agent] = 0.0
            positions[sample, agent] = states[agent, :3]
            quaternions[sample, agent] = states[agent, 3:7]
            velocities[sample, agent] = inertial_linear_velocity(model, states[agent])
        for edge_index, edge in enumerate(scenario.graph.edges):
            desired = sample_scenario.desired_relative_position(edge.observer, edge.target)
            actual = states[edge.target, :3] - states[edge.observer, :3]
            desired_edges[sample, edge_index] = desired
            edge_errors[sample, edge_index] = actual - desired
            try:
                observation = camera.observe(
                    states[edge.observer, :3],
                    rotation_matrix_from_quaternion(states[edge.observer, 3:7]),
                    states[edge.target, :3],
                )
            except ValueError:
                continue
            image_history[sample, edge.observer] = (
                observation.image_point.alpha_h,
                observation.image_point.alpha_v,
            )

    plant_integrator = RK4Integrator()
    filter_integrator = RK4Integrator()
    reference_integrator = RK4Integrator()
    record(0)

    for step in range(steps):
        step_dt = times[step + 1] - times[step]
        phase = phases[phase_index_at(times[step], boundaries)]
        if phase.formation != active_formation:
            active_formation = phase.formation
            scenario = scenario_with_formation(scenario, active_formation)
            templates = build_templates()
            relaxation_policies = build_policies()

        command = np.asarray(phase.velocity_command, dtype=float)
        leader_reference = reference_model.evaluate(reference_state, command)
        next_states = states.copy()
        next_relaxation_state = relaxation_state.copy()

        start_time = perf_counter()
        leader_evaluation = leader_controller.evaluate(
            state=states[leader],
            reference=leader_reference,
            filter_state=leader_filter,
        )
        controller_times[step, leader] = perf_counter() - start_time
        leader_allocation = allocation.bounded_least_squares(
            leader_evaluation.wrench_body
        )
        controls[step, leader] = leader_allocation.forces
        slacks[step, leader] = leader_evaluation.slack
        if leader_evaluation.required_slack is not None:
            required_slacks[step, leader] = leader_evaluation.required_slack
        if leader_evaluation.actuation_margin is not None:
            actuation_margins[step, leader] = leader_evaluation.actuation_margin
        leader_filter = filter_integrator.step(
            leader_controller.dynamics_controller.command_filter,
            leader_filter,
            leader_evaluation.controller.filter_command,
            step_dt,
        )
        next_states[leader] = plant_integrator.step(
            model,
            states[leader],
            leader_allocation.achieved_wrench,
            step_dt,
        )

        # Every follower evaluates the same state snapshot.  Parent velocity is
        # intentionally absent, matching the distributed ROS controller.
        for edge in scenario.graph:
            observer = edge.observer
            target = edge.target
            start_time = perf_counter()
            kinematics = evaluate_sensing_constraint_kinematics(
                model,
                camera,
                distance_domain,
                fov_domain,
                states[observer],
                states[target],
            )
            relaxation_state[observer], _ = relaxation_policies[observer].project_to_current_domain(
                relaxation_state[observer],
                kinematics.values,
                enabled=enabled_relaxation,
            )
            potential = build_edge_potential(
                scenario,
                camera,
                distance_domain,
                fov_domain,
                observer,
                target,
                templates[observer],
                relaxation_policies[observer].enlargement(
                    relaxation_state[observer]
                ),
                adaptive=True,
                formation_gain=2.0,
            )
            evaluation = follower_controller.evaluate(
                follower_state=states[observer],
                parent_position=states[target, :3],
                parent_linear_velocity_inertial=None,
                edge_potential=potential,
                filter_state=follower_filters[observer],
            )
            required = (
                0.0
                if evaluation.required_slack is None
                else max(float(evaluation.required_slack), 0.0)
            )
            (
                next_relaxation_state[observer],
                relaxation_rates[step, observer],
            ) = relaxation_policies[observer].advance(
                relaxation_state[observer],
                conservative_values=kinematics.values,
                required_slack=required,
                enabled=enabled_relaxation,
                sample_time=step_dt,
            )
            controller_times[step, observer] = perf_counter() - start_time
            controls[step, observer] = (
                design.representative_thruster_forces(evaluation)
            )
            slacks[step, observer] = evaluation.slack
            if evaluation.required_slack is not None:
                required_slacks[step, observer] = evaluation.required_slack
            if evaluation.actuation_margin is not None:
                actuation_margins[step, observer] = evaluation.actuation_margin

            controller_clf = evaluation.controller.clf
            qp_result = evaluation.controller.qp
            generalized_velocity = model.split_state(states[observer])[2]
            zeta = evaluation.generalized_configuration_gradient
            configuration_local_rate = float(zeta @ generalized_velocity)
            # The distributed implementation deliberately does not use parent
            # velocity, so the modeled parent-rate contribution is zero.
            parent_rate = 0.0
            dynamics_bias = model.drift_wrench(states[observer])
            filtered_velocity = evaluation.controller.filter.output
            filtered_derivative = evaluation.controller.filter.output_derivative
            desired_velocity = evaluation.controller.desired_velocity
            unlimited_desired_velocity = evaluation.controller.unlimited_desired_velocity
            velocity_error = controller_clf.velocity_error
            command_acceleration_wrench = model.mass_matrix @ filtered_derivative

            dynamics_bias_linear_rate = float(-velocity_error[:3] @ dynamics_bias[:3])
            dynamics_bias_angular_rate = float(-velocity_error[3:] @ dynamics_bias[3:])
            dynamics_bias_rate = dynamics_bias_linear_rate + dynamics_bias_angular_rate
            command_acceleration_linear_rate = float(
                -velocity_error[:3] @ command_acceleration_wrench[:3]
            )
            command_acceleration_angular_rate = float(
                -velocity_error[3:] @ command_acceleration_wrench[3:]
            )
            command_acceleration_rate = (
                command_acceleration_linear_rate + command_acceleration_angular_rate
            )
            decay = float(
                follower_controller.dynamics_controller.qp.alpha(controller_clf.value)
            )
            feasibility = qp_result.actuation_feasibility
            if feasibility is None:
                minimum_modeled_derivative = np.nan
                best_actuator_contribution = np.nan
                hard_clf_residual = np.nan
            else:
                minimum_modeled_derivative = float(feasibility.minimum_modeled_derivative)
                best_actuator_contribution = float(
                    minimum_modeled_derivative - controller_clf.drift
                )
                hard_clf_residual = float(
                    controller_clf.drift + decay + best_actuator_contribution
                )

            scalar_values = {
                "value": controller_clf.value,
                "decay": decay,
                "drift": controller_clf.drift,
                "configuration_local_rate": configuration_local_rate,
                "parent_rate": parent_rate,
                "velocity_backstepping_rate": (
                    dynamics_bias_rate + command_acceleration_rate
                ),
                "dynamics_bias_rate": dynamics_bias_rate,
                "dynamics_bias_linear_rate": dynamics_bias_linear_rate,
                "dynamics_bias_angular_rate": dynamics_bias_angular_rate,
                "command_acceleration_rate": command_acceleration_rate,
                "command_acceleration_linear_rate": command_acceleration_linear_rate,
                "command_acceleration_angular_rate": command_acceleration_angular_rate,
                "velocity_error_norm": np.linalg.norm(velocity_error),
                "velocity_error_linear_norm": np.linalg.norm(velocity_error[:3]),
                "velocity_error_angular_norm": np.linalg.norm(velocity_error[3:]),
                "filtered_velocity_derivative_norm": np.linalg.norm(filtered_derivative),
                "filtered_linear_acceleration_norm": np.linalg.norm(filtered_derivative[:3]),
                "filtered_angular_acceleration_norm": np.linalg.norm(filtered_derivative[3:]),
                "best_actuator_contribution": best_actuator_contribution,
                "minimum_modeled_derivative": minimum_modeled_derivative,
                "hard_clf_residual": hard_clf_residual,
            }
            for name, value in scalar_values.items():
                clf_histories[name][step, observer] = value
            vector_values = {
                "generalized_velocity": generalized_velocity,
                "filtered_velocity": filtered_velocity,
                "desired_velocity": desired_velocity,
                "unlimited_desired_velocity": unlimited_desired_velocity,
                "filtered_velocity_derivative": filtered_derivative,
                "dynamics_bias": dynamics_bias,
            }
            for name, value in vector_values.items():
                clf_histories[name][step, observer] = value
            follower_filters[observer] = filter_integrator.step(
                follower_controller.dynamics_controller.command_filter,
                follower_filters[observer],
                evaluation.controller.filter_command,
                step_dt,
            )
            next_states[observer] = plant_integrator.step(
                model,
                states[observer],
                evaluation.wrench_body,
                step_dt,
            )

        reference_state = reference_integrator.step(
            reference_model,
            reference_state,
            command,
            step_dt,
        )
        states = next_states
        relaxation_state = next_relaxation_state
        record(step + 1)

    return FiveRobotMissionResult(
        trajectory=FormationTrajectory(
            times=times,
            positions=positions,
            velocities=velocities,
            controls=controls,
            quaternions=quaternions,
        ),
        scenario=scenario,
        leader_reference_position=reference_positions,
        leader_reference_velocity=reference_velocities,
        velocity_commands=velocity_commands,
        desired_edge_relative_positions=desired_edges,
        edge_errors=edge_errors,
        normalized_relaxation=relaxation_history,
        relaxation_enlargement=relaxation_enlargement,
        relaxation_rates=relaxation_rates,
        image_history=image_history,
        slacks=slacks,
        required_slacks=required_slacks,
        actuation_margins=actuation_margins,
        controller_times=controller_times,
        clf=CLFDiagnosticHistory(**clf_histories),
        camera=camera,
        distance_domain=distance_domain,
        fov_domain=fov_domain,
        phase_indices=phase_indices,
        phases=phases,
        allocation=allocation,
    )


def print_summary(result: FiveRobotMissionResult) -> None:
    leader = result.scenario.graph.root
    leader_position_error = np.linalg.norm(
        result.trajectory.positions[:, leader] - result.leader_reference_position,
        axis=1,
    )
    edge_error_norm = np.linalg.norm(result.edge_errors, axis=2)
    utilization = np.empty(result.trajectory.controls.shape[:2])
    for step in range(utilization.shape[0]):
        for agent in range(utilization.shape[1]):
            utilization[step, agent] = np.max(
                result.allocation.utilization(result.trajectory.controls[step, agent])
            )

    print("Pure-Python five-robot tree mission: full profile")
    print(
        "Gazebo per-thruster limits: "
        f"-{result.allocation.configuration.force_limits.reverse:.3f} N, "
        f"+{result.allocation.configuration.force_limits.forward:.3f} N"
    )
    print(f"Duration: {result.trajectory.times[-1]:.1f} s")
    print(
        "Leader reference tracking: "
        f"RMS {np.sqrt(np.mean(leader_position_error**2)):.4f} m, "
        f"peak {np.max(leader_position_error):.4f} m"
    )
    for edge_index, label in enumerate(EDGE_LABELS):
        print(
            f"Edge {label}: RMS {np.sqrt(np.mean(edge_error_norm[:, edge_index] ** 2)):.4f} m, "
            f"peak {np.max(edge_error_norm[:, edge_index]):.4f} m"
        )
    print(f"Maximum normalized domain enlargement: {np.max(result.normalized_relaxation):.3f}")
    print(f"Maximum thruster utilization: {np.max(utilization):.3f}")


def _draw_action_boundaries(axes: plt.Axes, result: FiveRobotMissionResult) -> None:
    boundaries = phase_boundaries(result.phases)
    for phase_index, boundary in enumerate(boundaries[:-1]):
        next_phase = result.phases[phase_index + 1]
        if next_phase.label.startswith(("command", "leader")):
            axes.axvline(boundary, color="0.7", linewidth=0.7, linestyle=":")


def _load_common_diagnostic_plotters() -> ModuleType:
    """Load the plotters also used by ``plot_formation_experiment.py``."""

    path = Path(__file__).with_name("04_bluerov2_fov_clf_qp.py")
    module_name = "_five_robot_common_bluerov2_plotters"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load diagnostic plotters from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def plot_results(
    result: FiveRobotMissionResult,
    *,
    paper_quality: bool = False,
) -> dict[str, plt.Figure | None]:
    """Create every applicable paper and debug plot from the ROS plotter.

    Workspace and workspace-relaxation plots are the only omissions: this
    pure-Python example intentionally has no water-tank/workspace model.
    """

    times = result.trajectory.times
    leader = result.scenario.graph.root
    legacy = _load_common_diagnostic_plotters()
    # This diagnostic set intentionally exceeds Matplotlib's default warning
    # threshold; figures are closed in ``main`` after saving when not shown.
    plt.rcParams["figure.max_open_warning"] = 0
    figures: dict[str, plt.Figure | None] = {}

    edge_quality = legacy.connection_quality_history(
        result.scenario,
        result.trajectory,
        result.distance_domain,
        result.fov_domain,
        result.image_history,
        distance_constraints=True,
        fov_constraints=True,
    )
    desired_positions = (
        result.leader_reference_position[-1]
        + formation_offsets()[result.phases[-1].formation]
    )
    trajectory_plot = plot_formation_3d(
        result.trajectory,
        result.scenario.graph,
        desired_positions=desired_positions,
        camera=result.camera,
        camera_agents=tuple(range(1, result.scenario.n_agents)),
        camera_depth=0.75,
        vehicle_geometry=BlueROV2HeavyVisualGeometry(
            thruster_configuration=result.allocation.configuration
        ).wireframe(),
        edge_quality=edge_quality,
        paper_quality=paper_quality,
        show_body_forward=False,
        title="Five-BlueROV full mission (no tank constraints)",
    )
    figures["trajectory_3d"] = (
        trajectory_plot[0] if isinstance(trajectory_plot, tuple) else trajectory_plot
    )

    leader_tracking, (position_error_axes, velocity_error_axes) = plt.subplots(
        2, 1, sharex=True
    )
    position_error_axes.plot(
        times,
        np.linalg.norm(
            result.trajectory.positions[:, leader] - result.leader_reference_position,
            axis=1,
        ),
    )
    position_error_axes.set_ylabel(r"$\|p_1-p_r\|$ [m]")
    position_error_axes.set_title("Leader tracking error")
    position_error_axes.grid(True, alpha=0.3)
    velocity_error_axes.plot(
        times,
        np.linalg.norm(
            result.trajectory.velocities[:, leader] - result.leader_reference_velocity,
            axis=1,
        ),
    )
    velocity_error_axes.set(xlabel="time [s]", ylabel=r"$\|v_1-v_r\|$ [m/s]")
    velocity_error_axes.grid(True, alpha=0.3)
    leader_tracking.tight_layout()
    figures["leader_tracking"] = leader_tracking

    leader_position, position_axes = plt.subplots(3, 1, sharex=True)
    component_labels = ("x", "y", "z")
    for component, axes in enumerate(position_axes):
        axes.plot(times, result.trajectory.positions[:, leader, component], label="actual")
        axes.plot(
            times,
            result.leader_reference_position[:, component],
            "--",
            label="reference",
        )
        axes.set_ylabel(f"{component_labels[component]} [m]")
        axes.grid(True, alpha=0.3)
    position_axes[0].set_title("Leader position tracking")
    position_axes[0].legend()
    position_axes[-1].set_xlabel("time [s]")
    leader_position.tight_layout()
    figures["leader_position"] = leader_position

    tracking, tracking_axes = plt.subplots(3, 1, sharex=True)
    actual_velocity = result.trajectory.velocities[:, leader]
    for component, axes in enumerate(tracking_axes):
        axes.plot(times, actual_velocity[:, component], label="actual")
        axes.plot(times, result.leader_reference_velocity[:, component], "--", label="reference")
        axes.step(
            times,
            result.velocity_commands[:, component],
            where="post",
            linestyle=":",
            label="command",
        )
        axes.set_ylabel(f"v{component_labels[component]} [m/s]")
        axes.grid(True, alpha=0.3)
        _draw_action_boundaries(axes, result)
    tracking_axes[0].legend(ncol=3)
    tracking_axes[-1].set_xlabel("time [s]")
    tracking.suptitle("Leader velocity command and tracking")
    tracking.tight_layout()
    figures["leader_velocity_command"] = tracking

    errors, error_axes = plt.subplots()
    error_norms = np.linalg.norm(result.edge_errors, axis=2)
    for edge_index, label in enumerate(EDGE_LABELS):
        error_axes.plot(times, error_norms[:, edge_index], label=label)
    error_axes.set(xlabel="time [s]", ylabel="relative-position error [m]")
    error_axes.set_title("Formation tracking error")
    error_axes.grid(True, alpha=0.3)
    error_axes.legend(title="follower→parent")
    _draw_action_boundaries(error_axes, result)
    errors.tight_layout()
    figures["formation_error"] = errors

    for edge_index, edge in enumerate(result.scenario.graph.edges):
        edge_figure, edge_axes = plt.subplots(3, 1, sharex=True)
        actual = (
            result.trajectory.positions[:, edge.target]
            - result.trajectory.positions[:, edge.observer]
        )
        desired = result.desired_edge_relative_positions[:, edge_index]
        for component, axes in enumerate(edge_axes):
            axes.plot(times, actual[:, component], label="actual")
            axes.plot(times, desired[:, component], "--", label="desired")
            axes.set_ylabel(f"d{component_labels[component]} [m]")
            axes.grid(True, alpha=0.3)
            _draw_action_boundaries(axes, result)
        edge_axes[0].set_title(
            f"Formation tracking: {ROBOT_NAMES[edge.observer]}→{ROBOT_NAMES[edge.target]}"
        )
        edge_axes[0].legend()
        edge_axes[-1].set_xlabel("time [s]")
        edge_figure.tight_layout()
        figures[
            f"formation_tracking_{ROBOT_NAMES[edge.observer]}_to_{ROBOT_NAMES[edge.target]}"
        ] = edge_figure

    figures["distance"] = legacy.plot_distance_diagnostics(
        result.scenario,
        result.trajectory,
        result.distance_domain,
        result.relaxation_enlargement,
        adaptive=True,
    )
    horizontal_fov, vertical_fov = legacy.plot_fov_diagnostics(
        result.scenario,
        result.trajectory,
        result.fov_domain,
        result.relaxation_enlargement,
        result.image_history,
        adaptive=True,
    )
    figures["fov_horizontal"] = horizontal_fov
    figures["fov_vertical"] = vertical_fov
    adaptive_fov, _ = plot_vertical_fov_relaxation_from_histories(
        times,
        result.image_history,
        result.normalized_relaxation,
        observer=result.scenario.graph.edges[0].observer,
        alpha_conservative=result.fov_domain.alpha_v_conservative,
        alpha_physical=1.0,
        domain_margin_ratio=0.1,
        paper_quality=paper_quality,
    )
    figures["adaptive_fov_vertical"] = adaptive_fov

    figures["slack"] = legacy.plot_slack(
        result.scenario,
        result.trajectory,
        result.slacks,
        result.required_slacks,
    )
    figures["actuation_margin"] = legacy.plot_actuation_margin(
        result.scenario,
        result.trajectory,
        result.actuation_margins,
    )
    figures["domain_enlargement"] = legacy.plot_domain_enlargement(
        result.scenario,
        result.trajectory,
        result.distance_domain,
        result.fov_domain,
        result.relaxation_enlargement,
    )
    figures["relaxation_rates"] = legacy.plot_relaxation_rates(
        result.scenario,
        result.trajectory,
        result.relaxation_rates,
    )

    control_times = times[:-1]
    for agent, robot_name in enumerate(ROBOT_NAMES):
        thruster_figure, thruster_axes = plt.subplots()
        for thruster, thruster_name in enumerate(result.allocation.configuration.names):
            thruster_axes.plot(
                control_times,
                result.trajectory.controls[:, agent, thruster],
                label=thruster_name,
            )
        thruster_axes.axhline(
            result.allocation.configuration.force_limits.forward,
            linestyle="--",
            label="forward limit",
        )
        thruster_axes.axhline(
            -result.allocation.configuration.force_limits.reverse,
            linestyle="--",
            label="reverse limit",
        )
        thruster_axes.set(
            xlabel="time [s]",
            ylabel="thruster force [N]",
            title=f"Thruster forces: {robot_name}",
        )
        thruster_axes.grid(True, alpha=0.3)
        thruster_axes.legend(ncol=2)
        thruster_figure.tight_layout()
        figures[f"thruster_forces_{robot_name}"] = thruster_figure

    figures["controller_time"] = legacy.plot_controller_times(
        result.scenario,
        result.trajectory,
        result.controller_times,
        control_space="thruster",
    )
    figures["clf_value_decay"] = legacy.plot_clf_value_and_decay(
        result.scenario, result.trajectory, result.clf
    )
    figures["clf_feasibility_balance"] = legacy.plot_clf_feasibility_balance(
        result.scenario, result.trajectory, result.clf
    )
    figures["clf_drift_components"] = legacy.plot_clf_drift_components(
        result.scenario, result.trajectory, result.clf
    )
    figures["velocity_backstepping_split"] = legacy.plot_velocity_backstepping_split(
        result.scenario, result.trajectory, result.clf
    )
    figures["velocity_backstepping_peak"] = legacy.plot_velocity_backstepping_peak_detail(
        result.trajectory, result.required_slacks, result.clf
    )
    figures["command_velocity_peak"] = legacy.plot_command_velocity_peak_detail(
        result.trajectory, result.required_slacks, result.clf
    )
    figures["command_acceleration_peak"] = legacy.plot_command_acceleration_peak_detail(
        result.trajectory, result.required_slacks, result.clf
    )
    figures["clf_peak_detail"] = legacy.plot_clf_peak_detail(
        result.scenario,
        result.trajectory,
        result.required_slacks,
        result.clf,
    )
    return figures


def save_histories(result: FiveRobotMissionResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "five_robot_tree_full_mission.npz",
        times=result.trajectory.times,
        positions=result.trajectory.positions,
        velocities=result.trajectory.velocities,
        quaternions=result.trajectory.quaternions,
        controls=result.trajectory.controls,
        leader_reference_position=result.leader_reference_position,
        leader_reference_velocity=result.leader_reference_velocity,
        velocity_commands=result.velocity_commands,
        desired_edge_relative_positions=result.desired_edge_relative_positions,
        edge_errors=result.edge_errors,
        normalized_relaxation=result.normalized_relaxation,
        relaxation_enlargement=result.relaxation_enlargement,
        relaxation_rates=result.relaxation_rates,
        image_history=result.image_history,
        slacks=result.slacks,
        required_slacks=result.required_slacks,
        actuation_margins=result.actuation_margins,
        controller_times=result.controller_times,
        **{
            f"clf_{name}": value
            for name, value in vars(result.clf).items()
        },
        phase_indices=result.phase_indices,
        phase_labels=np.asarray([phase.label for phase in result.phases]),
        phase_durations=np.asarray([phase.duration for phase in result.phases]),
        robot_names=np.asarray(ROBOT_NAMES),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument(
        "--paper",
        action="store_true",
        help="compatibility flag; all paper and debug plots are generated",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="compatibility flag; all applicable plots are already generated",
    )
    parser.add_argument("--paper-quality", action="store_true")
    parser.add_argument("--animate", action="store_true")
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--save-animation", action="store_true")
    parser.add_argument(
        "--animation-format",
        choices=("gif", "mp4"),
        default="mp4",
    )
    parser.add_argument(
        "--figure-format",
        "--format",
        dest="figure_format",
        choices=("png", "pdf", "svg"),
        default="png",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/bluerov2_five_robot_tree_full_mission"),
    )
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.dt <= 0.0:
        parser.error("--dt must be positive")
    if args.frame_stride <= 0:
        parser.error("--frame-stride must be positive")

    apply_visualization_style(paper_quality=args.paper_quality)
    result = simulate_full_mission(dt=args.dt)
    print_summary(result)
    figures = plot_results(result, paper_quality=args.paper_quality)

    animation = None
    if args.animate or args.save_animation:
        followers = tuple(range(1, result.scenario.n_agents))
        vehicle_geometry = BlueROV2HeavyVisualGeometry(
            thruster_configuration=result.allocation.configuration
        ).wireframe()
        animation = animate_formation_3d(
            result.trajectory,
            result.scenario.graph,
            camera=build_camera(),
            camera_agents=followers,
            camera_depth=0.75,
            vehicle_geometry=vehicle_geometry,
            edge_quality=_load_common_diagnostic_plotters().connection_quality_history(
                result.scenario,
                result.trajectory,
                result.distance_domain,
                result.fov_domain,
                result.image_history,
                distance_constraints=True,
                fov_constraints=True,
            ),
            paper_quality=args.paper_quality,
            show_body_forward=False,
            trail_length=300,
            frame_stride=args.frame_stride,
            title="Five-BlueROV full mission — pure Python",
        )

    if args.save:
        save_histories(result, args.output_dir)
        saved_figures = 0
        for name, figure in figures.items():
            if figure is None:
                continue
            save_figure(
                figure,
                args.output_dir / f"{name}.{args.figure_format}",
                paper_quality=args.paper_quality,
            )
            saved_figures += 1
        print(
            f"Saved histories and {saved_figures} figures to {args.output_dir}"
        )

    if args.save_animation:
        assert animation is not None
        args.output_dir.mkdir(parents=True, exist_ok=True)
        animation_path = args.output_dir / f"mission_animation.{args.animation_format}"
        save_animation(
            animation.animation,
            animation_path,
            paper_quality=args.paper_quality,
        )
        print(f"Saved animation to {animation_path}")

    if not args.no_show:
        plt.show()
    else:
        for figure in figures.values():
            if figure is not None:
                plt.close(figure)


if __name__ == "__main__":
    main()
