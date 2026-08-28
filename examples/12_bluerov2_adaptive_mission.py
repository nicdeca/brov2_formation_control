#!/usr/bin/env python3
"""Two-BlueROV adaptive-domain mission with moving leader and formation switches.

This paper-oriented pure-Python example is intentionally more mission-like than
the short stress test in example 04.  A leader executes a moderate translational
mission while one follower is commanded through several large relative-position
reconfigurations.  The desired formations remain inside the deliberately tight
conservative sensing domain; relaxation is therefore driven by transient
second-order dynamics and bounded actuation, not by an inadmissible steady
reference.

The default conservative sensing limits are tightened to

    0.8 <= d <= 2.4 m,
    |alpha_h| <= 0.45,
    |alpha_v| <= 0.45,

while the physical limits remain

    0.5 <= d <= 3.6 m,
    |alpha_h| < 1,
    |alpha_v| < 1.

The mission is designed to strongly excite range and horizontal-FoV relaxation
while retaining physical sensing.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from formation_control.actuation import (
    BlueROV2HeavyThrusterAllocation,
)
from formation_control.constraints import (
    DistanceDomain,
    FieldOfViewDomain,
    HorizontalFieldOfViewConstraint,
    MaximumDistanceConstraint,
    MinimumDistanceConstraint,
    NormalizedImagePoint,
    VerticalFieldOfViewConstraint,
    evaluate_sensing_constraint_kinematics,
)
from formation_control.control import (
    BlueROV2ControlSpace,
    FunnelRelaxationInfeasibleError,
    FunnelRelaxationPolicy,
    build_bluerov2_controller_design,
)
from formation_control.geometry import (
    PinholeCamera,
    quaternion_from_roll_pitch_yaw,
    rotation_matrix_from_quaternion,
)
from formation_control.graphs import DirectedSensingGraph
from formation_control.models import BlueROV2Model
from formation_control.potentials import (
    AdaptiveConstraintBarrierPotential,
    EdgePotential,
    ImageCenteringPotential,
    RelativePositionPotential,
)
from formation_control.simulation import (
    FormationReference,
    FormationScenario,
    FormationTrajectory,
    RK4Integrator,
)
from formation_control.visualization import (
    BlueROV2HeavyVisualGeometry,
    apply_visualization_style,
    save_animation,
    save_figure,
)
from formation_control.visualization.formation_3d import (
    DesiredVehicleStyle3D,
    animate_formation_3d,
)


@dataclass(frozen=True)
class MissionFormation:
    name: str
    start: float
    desired_relative: np.ndarray


@dataclass(frozen=True)
class MissionResult:
    scenario: FormationScenario
    trajectory: FormationTrajectory
    desired_positions: np.ndarray
    desired_relative: np.ndarray
    distance_domain: DistanceDomain
    fov_domain: FieldOfViewDomain
    camera: PinholeCamera
    rho_history: np.ndarray
    image_history: np.ndarray
    slacks: np.ndarray
    required_slacks: np.ndarray
    allocation: BlueROV2HeavyThrusterAllocation


def look_at_quaternion(relative_position_inertial: np.ndarray) -> np.ndarray:
    relative = np.asarray(relative_position_inertial, dtype=float)
    horizontal = float(np.hypot(relative[0], relative[1]))
    yaw = float(np.arctan2(relative[1], relative[0]))
    pitch = float(-np.arctan2(relative[2], horizontal))
    return quaternion_from_roll_pitch_yaw(0.0, pitch, yaw)


def build_camera() -> PinholeCamera:
    return PinholeCamera.from_degrees(
        horizontal_half_angle=45.0,
        vertical_half_angle=30.0,
    )


def build_domains(
    *,
    d_max_conservative: float,
    alpha_h_conservative: float,
    alpha_v_conservative: float,
) -> tuple[DistanceDomain, FieldOfViewDomain]:
    return (
        DistanceDomain(
            d_min=0.5,
            d_max=3.6,
            d_min_conservative=0.8,
            d_max_conservative=d_max_conservative,
        ),
        FieldOfViewDomain(
            alpha_h_conservative=alpha_h_conservative,
            alpha_v_conservative=alpha_v_conservative,
        ),
    )


def mission_formations() -> tuple[MissionFormation, ...]:
    # Parent-minus-follower references.  Every endpoint is strictly inside
    # d_max^c = 2.4 m.  B -> C is the main lateral reconfiguration; C -> D
    # combines a large lateral return with a near-range-boundary formation.
    return (
        MissionFormation(
            "A_nominal",
            0.0,
            np.array([0.70, -1.80, 0.00], dtype=float),
        ),
        MissionFormation(
            "B_starboard_high",
            18.0,
            np.array([1.45, -1.65, -0.40], dtype=float),
        ),
        MissionFormation(
            "C_port_low",
            38.0,
            np.array([-1.45, -1.65, 0.40], dtype=float),
        ),
        MissionFormation(
            "D_long",
            60.0,
            np.array([0.55, -2.30, -0.30], dtype=float),
        ),
        MissionFormation(
            "A_recover",
            80.0,
            np.array([0.70, -1.80, 0.00], dtype=float),
        ),
    )


def desired_relative_at(time: float) -> np.ndarray:
    current = mission_formations()[0].desired_relative
    for formation in mission_formations():
        if time + 1e-12 >= formation.start:
            current = formation.desired_relative
        else:
            break
    return current.copy()


def leader_velocity_reference(time: float) -> np.ndarray:
    # Moderate mission motion.  The aggressive event is the formation switch,
    # not an artificial high-speed leader pulse.
    if 12.0 <= time < 32.0:
        return np.array([0.10, 0.08, 0.00])
    if 32.0 <= time < 52.0:
        return np.array([0.08, -0.10, 0.03])
    if 52.0 <= time < 72.0:
        return np.array([-0.08, 0.10, -0.03])
    if 72.0 <= time < 88.0:
        return np.array([-0.06, -0.08, 0.00])
    return np.zeros(3)


def integrate_leader_reference(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    velocity = np.stack([leader_velocity_reference(float(t)) for t in times])
    position = np.zeros((times.size, 3), dtype=float)
    position[0] = np.array([0.0, 0.0, -1.0], dtype=float)
    for k in range(times.size - 1):
        dt = float(times[k + 1] - times[k])
        position[k + 1] = position[k] + dt * velocity[k]
    return position, velocity


def build_edge_potential(
    *,
    desired_relative: np.ndarray,
    camera: PinholeCamera,
    distance_domain: DistanceDomain,
    fov_domain: FieldOfViewDomain,
    rho: np.ndarray,
    adaptive: bool,
) -> EdgePotential:
    desired_image = NormalizedImagePoint(0.0, 0.0)

    if adaptive:
        collision = AdaptiveConstraintBarrierPotential(
            constraint=MinimumDistanceConstraint(distance_domain.d_min_conservative),
            reference_state=desired_relative,
            weight=0.18,
        ).bind(float(rho[0]))
        sensing_range = AdaptiveConstraintBarrierPotential(
            constraint=MaximumDistanceConstraint(distance_domain.d_max_conservative),
            reference_state=desired_relative,
            weight=0.18,
        ).bind(float(rho[1]))
        horizontal = AdaptiveConstraintBarrierPotential(
            constraint=HorizontalFieldOfViewConstraint(fov_domain.alpha_h_conservative),
            reference_state=desired_image,
            weight=0.25,
        ).bind(float(rho[2]))
        vertical = AdaptiveConstraintBarrierPotential(
            constraint=VerticalFieldOfViewConstraint(fov_domain.alpha_v_conservative),
            reference_state=desired_image,
            weight=0.25,
        ).bind(float(rho[3]))
    else:
        # The paper mission is intended to run adaptively.  Keeping this branch
        # makes the baseline comparison available without duplicating the file.
        from formation_control.potentials import ConstraintBarrierPotential

        collision = ConstraintBarrierPotential.from_reference(
            MinimumDistanceConstraint(distance_domain.d_min_conservative),
            desired_relative,
            weight=0.18,
        )
        sensing_range = ConstraintBarrierPotential.from_reference(
            MaximumDistanceConstraint(distance_domain.d_max_conservative),
            desired_relative,
            weight=0.18,
        )
        horizontal = ConstraintBarrierPotential.from_reference(
            HorizontalFieldOfViewConstraint(fov_domain.alpha_h_conservative),
            desired_image,
            weight=0.25,
        )
        vertical = ConstraintBarrierPotential.from_reference(
            VerticalFieldOfViewConstraint(fov_domain.alpha_v_conservative),
            desired_image,
            weight=0.25,
        )

    return EdgePotential(
        formation=RelativePositionPotential.isotropic(
            desired_relative,
            gain=1.4,
        ),
        image_centering=ImageCenteringPotential(
            horizontal_gain=0.8,
            vertical_gain=0.8,
        ),
        collision_barrier=collision,
        range_barrier=sensing_range,
        horizontal_fov_barrier=horizontal,
        vertical_fov_barrier=vertical,
        camera=camera,
    )


def inertial_velocity(model: BlueROV2Model, state: np.ndarray) -> np.ndarray:
    _, quaternion, velocity = model.split_state(state)
    return rotation_matrix_from_quaternion(quaternion) @ velocity[:3]


def leader_tracking_wrench(
    model: BlueROV2Model,
    state: np.ndarray,
    position_reference: np.ndarray,
    velocity_reference: np.ndarray,
) -> np.ndarray:
    position, quaternion, velocity = model.split_state(state)
    rotation = rotation_matrix_from_quaternion(quaternion)
    velocity_inertial = rotation @ velocity[:3]

    # The leader reference is deliberately modest.  This PD+model-drift law is
    # used only to make the pure-Python tuning mission move; the Gazebo port
    # will use the normal leader CLF-QP.
    force_inertial = 9.0 * (position_reference - position) + 12.0 * (
        velocity_reference - velocity_inertial
    )
    force_body = rotation.T @ force_inertial
    torque_body = -2.0 * velocity[3:]
    return model.drift_wrench(state) + np.concatenate((force_body, torque_body))


def simulate(
    *,
    duration: float,
    dt: float,
    adaptive: bool,
    control_space: BlueROV2ControlSpace,
    d_max_conservative: float,
    alpha_h_conservative: float,
    alpha_v_conservative: float,
    thrust_derating: float,
    relaxation_recovery_gain: float,
    relaxation_barrier_gain: float,
    relaxation_domain_margin_ratio: float,
    relaxation_activation_on_ratio: float,
    relaxation_activation_off_ratio: float,
) -> MissionResult:
    graph = DirectedSensingGraph.rooted_star(2)
    initial_relative = desired_relative_at(0.0)
    initial_leader = np.array([0.0, 0.0, -1.0])
    initial_follower = initial_leader - initial_relative

    scenario = FormationScenario(
        graph=graph,
        reference=FormationReference(
            offsets=np.array(
                [
                    [0.0, 0.0, 0.0],
                    -initial_relative,
                ]
            )
        ),
        initial_positions=np.stack((initial_leader, initial_follower)),
    )

    model = BlueROV2Model()
    camera = build_camera()
    distance_domain, fov_domain = build_domains(
        d_max_conservative=d_max_conservative,
        alpha_h_conservative=alpha_h_conservative,
        alpha_v_conservative=alpha_v_conservative,
    )
    allocation = BlueROV2HeavyThrusterAllocation.default_45deg(
        voltage=16,
        derating=thrust_derating,
    )
    design = build_bluerov2_controller_design(
        model,
        allocation,
        control_space=control_space,
        slack_linear_penalty=100.0,
        slack_penalty=5e3,
        virtual_velocity_norm_limits=np.array([1.5, 2.0]),
    )
    controller = design.agent_controller

    maximum = np.array(
        [
            distance_domain.collision_enlargement_max,
            distance_domain.range_enlargement_max,
            fov_domain.horizontal_enlargement_max,
            fov_domain.vertical_enlargement_max,
        ]
    )
    relaxation = FunnelRelaxationPolicy(
        maximum_enlargement=maximum,
        recovery_gain=relaxation_recovery_gain,
        barrier_gain=relaxation_barrier_gain,
        domain_margin_ratio=relaxation_domain_margin_ratio,
        activation_on_ratio=relaxation_activation_on_ratio,
        activation_off_ratio=relaxation_activation_off_ratio,
        minimum_constraint_margin=1e-5,
    )
    enabled = np.ones(4, dtype=bool)

    states = np.zeros((2, 13), dtype=float)
    states[0, :3] = initial_leader
    states[0, 3:7] = quaternion_from_roll_pitch_yaw(0.0, 0.0, 0.0)
    states[1, :3] = initial_follower
    states[1, 3:7] = look_at_quaternion(initial_relative)

    relaxation_state = np.zeros((2, 4), dtype=float)
    edge = tuple(graph.edges)[0]

    kinematics = evaluate_sensing_constraint_kinematics(
        model,
        camera,
        distance_domain,
        fov_domain,
        states[1],
        states[0],
    )
    if adaptive:
        projected, _ = relaxation.project_to_current_domain(
            relaxation_state[1],
            kinematics.values,
            enabled=enabled,
        )
        relaxation_state[1] = projected

    potential = build_edge_potential(
        desired_relative=initial_relative,
        camera=camera,
        distance_domain=distance_domain,
        fov_domain=fov_domain,
        rho=relaxation.enlargement(relaxation_state[1]),
        adaptive=adaptive,
    )
    filter_state = controller.initialize_filter(
        follower_state=states[1],
        parent_position=states[0, :3],
        edge_potential=potential,
    )

    steps = int(np.ceil(duration / dt))
    times = np.arange(steps + 1, dtype=float) * dt
    leader_ref_position, leader_ref_velocity = integrate_leader_reference(times)

    positions = np.empty((steps + 1, 2, 3))
    quaternions = np.empty((steps + 1, 2, 4))
    velocities = np.empty((steps + 1, 2, 3))
    controls = np.zeros((steps, 2, 8))
    rho_history = np.zeros((steps + 1, 2, 4))
    image_history = np.full((steps + 1, 2, 2), np.nan)
    desired_relative_history = np.empty((steps + 1, 3))
    desired_positions = np.empty((steps + 1, 2, 3))
    slacks = np.zeros((steps, 2))
    required_slacks = np.full((steps, 2), np.nan)

    plant_integrator = RK4Integrator()
    filter_integrator = RK4Integrator()

    def record(sample: int) -> None:
        positions[sample] = states[:, :3]
        quaternions[sample] = states[:, 3:7]
        velocities[sample, 0] = inertial_velocity(model, states[0])
        velocities[sample, 1] = inertial_velocity(model, states[1])
        rho_history[sample, 1] = relaxation.enlargement(relaxation_state[1])
        desired_relative_history[sample] = desired_relative_at(times[sample])
        desired_positions[sample, 0] = leader_ref_position[sample]
        desired_positions[sample, 1] = (
            leader_ref_position[sample] - desired_relative_history[sample]
        )
        try:
            observation = camera.observe(
                states[1, :3],
                rotation_matrix_from_quaternion(states[1, 3:7]),
                states[0, :3],
            )
            image_history[sample, 1] = np.array(
                [
                    observation.image_point.alpha_h,
                    observation.image_point.alpha_v,
                ]
            )
        except ValueError:
            pass

    record(0)

    for step in range(steps):
        time = float(times[step])
        next_states = states.copy()

        # Moving leader.
        requested = leader_tracking_wrench(
            model,
            states[0],
            leader_ref_position[step],
            leader_ref_velocity[step],
        )
        leader_allocation = allocation.bounded_least_squares(requested)
        controls[step, 0] = leader_allocation.forces
        next_states[0] = plant_integrator.step(
            model,
            states[0],
            leader_allocation.achieved_wrench,
            dt,
        )

        # Follower.
        desired_relative = desired_relative_at(time)
        kinematics = evaluate_sensing_constraint_kinematics(
            model,
            camera,
            distance_domain,
            fov_domain,
            states[1],
            states[0],
        )

        if adaptive:
            projected, _ = relaxation.project_to_current_domain(
                relaxation_state[1],
                kinematics.values,
                enabled=enabled,
            )
            relaxation_state[1] = projected

        rho = relaxation.enlargement(relaxation_state[1])
        potential = build_edge_potential(
            desired_relative=desired_relative,
            camera=camera,
            distance_domain=distance_domain,
            fov_domain=fov_domain,
            rho=rho,
            adaptive=adaptive,
        )

        evaluation = controller.evaluate(
            follower_state=states[1],
            parent_position=states[0, :3],
            parent_linear_velocity_inertial=inertial_velocity(model, states[0]),
            edge_potential=potential,
            filter_state=filter_state,
        )

        controls[step, 1] = design.representative_thruster_forces(evaluation)
        slacks[step, 1] = evaluation.slack
        if evaluation.required_slack is not None:
            required_slacks[step, 1] = evaluation.required_slack

        if adaptive:
            relaxation_eval = relaxation.evaluate(
                relaxation_state[1],
                conservative_values=kinematics.values,
                enabled=enabled,
            )
            relaxation_state[1] = np.maximum(
                relaxation_state[1] + dt * relaxation_eval.selected_rate,
                0.0,
            )

        filter_state = filter_integrator.step(
            controller.dynamics_controller.command_filter,
            filter_state,
            evaluation.controller.desired_velocity,
            dt,
        )
        next_states[1] = plant_integrator.step(
            model,
            states[1],
            evaluation.wrench_body,
            dt,
        )

        states = next_states
        record(step + 1)

    trajectory = FormationTrajectory(
        times=times,
        positions=positions,
        velocities=velocities,
        controls=controls,
        quaternions=quaternions,
    )

    return MissionResult(
        scenario=scenario,
        trajectory=trajectory,
        desired_positions=desired_positions,
        desired_relative=desired_relative_history,
        distance_domain=distance_domain,
        fov_domain=fov_domain,
        camera=camera,
        rho_history=rho_history,
        image_history=image_history,
        slacks=slacks,
        required_slacks=required_slacks,
        allocation=allocation,
    )


def plot_formation_tracking(result: MissionResult) -> plt.Figure:
    actual = result.trajectory.positions[:, 0] - result.trajectory.positions[:, 1]
    fig, axes = plt.subplots(3, 1, sharex=True)
    labels = ("x", "y", "z")
    for k, axis in enumerate(axes):
        axis.plot(
            result.trajectory.times,
            actual[:, k],
            label=rf"$p_{{21,{labels[k]}}}$",
        )
        axis.plot(
            result.trajectory.times,
            result.desired_relative[:, k],
            "--",
            label=rf"$p_{{21,{labels[k]}}}^d$",
        )
        axis.grid(True, alpha=0.3)
        axis.legend()
    axes[-1].set_xlabel(r"$t$ [s]")
    axes[1].set_ylabel("relative position [m]")
    fig.tight_layout()
    return fig


def plot_distance(result: MissionResult) -> plt.Figure:
    times = result.trajectory.times
    relative = result.trajectory.positions[:, 0] - result.trajectory.positions[:, 1]
    distance = np.linalg.norm(relative, axis=1)
    adaptive_min = np.array(
        [
            result.distance_domain.effective_minimum_distance(rho)
            for rho in result.rho_history[:, 1, 0]
        ]
    )
    adaptive_max = np.array(
        [
            result.distance_domain.effective_maximum_distance(rho)
            for rho in result.rho_history[:, 1, 1]
        ]
    )

    fig, ax = plt.subplots()
    ax.plot(times, distance, label=r"$d_{21}$")
    ax.plot(times, adaptive_min, "--", label=r"$d_{\min,21}^a$")
    ax.plot(times, adaptive_max, "--", label=r"$d_{\max,21}^a$")
    ax.axhline(result.distance_domain.d_min, linestyle=":", label=r"$d_{\min}$")
    ax.axhline(result.distance_domain.d_max, linestyle=":", label=r"$d_{\max}$")
    ax.axhline(
        result.distance_domain.d_min_conservative,
        linestyle="-.",
        label=r"$d_{\min}^c$",
    )
    ax.axhline(
        result.distance_domain.d_max_conservative,
        linestyle="-.",
        label=r"$d_{\max}^c$",
    )
    ax.set_xlabel(r"$t$ [s]")
    ax.set_ylabel(r"$d_{21}$ [m]")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    return fig


def plot_fov(result: MissionResult, component: int) -> plt.Figure:
    times = result.trajectory.times
    symbol = "h" if component == 0 else "v"
    channel = 2 if component == 0 else 3
    conservative = (
        result.fov_domain.alpha_h_conservative
        if component == 0
        else result.fov_domain.alpha_v_conservative
    )
    if component == 0:
        adaptive = np.array(
            [
                result.fov_domain.effective_horizontal_limit(rho)
                for rho in result.rho_history[:, 1, channel]
            ]
        )
    else:
        adaptive = np.array(
            [
                result.fov_domain.effective_vertical_limit(rho)
                for rho in result.rho_history[:, 1, channel]
            ]
        )

    fig, ax = plt.subplots()
    ax.plot(
        times,
        result.image_history[:, 1, component],
        label=rf"$\alpha_{{{symbol},21}}$",
    )
    ax.plot(times, adaptive, "--", label=rf"$+\alpha_{{{symbol},21}}^a$")
    ax.plot(times, -adaptive, "--", label=rf"$-\alpha_{{{symbol},21}}^a$")
    ax.axhline(1.0, linestyle=":", label=rf"$+\alpha_{symbol}$")
    ax.axhline(-1.0, linestyle=":")
    ax.axhline(
        conservative,
        linestyle="-.",
        label=rf"$+\alpha_{symbol}^c$",
    )
    ax.axhline(-conservative, linestyle="-.")
    ax.set_xlabel(r"$t$ [s]")
    ax.set_ylabel(rf"$\alpha_{{{symbol},21}}$")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    return fig


def plot_normalized_relaxation(result: MissionResult) -> plt.Figure:
    maxima = np.array(
        [
            result.distance_domain.collision_enlargement_max,
            result.distance_domain.range_enlargement_max,
            result.fov_domain.horizontal_enlargement_max,
            result.fov_domain.vertical_enlargement_max,
        ]
    )
    normalized = result.rho_history[:, 1] / maxima
    labels = (
        r"$s_\delta$",
        r"$s_\Delta$",
        r"$s_h$",
        r"$s_v$",
    )
    fig, ax = plt.subplots()
    for k, label in enumerate(labels):
        ax.plot(result.trajectory.times, normalized[:, k], label=label)
    ax.axhline(1.0, linestyle=":", label="physical-span level")
    ax.set_xlabel(r"$t$ [s]")
    ax.set_ylabel("normalized enlargement")
    ax.set_ylim(bottom=-0.02)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    return fig


def print_summary(result: MissionResult) -> None:
    maxima = np.array(
        [
            result.distance_domain.collision_enlargement_max,
            result.distance_domain.range_enlargement_max,
            result.fov_domain.horizontal_enlargement_max,
            result.fov_domain.vertical_enlargement_max,
        ]
    )
    normalized = result.rho_history[:, 1] / maxima
    names = ("collision", "range", "horizontal FoV", "vertical FoV")
    print("\nAdaptive-domain mission summary")
    print("--------------------------------")
    for k, name in enumerate(names):
        index = int(np.nanargmax(normalized[:, k]))
        peak = float(normalized[index, k])
        print(
            f"{name:16s}: peak normalized relaxation {peak:6.3f} "
            f"at t={result.trajectory.times[index]:6.2f} s"
        )

    relative = result.trajectory.positions[:, 0] - result.trajectory.positions[:, 1]
    distance = np.linalg.norm(relative, axis=1)
    ah = np.abs(result.image_history[:, 1, 0])
    av = np.abs(result.image_history[:, 1, 1])
    print(
        "physical margins: "
        f"range={np.min(result.distance_domain.d_max - distance):.3f} m, "
        f"horizontal FoV={np.nanmin(1.0 - ah):.3f}, "
        f"vertical FoV={np.nanmin(1.0 - av):.3f}"
    )
    follower_utilization = np.stack(
        [
            np.asarray(result.allocation.utilization(forces), dtype=float)
            for forces in result.trajectory.controls[:, 1]
        ]
    )
    print(
        "maximum follower thruster utilization: "
        f"{float(np.max(follower_utilization)):.3f}"
    )
    finite = result.required_slacks[np.isfinite(result.required_slacks)]
    if finite.size:
        print(
            "CLF actuation infeasibility: "
            f"max required slack={np.max(finite):.3g}, "
            f"samples={100*np.mean(finite > 1e-10):.2f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=100.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument(
        "--adaptive",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--control-space",
        choices=("thruster", "wrench"),
        default="thruster",
    )
    parser.add_argument("--d-max-conservative", type=float, default=2.4)
    parser.add_argument("--alpha-h-conservative", type=float, default=0.45)
    parser.add_argument("--alpha-v-conservative", type=float, default=0.45)
    parser.add_argument("--thrust-derating", type=float, default=1.0)
    parser.add_argument("--relaxation-recovery-gain", type=float, default=0.8)
    parser.add_argument("--relaxation-barrier-gain", type=float, default=0.20)
    parser.add_argument(
        "--relaxation-domain-margin-ratio",
        type=float,
        default=0.02,
        help=(
            "positive adaptive-domain margin as a fraction of the "
            "conservative-to-physical h-reserve"
        ),
    )
    parser.add_argument(
        "--relaxation-activation-on-ratio",
        type=float,
        default=0.001,
        help=(
            "residual-margin ratio below which the smooth relaxation "
            "activation is fully on"
        ),
    )
    parser.add_argument(
        "--relaxation-activation-off-ratio",
        type=float,
        default=0.1,
        help=(
            "residual-margin ratio above which relaxation is exactly off; "
            "the small default delays activation until close to the "
            "conservative boundary"
        ),
    )
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--no-animation", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--paper-quality", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/bluerov2_adaptive_mission"),
    )
    parser.add_argument(
        "--animation-format",
        choices=("mp4", "gif"),
        default="mp4",
    )
    args = parser.parse_args()

    apply_visualization_style(paper_quality=args.paper_quality)

    result = simulate(
        duration=args.duration,
        dt=args.dt,
        adaptive=args.adaptive,
        control_space=args.control_space,
        d_max_conservative=args.d_max_conservative,
        alpha_h_conservative=args.alpha_h_conservative,
        alpha_v_conservative=args.alpha_v_conservative,
        thrust_derating=args.thrust_derating,
        relaxation_recovery_gain=args.relaxation_recovery_gain,
        relaxation_barrier_gain=args.relaxation_barrier_gain,
        relaxation_domain_margin_ratio=args.relaxation_domain_margin_ratio,
        relaxation_activation_on_ratio=args.relaxation_activation_on_ratio,
        relaxation_activation_off_ratio=args.relaxation_activation_off_ratio,
    )

    print_summary(result)

    figures = {
        "formation_tracking": plot_formation_tracking(result),
        "distance": plot_distance(result),
        "fov_horizontal": plot_fov(result, 0),
        "fov_vertical": plot_fov(result, 1),
        "domain_enlargement": plot_normalized_relaxation(result),
    }

    animation = None
    if not args.no_animation:
        geometry = BlueROV2HeavyVisualGeometry(
            thruster_configuration=result.allocation.configuration
        ).wireframe()
        animation = animate_formation_3d(
            result.trajectory,
            result.scenario.graph,
            desired_position_history=result.desired_positions,
            desired_vehicle_style=DesiredVehicleStyle3D(
                color="0.45",
                alpha=0.20,
                linestyle="--",
                linewidth_scale=0.85,
            ),
            camera=result.camera,
            camera_agents=(1,),
            camera_depth=0.75,
            vehicle_geometry=geometry,
            paper_quality=args.paper_quality,
            show_body_forward=False,
            trail_length=250,
            frame_stride=args.frame_stride,
            title="Adaptive two-BlueROV mission",
        )

    if args.save:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving outputs to: {args.output_dir.resolve()}")
        fmt = "pdf" if args.paper_quality else "png"
        for name, figure in figures.items():
            save_figure(
                figure,
                args.output_dir / f"{name}.{fmt}",
                paper_quality=args.paper_quality,
            )
        if animation is not None:
            save_animation(
                animation.animation,
                args.output_dir / f"formation_animation.{args.animation_format}",
                paper_quality=args.paper_quality,
                fps=30,
            )

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
