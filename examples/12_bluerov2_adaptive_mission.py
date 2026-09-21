#!/usr/bin/env python3
"""Two-BlueROV adaptive-domain mission matching the Gazebo mission geometry.

This paper-oriented pure-Python example is intentionally more mission-like than
the short stress test in example 04.  It uses the same initial positions and
piecewise-constant follower-to-leader formation references as
``two_robot_adaptive_mission.launch.py``.  A leader executes a moderate
translational mission while one follower is commanded through several large
relative-position reconfigurations.  The desired formations remain inside the
deliberately tight conservative sensing domain; relaxation is therefore driven
by transient second-order dynamics and bounded actuation, not by an inadmissible
steady reference.

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


# Keep these values identical to
# ros2/formation_control_ros/launch/two_robot_adaptive_mission.launch.py.
GAZEBO_INITIAL_LEADER_POSITION = np.array([2.675, 0.050, -0.775], dtype=float)
GAZEBO_INITIAL_RELATIVE = np.array([-1.800, -0.700, 0.000], dtype=float)
GAZEBO_INITIAL_FOLLOWER_POSITION = np.array([4.475, 0.750, -0.775], dtype=float)
BASE_RANGE_STRESS_RELATIVE = np.array([-2.30, -0.55, -0.30], dtype=float)



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
    relaxation_gamma: np.ndarray
    relaxation_continuous_rate: np.ndarray
    relaxation_recovery_rate: np.ndarray
    relaxation_expansion_rate: np.ndarray
    relaxation_continuous_state_change: np.ndarray
    predictive_guard_correction: np.ndarray
    emergency_guard_correction: np.ndarray
    generalized_velocities: np.ndarray
    desired_generalized_velocity: np.ndarray
    unlimited_desired_generalized_velocity: np.ndarray
    filtered_generalized_velocity: np.ndarray
    filtered_generalized_acceleration: np.ndarray
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


def build_relaxation_policy(
    distance_domain: DistanceDomain,
    fov_domain: FieldOfViewDomain,
    desired_relative_position: np.ndarray,
    *,
    recovery_gain: float,
    barrier_gain: float,
    activation_on_ratio: float,
    activation_off_ratio: float,
    infeasibility_epsilon: float,
    sampled_guard_margin_ratio: float,
) -> FunnelRelaxationPolicy:
    desired_relative = np.asarray(desired_relative_position, dtype=float)
    desired_image = NormalizedImagePoint(0.0, 0.0)
    desired_values = np.array(
        [
            MinimumDistanceConstraint(
                distance_domain.d_min_conservative,
                squared=distance_domain.squared,
            ).evaluate(desired_relative).value,
            MaximumDistanceConstraint(
                distance_domain.d_max_conservative,
                squared=distance_domain.squared,
            ).evaluate(desired_relative).value,
            HorizontalFieldOfViewConstraint(
                fov_domain.alpha_h_conservative
            ).evaluate(desired_image).value,
            VerticalFieldOfViewConstraint(
                fov_domain.alpha_v_conservative
            ).evaluate(desired_image).value,
        ],
        dtype=float,
    )
    return FunnelRelaxationPolicy(
        maximum_enlargement=np.array(
            [
                distance_domain.collision_enlargement_max,
                distance_domain.range_enlargement_max,
                fov_domain.horizontal_enlargement_max,
                fov_domain.vertical_enlargement_max,
            ],
            dtype=float,
        ),
        desired_conservative_values=desired_values,
        barrier_weights=np.array([0.18, 0.18, 0.25, 0.25], dtype=float),
        recovery_gain=recovery_gain,
        barrier_gain=barrier_gain,
        activation_on_ratio=activation_on_ratio,
        activation_off_ratio=activation_off_ratio,
        infeasibility_epsilon=infeasibility_epsilon,
        minimum_constraint_margin=1e-8,
        sampled_guard_margin_ratio=sampled_guard_margin_ratio,
    )


def _range_stress_relative(distance: float | None) -> np.ndarray:
    if distance is None:
        return BASE_RANGE_STRESS_RELATIVE.copy()
    if distance <= 0.0:
        raise ValueError("range-stress distance must be positive")
    direction = BASE_RANGE_STRESS_RELATIVE / np.linalg.norm(BASE_RANGE_STRESS_RELATIVE)
    return float(distance) * direction


def mission_formations(
    *, range_stress_distance: float | None = None
) -> tuple[MissionFormation, ...]:
    # Parent-minus-follower references.  The default values are exactly the
    # formations configured in two_robot_adaptive_mission.launch.py.  The
    # optional range-stress distance keeps the direction of formation D while
    # moving its equilibrium farther inside the conservative range boundary.
    range_stress_relative = _range_stress_relative(range_stress_distance)
    return (
        MissionFormation(
            "adaptive_A",
            0.0,
            np.array([-1.80, -0.70, 0.00], dtype=float),
        ),
        MissionFormation(
            "adaptive_B",
            18.0,
            np.array([-1.65, -1.45, -0.40], dtype=float),
        ),
        MissionFormation(
            "adaptive_C",
            38.0,
            np.array([-1.65, 1.45, 0.40], dtype=float),
        ),
        MissionFormation(
            "adaptive_D",
            60.0,
            range_stress_relative,
        ),
        MissionFormation(
            "adaptive_A_recover",
            80.0,
            np.array([-1.80, -0.70, 0.00], dtype=float),
        ),
    )


def desired_relative_at(
    time: float, formations: tuple[MissionFormation, ...]
) -> np.ndarray:
    current = formations[0].desired_relative
    for formation in formations:
        if time + 1e-12 >= formation.start:
            current = formation.desired_relative
        else:
            break
    return current.copy()


def leader_velocity_reference(
    time: float,
    *,
    range_stress_relative: np.ndarray,
    range_stress_speed: float | None = None,
) -> np.ndarray:
    # Optional bounded-authority stress maneuver.  During formation D, move the
    # leader outward along the desired parent-minus-follower direction.  This
    # can transiently stretch the range while the desired equilibrium itself
    # remains comfortably inside the conservative range boundary.
    if range_stress_speed is not None and 60.0 <= time < 65.0:
        if range_stress_speed < 0.0:
            raise ValueError("range-stress leader speed must be non-negative")
        direction = np.asarray(range_stress_relative, dtype=float)
        direction = direction / np.linalg.norm(direction)
        return float(range_stress_speed) * direction

    # Moderate nominal mission motion.
    if 12.0 <= time < 32.0:
        return np.array([0.10, 0.08, 0.00])
    if 32.0 <= time < 52.0:
        return np.array([0.08, -0.10, 0.03])
    if 52.0 <= time < 72.0:
        return np.array([-0.08, 0.10, -0.03])
    if 72.0 <= time < 88.0:
        return np.array([-0.06, -0.08, 0.00])
    return np.zeros(3)


def integrate_leader_reference(
    times: np.ndarray,
    *,
    range_stress_relative: np.ndarray,
    range_stress_speed: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    velocity = np.stack(
        [
            leader_velocity_reference(
                float(t),
                range_stress_relative=range_stress_relative,
                range_stress_speed=range_stress_speed,
            )
            for t in times
        ]
    )
    position = np.zeros((times.size, 3), dtype=float)
    position[0] = GAZEBO_INITIAL_LEADER_POSITION
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
    virtual_linear_gain: float,
    virtual_angular_gain: float,
    command_filter_linear_bandwidth: float,
    command_filter_angular_bandwidth: float,
    alpha_gain: float,
    relaxation_recovery_gain: float,
    relaxation_barrier_gain: float,
    relaxation_activation_on_ratio: float,
    relaxation_activation_off_ratio: float,
    relaxation_infeasibility_epsilon: float,
    range_stress_distance: float | None = None,
    range_stress_leader_speed: float | None = None,
    sampled_guard_margin_ratio: float = 0.02,
) -> MissionResult:
    graph = DirectedSensingGraph.rooted_star(2)
    formations = mission_formations(range_stress_distance=range_stress_distance)
    range_stress_relative = formations[3].desired_relative
    initial_relative = GAZEBO_INITIAL_RELATIVE.copy()
    initial_leader = GAZEBO_INITIAL_LEADER_POSITION.copy()
    initial_follower = GAZEBO_INITIAL_FOLLOWER_POSITION.copy()
    if not np.allclose(initial_leader - initial_follower, initial_relative):
        raise RuntimeError("Gazebo initial positions and relative reference are inconsistent.")

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
        virtual_gain=np.diag(
            [
                virtual_linear_gain,
                virtual_linear_gain,
                virtual_linear_gain,
                virtual_angular_gain,
                virtual_angular_gain,
                virtual_angular_gain,
            ]
        ),
        virtual_velocity_norm_limits=np.array([1.5, 2.0]),
        filter_bandwidth=np.array(
            [
                command_filter_linear_bandwidth,
                command_filter_linear_bandwidth,
                command_filter_linear_bandwidth,
                command_filter_angular_bandwidth,
                command_filter_angular_bandwidth,
                command_filter_angular_bandwidth,
            ],
            dtype=float,
        ),
        alpha_gain=alpha_gain,
        slack_linear_penalty=100.0,
        slack_penalty=5e3,
    )
    controller = design.agent_controller

    relaxation = build_relaxation_policy(
        distance_domain,
        fov_domain,
        initial_relative,
        recovery_gain=relaxation_recovery_gain,
        barrier_gain=relaxation_barrier_gain,
        activation_on_ratio=relaxation_activation_on_ratio,
        activation_off_ratio=relaxation_activation_off_ratio,
        infeasibility_epsilon=relaxation_infeasibility_epsilon,
        sampled_guard_margin_ratio=sampled_guard_margin_ratio,
    )
    enabled = np.ones(4, dtype=bool)

    states = np.zeros((2, 13), dtype=float)
    states[0, :3] = initial_leader
    states[0, 3:7] = quaternion_from_roll_pitch_yaw(0.0, 0.0, 0.0)
    states[1, :3] = initial_follower
    states[1, 3:7] = look_at_quaternion(initial_relative)

    relaxation_state = np.zeros((2, 4), dtype=float)
    previous_conservative_values = np.full(4, np.nan, dtype=float)
    predictive_guard_count = 0
    predictive_guard_maximum_correction = 0.0
    emergency_guard_count = 0
    emergency_guard_maximum_correction = 0.0
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
        relaxation_state[1] = relaxation.initialize_for_constraint_values(
            kinematics.values,
            enabled=enabled,
        )

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
    leader_ref_position, leader_ref_velocity = integrate_leader_reference(
        times,
        range_stress_relative=range_stress_relative,
        range_stress_speed=range_stress_leader_speed,
    )

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
    relaxation_gamma = np.zeros(steps, dtype=float)
    relaxation_continuous_rate = np.zeros((steps, 4), dtype=float)
    relaxation_recovery_rate = np.zeros((steps, 4), dtype=float)
    relaxation_expansion_rate = np.zeros((steps, 4), dtype=float)
    relaxation_continuous_state_change = np.zeros((steps, 4), dtype=float)
    predictive_guard_correction = np.zeros((steps, 4), dtype=float)
    emergency_guard_correction = np.zeros((steps, 4), dtype=float)
    generalized_velocities = np.empty((steps + 1, 2, 6), dtype=float)
    desired_generalized_velocity = np.empty((steps, 6), dtype=float)
    unlimited_desired_generalized_velocity = np.empty((steps, 6), dtype=float)
    filtered_generalized_velocity = np.empty((steps, 6), dtype=float)
    filtered_generalized_acceleration = np.empty((steps, 6), dtype=float)

    plant_integrator = RK4Integrator()
    filter_integrator = RK4Integrator()

    def record(sample: int) -> None:
        positions[sample] = states[:, :3]
        quaternions[sample] = states[:, 3:7]
        velocities[sample, 0] = inertial_velocity(model, states[0])
        velocities[sample, 1] = inertial_velocity(model, states[1])
        generalized_velocities[sample, 0] = model.split_state(states[0])[2]
        generalized_velocities[sample, 1] = model.split_state(states[1])[2]
        rho_history[sample, 1] = relaxation.enlargement(relaxation_state[1])
        desired_relative_history[sample] = desired_relative_at(times[sample], formations)
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
        desired_relative = desired_relative_at(time, formations)
        kinematics = evaluate_sensing_constraint_kinematics(
            model,
            camera,
            distance_domain,
            fov_domain,
            states[1],
            states[0],
        )
        if np.all(np.isfinite(previous_conservative_values)):
            conservative_rates = (kinematics.values - previous_conservative_values) / dt
        else:
            conservative_rates = np.zeros(4, dtype=float)
        previous_conservative_values = kinematics.values.copy()

        relaxation = relaxation.with_desired_conservative_values(
            build_relaxation_policy(
                distance_domain,
                fov_domain,
                desired_relative,
                recovery_gain=relaxation_recovery_gain,
                barrier_gain=relaxation_barrier_gain,
                activation_on_ratio=relaxation_activation_on_ratio,
                activation_off_ratio=relaxation_activation_off_ratio,
                infeasibility_epsilon=relaxation_infeasibility_epsilon,
                sampled_guard_margin_ratio=sampled_guard_margin_ratio,
            ).desired_conservative_values
        )

        if adaptive:
            relaxation_state[1], emergency_correction = relaxation.project_to_current_domain(
                relaxation_state[1],
                kinematics.values,
                enabled=enabled,
            )
            emergency_guard_correction[step] = emergency_correction
            if np.any(emergency_correction > 0.0):
                emergency_guard_count += 1
                emergency_guard_maximum_correction = max(
                    emergency_guard_maximum_correction,
                    float(np.max(emergency_correction)),
                )
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
        desired_generalized_velocity[step] = evaluation.controller.desired_velocity
        unlimited_desired_generalized_velocity[step] = (
            evaluation.controller.unlimited_desired_velocity
        )
        filtered_generalized_velocity[step] = evaluation.controller.filter.output
        filtered_generalized_acceleration[step] = (
            evaluation.controller.filter.output_derivative
        )
        slacks[step, 1] = evaluation.slack
        if evaluation.required_slack is not None:
            required_slacks[step, 1] = evaluation.required_slack

        if adaptive:
            required = (
                0.0
                if evaluation.required_slack is None
                else max(float(evaluation.required_slack), 0.0)
            )

            # Diagnose the paper continuous-time adaptive law separately from
            # the implementation-only sampled-data safeguards.  This makes it
            # explicit whether relaxation is driven by CLF actuation
            # infeasibility (gamma > 0) or by a sampled guard.
            relaxation_evaluation = relaxation.evaluate(
                relaxation_state[1],
                conservative_values=kinematics.values,
                required_slack=required,
                enabled=enabled,
            )
            relaxation_gamma[step] = relaxation_evaluation.infeasibility_activation
            relaxation_continuous_rate[step] = relaxation_evaluation.selected_rate
            relaxation_recovery_rate[step] = relaxation_evaluation.recovery_rate
            relaxation_expansion_rate[step] = relaxation_evaluation.enlargement_rate

            state_before_continuous_step = relaxation_state[1].copy()
            relaxation_state[1], _ = relaxation.advance(
                relaxation_state[1],
                conservative_values=kinematics.values,
                required_slack=required,
                enabled=enabled,
                sample_time=dt,
            )
            relaxation_continuous_state_change[step] = (
                relaxation_state[1] - state_before_continuous_step
            )

            relaxation_state[1], predictive_correction = (
                relaxation.project_to_predicted_domain(
                    relaxation_state[1],
                    kinematics.values,
                    conservative_rates,
                    sample_time=dt,
                    enabled=enabled,
                )
            )
            predictive_guard_correction[step] = predictive_correction
            if np.any(predictive_correction > 0.0):
                predictive_guard_count += 1
                predictive_guard_maximum_correction = max(
                    predictive_guard_maximum_correction,
                    float(np.max(predictive_correction)),
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

    if adaptive:
        print(
            "Predictive sampled guard: "
            f"{predictive_guard_count} adjusted follower-samples, "
            "maximum normalized adjustment "
            f"{predictive_guard_maximum_correction:.3e}"
        )
        print(
            "Emergency sampled projection: "
            f"{emergency_guard_count} adjusted follower-samples, "
            "maximum normalized adjustment "
            f"{emergency_guard_maximum_correction:.3e}"
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
        relaxation_gamma=relaxation_gamma,
        relaxation_continuous_rate=relaxation_continuous_rate,
        relaxation_recovery_rate=relaxation_recovery_rate,
        relaxation_expansion_rate=relaxation_expansion_rate,
        relaxation_continuous_state_change=relaxation_continuous_state_change,
        predictive_guard_correction=predictive_guard_correction,
        emergency_guard_correction=emergency_guard_correction,
        generalized_velocities=generalized_velocities,
        desired_generalized_velocity=desired_generalized_velocity,
        unlimited_desired_generalized_velocity=unlimited_desired_generalized_velocity,
        filtered_generalized_velocity=filtered_generalized_velocity,
        filtered_generalized_acceleration=filtered_generalized_acceleration,
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


def plot_generalized_velocity_diagnostics(
    result: MissionResult,
    *,
    angular: bool,
) -> plt.Figure:
    """Compare actual, desired, raw-desired, and filtered follower velocities."""
    if angular:
        indices = range(3, 6)
        symbols = ("p", "q", "r")
        ylabel = "angular velocity [rad/s]"
    else:
        indices = range(3)
        symbols = ("u", "v", "w")
        ylabel = "linear velocity [m/s]"

    sample_times = result.trajectory.times
    control_times = sample_times[:-1]
    actual = result.generalized_velocities[:, 1]
    desired = result.desired_generalized_velocity
    unlimited = result.unlimited_desired_generalized_velocity
    filtered = result.filtered_generalized_velocity

    fig, axes = plt.subplots(3, 1, sharex=True)
    for axis, index, symbol in zip(axes, indices, symbols, strict=True):
        axis.plot(sample_times, actual[:, index], label=rf"${symbol}_i$")
        axis.plot(
            control_times,
            desired[:, index],
            "--",
            label=rf"${symbol}_{{i,d}}$",
        )
        axis.plot(
            control_times,
            filtered[:, index],
            "-.",
            label=rf"${symbol}_{{i,f}}$",
        )
        if not np.allclose(
            unlimited[:, index],
            desired[:, index],
            rtol=1e-8,
            atol=1e-10,
        ):
            axis.plot(
                control_times,
                unlimited[:, index],
                ":",
                label=rf"${symbol}_{{i,d,raw}}$",
            )
        axis.grid(True, alpha=0.3)
        axis.legend(ncol=4, loc="best")

    axes[-1].set_xlabel(r"$t$ [s]")
    axes[1].set_ylabel(ylabel)
    fig.tight_layout()
    return fig


def plot_thruster_diagnostics(result: MissionResult) -> plt.Figure:
    """Plot follower thruster forces and aggregate bound utilization."""
    times = result.trajectory.times[:-1]
    forces = result.trajectory.controls[:, 1]
    utilization = np.stack(
        [
            np.asarray(result.allocation.utilization(sample), dtype=float)
            for sample in forces
        ]
    )
    maximum_utilization = np.max(utilization, axis=1)

    fig, axes = plt.subplots(2, 1, sharex=True)
    for k in range(forces.shape[1]):
        axes[0].plot(times, forces[:, k], label=rf"$f_{{{k + 1}}}$")
    axes[0].set_ylabel("thruster force [N]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=4)

    axes[1].plot(times, maximum_utilization, label="maximum utilization")
    axes[1].axhline(1.0, linestyle=":", label="thruster limit")
    axes[1].set_xlabel(r"$t$ [s]")
    axes[1].set_ylabel("utilization")
    axes[1].set_ylim(bottom=0.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    return fig



def plot_relaxation_mechanism_diagnostics(
    result: MissionResult,
    *,
    channel: int = 1,
) -> plt.Figure:
    """Separate paper-law adaptation from sampled-data safeguard action.

    The default ``channel=1`` is the maximum-range channel ``s_Delta``, which
    is the active adaptive channel in this mission.  The figure exposes the
    CLF-required relaxation and gate, the continuous adaptive-law rate, and the
    two implementation-only sampled-data corrections.
    """
    if channel not in range(4):
        raise ValueError("channel must be one of 0, 1, 2, 3.")

    times = result.trajectory.times[:-1]
    required = np.nan_to_num(
        result.required_slacks[:, 1],
        nan=0.0,
        posinf=np.nan,
        neginf=np.nan,
    )
    channel_symbols = (r"\delta", r"\Delta", "h", "v")
    symbol = channel_symbols[channel]

    fig, axes = plt.subplots(4, 1, sharex=True)

    axes[0].plot(times, required, label=r"$\delta_{\mathrm{req}}$")
    axes[0].set_ylabel(r"$\delta_{\mathrm{req}}$")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(times, result.relaxation_gamma, label=r"$\gamma$")
    axes[1].set_ylabel(r"$\gamma$")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(
        times,
        result.relaxation_continuous_rate[:, channel],
        label=rf"$\dot s_{{{symbol}}}$",
    )
    axes[2].plot(
        times,
        result.relaxation_expansion_rate[:, channel],
        "--",
        label=rf"$\dot s_{{{symbol}}}^{{\mathrm{{exp}}}}$",
    )
    axes[2].plot(
        times,
        result.relaxation_recovery_rate[:, channel],
        ":",
        label=rf"$\dot s_{{{symbol}}}^{{\mathrm{{rec}}}}$",
    )
    axes[2].set_ylabel(r"continuous rate [s$^{-1}$]")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(ncol=3)

    axes[3].plot(
        times,
        result.predictive_guard_correction[:, channel],
        label="predictive guard",
    )
    axes[3].plot(
        times,
        result.emergency_guard_correction[:, channel],
        "--",
        label="emergency projection",
    )
    axes[3].set_xlabel(r"$t$ [s]")
    axes[3].set_ylabel(rf"$\Delta s_{{{symbol}}}$")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

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
    peak_overall = float(np.nanmax(normalized))
    triggered = peak_overall > 1e-4
    print(
        "adaptive relaxation triggered: "
        f"{'YES' if triggered else 'NO'} "
        f"(max s={peak_overall:.3f})"
    )
    for k, name in enumerate(names):
        index = int(np.nanargmax(normalized[:, k]))
        peak = float(normalized[index, k])
        active = np.flatnonzero(normalized[:, k] > 1e-4)
        first = "-" if active.size == 0 else f"{result.trajectory.times[int(active[0])]:.2f} s"
        print(
            f"{name:16s}: peak s={peak:6.3f} at "
            f"t={result.trajectory.times[index]:6.2f} s; first activation={first}"
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
        "follower thruster bounds: "
        f"[{float(result.allocation.lower_bounds[0]):.3f}, "
        f"{float(result.allocation.upper_bounds[0]):.3f}] N"
    )
    print(
        "maximum follower thruster utilization: "
        f"{float(np.max(follower_utilization)):.3f}"
    )
    follower_required = result.required_slacks[:, 1]
    finite = follower_required[np.isfinite(follower_required)]
    if finite.size:
        max_index = int(np.nanargmax(follower_required))
        print(
            "CLF actuation infeasibility: "
            f"max required slack={np.max(finite):.3g} "
            f"at t={result.trajectory.times[max_index]:.2f} s, "
            f"samples={100*np.mean(finite > 1e-10):.2f}%"
        )

    range_channel = 1
    continuous_positive = np.maximum(
        result.relaxation_continuous_state_change[:, range_channel],
        0.0,
    )
    predictive_positive = np.maximum(
        result.predictive_guard_correction[:, range_channel],
        0.0,
    )
    emergency_positive = np.maximum(
        result.emergency_guard_correction[:, range_channel],
        0.0,
    )
    print(
        "range-relaxation source: "
        f"max gamma={float(np.max(result.relaxation_gamma)):.3f}, "
        f"continuous-expansion samples={int(np.count_nonzero(continuous_positive > 1e-12))}, "
        f"sum positive continuous Delta-s={float(np.sum(continuous_positive)):.3e}, "
        f"sum predictive Delta-s={float(np.sum(predictive_positive)):.3e}, "
        f"sum emergency Delta-s={float(np.sum(emergency_positive)):.3e}"
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
    parser.add_argument(
        "--thrust-derating",
        type=float,
        default=1.0,
        help=(
            "uniform scale applied to the BlueROV2 T200 forward/reverse "
            "force bounds; e.g. 0.4 gives 40% of nominal authority"
        ),
    )
    parser.add_argument(
        "--virtual-linear-gain",
        type=float,
        default=0.55,
        help="linear part of K_eta; reduce this to soften translational barrier response",
    )
    parser.add_argument(
        "--virtual-angular-gain",
        type=float,
        default=0.80,
        help="angular part of K_eta",
    )
    parser.add_argument(
        "--command-filter-linear-bandwidth",
        type=float,
        default=3.0,
        help="linear command-filter bandwidth [rad/s]; smaller values smooth sharp virtual commands",
    )
    parser.add_argument(
        "--command-filter-angular-bandwidth",
        type=float,
        default=4.0,
        help="angular command-filter bandwidth [rad/s]",
    )
    parser.add_argument(
        "--alpha-gain",
        type=float,
        default=0.8,
        help="CLF decay gain",
    )
    parser.add_argument("--relaxation-recovery-gain", type=float, default=0.8)
    parser.add_argument("--relaxation-barrier-gain", type=float, default=0.20)
    parser.add_argument(
        "--relaxation-activation-on-ratio",
        type=float,
        default=0.10,
        help=(
            "h_on / h_d,c: activation is fully on below this adaptive margin"
        ),
    )
    parser.add_argument(
        "--relaxation-activation-off-ratio",
        type=float,
        default=0.30,
        help=(
            "h_off / h_d,c: activation is exactly off above this margin"
        ),
    )
    parser.add_argument(
        "--relaxation-infeasibility-epsilon",
        type=float,
        default=1e-3,
        help="epsilon_delta in the required-slack gate gamma",
    )
    parser.add_argument(
        "--sampled-guard-margin-ratio",
        type=float,
        default=0.02,
        help=(
            "implementation-only sampled guard h_guard / rho_bar; "
            "the default is preserved, while smaller values let the "
            "continuous adaptive law act closer to the barrier"
        ),
    )
    parser.add_argument(
        "--range-stress-distance",
        type=float,
        default=None,
        help=(
            "optional norm [m] of formation D, preserving its direction; "
            "the default keeps the original Gazebo-aligned D reference"
        ),
    )
    parser.add_argument(
        "--range-stress-leader-speed",
        type=float,
        default=None,
        help=(
            "optional leader speed [m/s] from t=60 to 65 s along the "
            "formation-D direction; used to demonstrate authority-triggered "
            "range relaxation with an interior equilibrium"
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

    if args.range_stress_distance is not None or args.range_stress_leader_speed is not None:
        stress_relative = _range_stress_relative(args.range_stress_distance)
        print(
            "Range-stress override: "
            f"||p_21,d^D||={np.linalg.norm(stress_relative):.3f} m, "
            f"leader speed={0.0 if args.range_stress_leader_speed is None else args.range_stress_leader_speed:.3f} m/s "
            "during t in [60, 65) s"
        )

    result = simulate(
        duration=args.duration,
        dt=args.dt,
        adaptive=args.adaptive,
        control_space=args.control_space,
        d_max_conservative=args.d_max_conservative,
        alpha_h_conservative=args.alpha_h_conservative,
        alpha_v_conservative=args.alpha_v_conservative,
        thrust_derating=args.thrust_derating,
        virtual_linear_gain=args.virtual_linear_gain,
        virtual_angular_gain=args.virtual_angular_gain,
        command_filter_linear_bandwidth=args.command_filter_linear_bandwidth,
        command_filter_angular_bandwidth=args.command_filter_angular_bandwidth,
        alpha_gain=args.alpha_gain,
        relaxation_recovery_gain=args.relaxation_recovery_gain,
        relaxation_barrier_gain=args.relaxation_barrier_gain,
        relaxation_activation_on_ratio=args.relaxation_activation_on_ratio,
        relaxation_activation_off_ratio=args.relaxation_activation_off_ratio,
        relaxation_infeasibility_epsilon=args.relaxation_infeasibility_epsilon,
        range_stress_distance=args.range_stress_distance,
        range_stress_leader_speed=args.range_stress_leader_speed,
        sampled_guard_margin_ratio=args.sampled_guard_margin_ratio,
    )

    print_summary(result)

    figures = {
        "formation_tracking": plot_formation_tracking(result),
        "distance": plot_distance(result),
        "fov_horizontal": plot_fov(result, 0),
        "fov_vertical": plot_fov(result, 1),
        "domain_enlargement": plot_normalized_relaxation(result),
        "relaxation_mechanism_diagnostics": plot_relaxation_mechanism_diagnostics(
            result, channel=1
        ),
        "velocity_linear_body": plot_generalized_velocity_diagnostics(
            result, angular=False
        ),
        "velocity_angular_body": plot_generalized_velocity_diagnostics(
            result, angular=True
        ),
        "thruster_diagnostics": plot_thruster_diagnostics(result),
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
