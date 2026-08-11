"""Build plot-complete diagnostic dictionaries from BlueROV2 evaluations.

This file is deliberately ROS-independent.  It mirrors the detailed CLF
bookkeeping already done by ``examples/04_bluerov2_fov_clf_qp.py`` so that a
ROS/SITL/hardware run can be fed back into exactly the same offline plotters.
The helper only *extracts* an evaluation that has already been computed; it
never reruns the controller.
"""

from __future__ import annotations

import numpy as np


def _domain_maxima(distance_domain, fov_domain) -> np.ndarray:
    return np.array(
        [
            distance_domain.collision_enlargement_max,
            distance_domain.range_enlargement_max,
            fov_domain.horizontal_enlargement_max,
            fov_domain.vertical_enlargement_max,
        ],
        dtype=float,
    )


def follower_snapshot_values(
    *,
    model,
    allocation,
    camera,
    follower_state: np.ndarray,
    parent_position: np.ndarray,
    parent_velocity_inertial: np.ndarray,
    desired_relative_position: np.ndarray,
    evaluation,
    conservative_values: np.ndarray,
    relaxation_state: np.ndarray,
    relaxation_rate: np.ndarray,
    controller_time_s: float,
    fallback: bool,
    distance_domain,
    fov_domain,
    constraints_enabled: np.ndarray,
    adaptive_enabled: bool,
    domain_margin_ratio: float,
) -> dict[str, object]:
    """Return every follower quantity used by the existing diagnostic plots.

    Frame convention is the controller's native convention: world NWU, body
    FLU, scalar-first quaternions. ``parent_velocity_inertial`` must be the
    translational parent velocity used for the parent term in the CLF drift.
    """
    state = np.asarray(follower_state, dtype=float)
    parent_position = np.asarray(parent_position, dtype=float)
    parent_velocity = np.asarray(parent_velocity_inertial, dtype=float)
    conservative_values = np.asarray(conservative_values, dtype=float)
    relaxation_state = np.asarray(relaxation_state, dtype=float)
    relaxation_rate = np.asarray(relaxation_rate, dtype=float)

    if conservative_values.shape != (4,):
        raise ValueError("conservative_values must have shape (4,).")
    if relaxation_state.shape != (4,):
        raise ValueError("relaxation_state must have shape (4,).")
    if relaxation_rate.shape != (4,):
        raise ValueError("relaxation_rate must have shape (4,).")

    maxima = _domain_maxima(distance_domain, fov_domain)
    physical_values = conservative_values + maxima

    # The potential already owns the camera observation used by the controller.
    # Prefer that exact value if available; otherwise recompute from the same
    # state geometry as a diagnostic only.
    try:
        image_point = evaluation.potential.image_point
        image_coordinates = np.asarray(image_point.as_array(), dtype=float)
    except (AttributeError, TypeError):
        position, quaternion, _ = model.split_state(state)
        from formation_control.geometry import rotation_matrix_from_quaternion

        rotation = rotation_matrix_from_quaternion(quaternion)
        image_coordinates = camera.observe(
            observer_position=position,
            observer_rotation=rotation,
            target_position=parent_position,
        ).image_point.as_array()

    clf = evaluation.controller.clf
    qp_result = evaluation.controller.qp
    generalized_velocity = np.asarray(model.split_state(state)[2], dtype=float)
    zeta = np.asarray(evaluation.generalized_configuration_gradient, dtype=float)

    configuration_local_rate = float(zeta @ generalized_velocity)
    parent_rate = float(
        evaluation.potential.target_position_gradient @ parent_velocity
    )

    dynamics_bias = np.asarray(model.drift_wrench(state), dtype=float)
    filtered_velocity = np.asarray(evaluation.controller.filter.output, dtype=float)
    filtered_velocity_derivative = np.asarray(
        evaluation.controller.filter.output_derivative,
        dtype=float,
    )
    desired_velocity = np.asarray(
        evaluation.controller.desired_velocity,
        dtype=float,
    )
    velocity_error = np.asarray(clf.velocity_error, dtype=float)
    command_acceleration_wrench = (
        np.asarray(model.mass_matrix, dtype=float) @ filtered_velocity_derivative
    )

    dynamics_bias_linear_rate = float(
        -velocity_error[:3] @ dynamics_bias[:3]
    )
    dynamics_bias_angular_rate = float(
        -velocity_error[3:] @ dynamics_bias[3:]
    )
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
    velocity_backstepping_rate = dynamics_bias_rate + command_acceleration_rate

    # CLFQPResult stores -alpha(W)+delta. Recover alpha(W) without reaching
    # into private controller internals.
    decay = float(qp_result.slack - qp_result.desired_clf_upper_bound)

    feasibility = qp_result.actuation_feasibility
    required_slack = np.nan
    actuation_margin = np.nan
    best_actuator_contribution = np.nan
    minimum_modeled_derivative = np.nan
    hard_clf_residual = np.nan

    if feasibility is not None:
        required_slack = float(feasibility.required_slack)
        actuation_margin = float(feasibility.margin)
        best_actuator_contribution = float(
            clf.control_gradient @ feasibility.minimizing_control
        )
        minimum_modeled_derivative = float(
            feasibility.minimum_modeled_derivative
        )
        hard_clf_residual = float(
            clf.drift + decay + best_actuator_contribution
        )

    optimized_input = np.asarray(evaluation.optimized_input, dtype=float)
    if optimized_input.shape != (8,):
        raise ValueError(
            "plot-complete online logging requires thruster-space control "
            f"with 8 inputs, got {optimized_input.shape}."
        )

    distance = float(np.linalg.norm(parent_position - state[:3]))
    constraints_enabled = np.asarray(constraints_enabled, dtype=bool)
    if constraints_enabled.shape != (4,):
        raise ValueError("constraints_enabled must have shape (4,).")
    coordinate_margins = np.array(
        [
            distance - distance_domain.d_min,
            distance_domain.d_max - distance,
            1.0 - abs(float(image_coordinates[0])),
            1.0 - abs(float(image_coordinates[1])),
        ],
        dtype=float,
    )
    enabled_margins = coordinate_margins[constraints_enabled]
    minimum_physical_margin = (
        float(np.min(enabled_margins)) if enabled_margins.size else np.nan
    )

    return {
        "role": 0.0,
        "fallback": float(bool(fallback)),
        "controller_time_s": float(controller_time_s),
        "slack": float(qp_result.slack),
        "required_slack": required_slack,
        "actuation_margin": actuation_margin,
        "thruster_utilization": float(np.max(allocation.utilization(optimized_input))),
        "minimum_physical_margin": minimum_physical_margin,
        "distance_domain": np.array(
            [
                distance_domain.d_min,
                distance_domain.d_max,
                distance_domain.d_min_conservative,
                distance_domain.d_max_conservative,
            ],
            dtype=float,
        ),
        "fov_domain": np.array(
            [
                fov_domain.alpha_h_conservative,
                fov_domain.alpha_v_conservative,
            ],
            dtype=float,
        ),
        "constraints_enabled": np.asarray(constraints_enabled, dtype=float),
        "adaptive_enabled": float(bool(adaptive_enabled)),
        "domain_margin_ratio": float(domain_margin_ratio),
        "camera_half_angles": np.array(
            [camera.horizontal_half_angle, camera.vertical_half_angle],
            dtype=float,
        ),
        "camera_position_body": camera.extrinsics.position_camera_in_body,
        "camera_rotation_camera_from_body": (
            camera.extrinsics.rotation_camera_from_body.reshape(-1)
        ),
        "conservative_values": conservative_values,
        "physical_values": physical_values,
        "relaxation_state": relaxation_state,
        "relaxation_rate": relaxation_rate,
        "image_coordinates": image_coordinates,
        "desired_relative_position": np.asarray(
            desired_relative_position,
            dtype=float,
        ),
        "relative_position": parent_position - state[:3],
        "thruster_forces": optimized_input,
        "thruster_force_limits": np.array(
            [
                float(-allocation.lower_bounds[0]),
                float(allocation.upper_bounds[0]),
            ],
            dtype=float,
        ),
        "body_wrench": np.asarray(evaluation.wrench_body, dtype=float),
        "clf_value": float(clf.value),
        "clf_decay": decay,
        "clf_drift": float(clf.drift),
        "clf_configuration_local_rate": configuration_local_rate,
        "clf_parent_rate": parent_rate,
        "clf_velocity_backstepping_rate": velocity_backstepping_rate,
        "clf_dynamics_bias_rate": dynamics_bias_rate,
        "clf_dynamics_bias_linear_rate": dynamics_bias_linear_rate,
        "clf_dynamics_bias_angular_rate": dynamics_bias_angular_rate,
        "clf_command_acceleration_rate": command_acceleration_rate,
        "clf_command_acceleration_linear_rate": (
            command_acceleration_linear_rate
        ),
        "clf_command_acceleration_angular_rate": (
            command_acceleration_angular_rate
        ),
        "clf_velocity_error_norm": float(np.linalg.norm(velocity_error)),
        "clf_velocity_error_linear_norm": float(
            np.linalg.norm(velocity_error[:3])
        ),
        "clf_velocity_error_angular_norm": float(
            np.linalg.norm(velocity_error[3:])
        ),
        "clf_filtered_velocity_derivative_norm": float(
            np.linalg.norm(filtered_velocity_derivative)
        ),
        "clf_filtered_linear_acceleration_norm": float(
            np.linalg.norm(filtered_velocity_derivative[:3])
        ),
        "clf_filtered_angular_acceleration_norm": float(
            np.linalg.norm(filtered_velocity_derivative[3:])
        ),
        "clf_best_actuator_contribution": best_actuator_contribution,
        "clf_minimum_modeled_derivative": minimum_modeled_derivative,
        "clf_hard_residual": hard_clf_residual,
        "generalized_velocity": generalized_velocity,
        "filtered_velocity": filtered_velocity,
        "desired_velocity": desired_velocity,
        "filtered_velocity_derivative": filtered_velocity_derivative,
        "dynamics_bias": dynamics_bias,
    }


def leader_snapshot_values(
    *,
    thruster_forces: np.ndarray,
    body_wrench: np.ndarray,
    controller_time_s: float,
    fallback: bool,
    allocation,
    reference_position: np.ndarray | None = None,
    reference_velocity: np.ndarray | None = None,
    reference_acceleration: np.ndarray | None = None,
    reference_quaternion: np.ndarray | None = None,
    extra_values: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build a leader snapshot.

    ``extra_values`` lets the leader runtime expose its own CLF diagnostics
    without changing the serialization schema.  This is useful if the leader
    tracking controller already computes the same quantities as a follower.
    """
    forces = np.asarray(thruster_forces, dtype=float)
    if forces.shape != (8,):
        raise ValueError("thruster_forces must have shape (8,).")

    values: dict[str, object] = {
        "role": 1.0,
        "fallback": float(bool(fallback)),
        "controller_time_s": float(controller_time_s),
        "thruster_forces": forces,
        "body_wrench": np.asarray(body_wrench, dtype=float),
        "thruster_utilization": float(np.max(allocation.utilization(forces))),
        "thruster_force_limits": np.array(
            [
                float(-allocation.lower_bounds[0]),
                float(allocation.upper_bounds[0]),
            ],
            dtype=float,
        ),
    }
    for name, value in (
        ("reference_position", reference_position),
        ("reference_velocity", reference_velocity),
        ("reference_acceleration", reference_acceleration),
        ("reference_quaternion", reference_quaternion),
    ):
        if value is not None:
            values[name] = np.asarray(value, dtype=float)

    if extra_values:
        values.update(extra_values)
    return values
