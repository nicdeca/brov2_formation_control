"""Helpers for controlled actuation-stress initial conditions."""

from __future__ import annotations

import numpy as np

from formation_control.geometry import rotation_matrix_from_quaternion
from formation_control.models import BlueROV2Model
from formation_control.models.base import FloatArray


def outward_unit_vector(
    observer_position: FloatArray,
    target_position: FloatArray,
) -> FloatArray:
    """Return the inertial unit vector pointing from target to observer."""
    observer = np.asarray(observer_position, dtype=float)
    target = np.asarray(target_position, dtype=float)

    if observer.shape != (3,) or target.shape != (3,):
        raise ValueError("positions must have shape (3,).")

    vector = observer - target
    norm = float(np.linalg.norm(vector))
    if norm <= np.finfo(float).eps:
        raise ValueError("observer and target positions must be distinct.")

    return vector / norm


def set_inertial_linear_velocity(
    model: BlueROV2Model,
    state: FloatArray,
    velocity_inertial: FloatArray,
) -> FloatArray:
    """Return a state whose body linear velocity realizes an inertial velocity."""
    state_array = np.asarray(state, dtype=float)
    velocity = np.asarray(velocity_inertial, dtype=float)

    if state_array.shape != (model.state_dim,):
        raise ValueError(f"state must have shape ({model.state_dim},), got {state_array.shape}.")
    if velocity.shape != (3,):
        raise ValueError("velocity_inertial must have shape (3,).")

    updated = state_array.copy()
    _, quaternion, _ = model.split_state(updated)
    rotation = rotation_matrix_from_quaternion(quaternion)
    updated[7:10] = rotation.T @ velocity
    return updated


def set_outward_linear_velocity(
    model: BlueROV2Model,
    state: FloatArray,
    target_position: FloatArray,
    *,
    speed: float,
) -> FloatArray:
    """Return a state moving directly away from the target in inertial space."""
    if not np.isfinite(speed) or speed < 0.0:
        raise ValueError("speed must be finite and nonnegative.")

    direction = outward_unit_vector(
        np.asarray(state, dtype=float)[:3],
        target_position,
    )
    return set_inertial_linear_velocity(
        model,
        state,
        speed * direction,
    )
