"""Leader tracking adapter for the BlueROV2 CLF-QP framework.

The leader uses exactly the same dynamic controller as the followers:

    potential -> generalized gradient -> smooth virtual-twist saturation
    -> command filter -> backstepping CLF -> actuator-constrained CLF-QP.

Only the configuration potential and feedforward signals differ.

For a translational reference ``(p_r, v_r, a_r)`` and a fixed desired
attitude ``R_r``, the position potential is

    V_p = 0.5 * (p - p_r).T K_p (p - p_r),

and the feedforward body twist is

    nu_ff = [R.T v_r, 0].

Its derivative is computed exactly as

    d/dt (R.T v_r) = R.T a_r - omega x (R.T v_r).

Thus a full position/velocity/acceleration trajectory can be used without
differentiating the reference numerically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from formation_control.geometry import rotation_matrix_from_quaternion
from formation_control.models import BlueROV2Model
from formation_control.models.base import FloatArray

from .second_order import (
    SecondOrderCLFQPController,
    SecondOrderControllerEvaluation,
)


def _vector3(value: FloatArray, *, name: str) -> FloatArray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {vector.shape}.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")
    return vector


def _vector6(value: FloatArray, *, name: str) -> FloatArray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (6,):
        raise ValueError(f"{name} must have shape (6,), got {vector.shape}.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")
    return vector


def _rotation(value: FloatArray, *, name: str) -> FloatArray:
    rotation = np.asarray(value, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3).")
    if not np.all(np.isfinite(rotation)):
        raise ValueError(f"{name} must contain only finite values.")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8):
        raise ValueError(f"{name} must be orthogonal.")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8):
        raise ValueError(f"{name} must have determinant one.")
    return rotation


def _positive_definite_matrix(
    value: FloatArray,
    *,
    name: str,
) -> FloatArray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3).")
    if not np.allclose(matrix, matrix.T):
        raise ValueError(f"{name} must be symmetric.")
    if np.any(np.linalg.eigvalsh(matrix) <= 0.0):
        raise ValueError(f"{name} must be positive definite.")
    return matrix


def _vee(skew_matrix: FloatArray) -> FloatArray:
    matrix = np.asarray(skew_matrix, dtype=float)
    return np.array(
        [
            matrix[2, 1],
            matrix[0, 2],
            matrix[1, 0],
        ],
        dtype=float,
    )


@dataclass(frozen=True, slots=True)
class LeaderTrajectorySample:
    """Translational reference supplied to the leader controller."""

    position: FloatArray
    velocity: FloatArray
    acceleration: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "position",
            _vector3(self.position, name="position").copy(),
        )
        object.__setattr__(
            self,
            "velocity",
            _vector3(self.velocity, name="velocity").copy(),
        )
        object.__setattr__(
            self,
            "acceleration",
            _vector3(self.acceleration, name="acceleration").copy(),
        )


@dataclass(frozen=True)
class BlueROV2LeaderEvaluation:
    """One evaluation of the leader tracking controller."""

    configuration_value: float
    generalized_configuration_gradient: FloatArray
    feedforward_velocity_body: FloatArray
    feedforward_velocity_derivative_body: FloatArray
    controller: SecondOrderControllerEvaluation
    input_matrix: FloatArray

    @property
    def optimized_input(self) -> FloatArray:
        return self.controller.control

    @property
    def wrench_body(self) -> FloatArray:
        return self.input_matrix @ self.optimized_input

    @property
    def slack(self) -> float:
        return self.controller.slack

    @property
    def required_slack(self) -> float | None:
        return self.controller.required_slack

    @property
    def actuation_margin(self) -> float | None:
        return self.controller.actuation_margin


@dataclass(frozen=True)
class BlueROV2LeaderController:
    """Trajectory/velocity-reference adapter for the common CLF-QP."""

    dynamics_controller: SecondOrderCLFQPController
    model: BlueROV2Model
    position_gain: FloatArray
    desired_rotation: FloatArray
    attitude_gain: float = 1.0

    def __post_init__(self) -> None:
        if self.dynamics_controller.velocity_dim != 6:
            raise ValueError(
                "BlueROV2 leader control requires a 6-D velocity controller."
            )
        if not np.allclose(
            self.dynamics_controller.clf.inertia,
            self.model.mass_matrix,
        ):
            raise ValueError("leader CLF inertia must match the BlueROV2 mass matrix.")

        object.__setattr__(
            self,
            "position_gain",
            _positive_definite_matrix(
                self.position_gain,
                name="position_gain",
            ),
        )
        object.__setattr__(
            self,
            "desired_rotation",
            _rotation(
                self.desired_rotation,
                name="desired_rotation",
            ),
        )
        if not np.isfinite(self.attitude_gain) or self.attitude_gain <= 0.0:
            raise ValueError("attitude_gain must be finite and positive.")

    @classmethod
    def from_initial_state(
        cls,
        dynamics_controller: SecondOrderCLFQPController,
        model: BlueROV2Model,
        initial_state: FloatArray,
        *,
        position_gain: FloatArray | None = None,
        attitude_gain: float = 1.0,
    ) -> BlueROV2LeaderController:
        """Build a leader controller holding its initial attitude."""
        _, quaternion, _ = model.split_state(initial_state)
        if position_gain is None:
            position_gain = np.diag([1.0, 1.0, 1.0])
        return cls(
            dynamics_controller=dynamics_controller,
            model=model,
            position_gain=position_gain,
            desired_rotation=rotation_matrix_from_quaternion(quaternion),
            attitude_gain=attitude_gain,
        )

    def tracking_terms(
        self,
        state: FloatArray,
        reference: LeaderTrajectorySample,
    ) -> tuple[float, FloatArray, FloatArray, FloatArray]:
        """Return ``(V, zeta, nu_ff, nu_ff_dot)``."""
        position, quaternion, generalized_velocity = self.model.split_state(state)
        rotation = rotation_matrix_from_quaternion(quaternion)
        angular_velocity = generalized_velocity[3:]

        position_error = position - reference.position
        position_gradient_inertial = self.position_gain @ position_error
        position_value = 0.5 * float(
            position_error @ self.position_gain @ position_error
        )

        relative_rotation = self.desired_rotation.T @ rotation
        attitude_error = 0.5 * _vee(relative_rotation - relative_rotation.T)
        attitude_trace_error = float(np.trace(np.eye(3) - relative_rotation))
        attitude_trace_error = float(np.clip(attitude_trace_error, 0.0, 4.0))
        attitude_value = 0.5 * self.attitude_gain * attitude_trace_error
        attitude_gradient_body = self.attitude_gain * attitude_error

        generalized_gradient = np.concatenate(
            (
                rotation.T @ position_gradient_inertial,
                attitude_gradient_body,
            )
        )

        linear_feedforward_body = rotation.T @ reference.velocity
        linear_feedforward_derivative_body = (
            rotation.T @ reference.acceleration
            - np.cross(
                angular_velocity,
                linear_feedforward_body,
            )
        )

        feedforward_velocity_body = np.concatenate(
            (
                linear_feedforward_body,
                np.zeros(3),
            )
        )
        feedforward_velocity_derivative_body = np.concatenate(
            (
                linear_feedforward_derivative_body,
                np.zeros(3),
            )
        )

        return (
            position_value + attitude_value,
            generalized_gradient,
            feedforward_velocity_body,
            feedforward_velocity_derivative_body,
        )

    def initialize_filter(
        self,
        *,
        state: FloatArray,
        reference: LeaderTrajectorySample,
        configuration_gradient_offset: FloatArray | None = None,
    ) -> FloatArray:
        """Initialize the filter on the feedback part of the virtual twist."""
        _, gradient, _, _ = self.tracking_terms(state, reference)
        if configuration_gradient_offset is not None:
            gradient = gradient + _vector6(
                configuration_gradient_offset,
                name="configuration_gradient_offset",
            )
        return self.dynamics_controller.initialize_feedback_filter(gradient)

    def evaluate(
        self,
        *,
        state: FloatArray,
        reference: LeaderTrajectorySample,
        filter_state: FloatArray,
        configuration_value_offset: float = 0.0,
        configuration_gradient_offset: FloatArray | None = None,
        control_reference: FloatArray | None = None,
        control_lower: FloatArray | None = None,
        control_upper: FloatArray | None = None,
    ) -> BlueROV2LeaderEvaluation:
        """Evaluate the common actuator-constrained CLF-QP for the leader."""
        _, _, generalized_velocity = self.model.split_state(state)
        (
            configuration_value,
            gradient,
            feedforward_velocity,
            feedforward_velocity_derivative,
        ) = self.tracking_terms(state, reference)

        if not np.isfinite(configuration_value_offset) or configuration_value_offset < 0.0:
            raise ValueError(
                "configuration_value_offset must be finite and nonnegative."
            )
        configuration_value = float(
            configuration_value + configuration_value_offset
        )
        if configuration_gradient_offset is not None:
            gradient = gradient + _vector6(
                configuration_gradient_offset,
                name="configuration_gradient_offset",
            )

        evaluation = self.dynamics_controller.evaluate_with_feedforward_derivative(
            configuration_value=configuration_value,
            configuration_gradient=gradient,
            generalized_velocity=generalized_velocity,
            filter_state=filter_state,
            dynamics_bias=self.model.drift_wrench(state),
            feedforward_velocity=feedforward_velocity,
            feedforward_velocity_derivative=(feedforward_velocity_derivative),
            control_reference=control_reference,
            control_lower=control_lower,
            control_upper=control_upper,
        )

        input_matrix = self.dynamics_controller.clf.input_matrix
        assert input_matrix is not None
        return BlueROV2LeaderEvaluation(
            configuration_value=configuration_value,
            generalized_configuration_gradient=gradient,
            feedforward_velocity_body=feedforward_velocity,
            feedforward_velocity_derivative_body=(feedforward_velocity_derivative),
            controller=evaluation,
            input_matrix=input_matrix,
        )
