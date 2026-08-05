"""Composite backstepping control-Lyapunov function.

For a constant generalized inertia matrix ``M`` and dynamics

    M nu_dot + h = B u,

the CLF is

    W = V + 1/2 e_nu.T M e_nu,
    e_nu = nu - nu_c,

where ``nu_c`` is the filtered virtual generalized velocity.

If the local configuration-potential derivative is

    V_dot = zeta.T nu + chi,

then

    W_dot = a + b.T u,

with

    a = zeta.T nu + chi - e_nu.T (h + M nu_c_dot),
    b = B.T e_nu.

The optional scalar ``chi`` can contain known parent-motion or explicit
time-dependence terms.  Unknown contributions can instead be omitted from the
implemented CLF constraint and handled as perturbations in the analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from formation_control.models.base import FloatArray


def _vector(value: FloatArray, dimension: int, *, name: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.shape != (dimension,):
        raise ValueError(f"{name} must have shape ({dimension},), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _symmetric_positive_definite(
    value: FloatArray,
    *,
    name: str,
) -> FloatArray:
    matrix = np.asarray(value, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square.")
    if not np.allclose(matrix, matrix.T):
        raise ValueError(f"{name} must be symmetric.")
    if np.any(np.linalg.eigvalsh(matrix) <= 0.0):
        raise ValueError(f"{name} must be positive definite.")
    return matrix


@dataclass(frozen=True)
class BacksteppingCLFEvaluation:
    """Value and control-affine derivative of the composite CLF."""

    value: float
    configuration_value: float
    velocity_error_value: float
    velocity_error: FloatArray
    drift: float
    control_gradient: FloatArray

    def __post_init__(self) -> None:
        velocity_error = np.asarray(self.velocity_error, dtype=float)
        control_gradient = np.asarray(self.control_gradient, dtype=float)

        if velocity_error.ndim != 1:
            raise ValueError("velocity_error must be one-dimensional.")
        if control_gradient.ndim != 1:
            raise ValueError("control_gradient must be one-dimensional.")
        if self.configuration_value < 0.0:
            raise ValueError("configuration_value must be nonnegative.")
        if self.velocity_error_value < -1e-12:
            raise ValueError("velocity_error_value must be nonnegative.")
        if not np.isfinite(self.value) or not np.isfinite(self.drift):
            raise ValueError("CLF value and drift must be finite.")

        object.__setattr__(self, "velocity_error", velocity_error)
        object.__setattr__(self, "control_gradient", control_gradient)

    def derivative(self, control: FloatArray) -> float:
        """Evaluate the modeled CLF derivative for a supplied control."""
        control = np.asarray(control, dtype=float)
        if control.shape != self.control_gradient.shape:
            raise ValueError(
                f"control must have shape {self.control_gradient.shape}, got {control.shape}."
            )
        return float(self.drift + self.control_gradient @ control)


@dataclass(frozen=True)
class BacksteppingCLF:
    """Composite CLF for constant-inertia second-order dynamics.

    ``input_matrix`` maps the optimized control into generalized wrench:

        wrench = B u.

    Leaving it unset gives ``B = I`` and therefore direct wrench control.
    Supplying a thruster allocation matrix later gives the same CLF directly
    in thruster-force coordinates.
    """

    inertia: FloatArray
    input_matrix: FloatArray | None = None

    def __post_init__(self) -> None:
        inertia = _symmetric_positive_definite(
            self.inertia,
            name="inertia",
        )
        dimension = inertia.shape[0]

        if self.input_matrix is None:
            input_matrix = np.eye(dimension)
        else:
            input_matrix = np.asarray(self.input_matrix, dtype=float)
            if input_matrix.ndim != 2 or input_matrix.shape[0] != dimension:
                raise ValueError("input_matrix must have one row per generalized velocity.")
            if not np.all(np.isfinite(input_matrix)):
                raise ValueError("input_matrix must contain only finite values.")

        object.__setattr__(self, "inertia", inertia)
        object.__setattr__(self, "input_matrix", input_matrix)

    @property
    def velocity_dim(self) -> int:
        return self.inertia.shape[0]

    @property
    def input_dim(self) -> int:
        assert self.input_matrix is not None
        return self.input_matrix.shape[1]

    def evaluate(
        self,
        *,
        configuration_value: float,
        generalized_configuration_gradient: FloatArray,
        generalized_velocity: FloatArray,
        filtered_velocity: FloatArray,
        filtered_velocity_derivative: FloatArray,
        dynamics_bias: FloatArray,
        configuration_rate_offset: float = 0.0,
    ) -> BacksteppingCLFEvaluation:
        """Evaluate ``W`` and the affine modeled derivative ``a + b.T u``."""
        if configuration_value < 0.0:
            raise ValueError("configuration_value must be nonnegative.")
        if not np.isfinite(configuration_value):
            raise ValueError("configuration_value must be finite.")
        if not np.isfinite(configuration_rate_offset):
            raise ValueError("configuration_rate_offset must be finite.")

        dimension = self.velocity_dim
        gradient = _vector(
            generalized_configuration_gradient,
            dimension,
            name="generalized_configuration_gradient",
        )
        velocity = _vector(
            generalized_velocity,
            dimension,
            name="generalized_velocity",
        )
        filtered = _vector(
            filtered_velocity,
            dimension,
            name="filtered_velocity",
        )
        filtered_derivative = _vector(
            filtered_velocity_derivative,
            dimension,
            name="filtered_velocity_derivative",
        )
        dynamics_bias = _vector(
            dynamics_bias,
            dimension,
            name="dynamics_bias",
        )

        velocity_error = velocity - filtered
        velocity_error_value = 0.5 * float(velocity_error @ self.inertia @ velocity_error)
        value = float(configuration_value + velocity_error_value)

        configuration_rate = float(gradient @ velocity + configuration_rate_offset)
        drift = float(
            configuration_rate
            - velocity_error @ (dynamics_bias + self.inertia @ filtered_derivative)
        )

        assert self.input_matrix is not None
        control_gradient = self.input_matrix.T @ velocity_error

        return BacksteppingCLFEvaluation(
            value=value,
            configuration_value=float(configuration_value),
            velocity_error_value=velocity_error_value,
            velocity_error=velocity_error,
            drift=drift,
            control_gradient=control_gradient,
        )
