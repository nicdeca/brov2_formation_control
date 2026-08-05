"""Virtual generalized-velocity commands for configuration regulation."""

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


def _positive_definite_matrix(
    value: FloatArray,
    dimension: int,
    *,
    name: str,
) -> FloatArray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (dimension, dimension):
        raise ValueError(f"{name} must have shape ({dimension}, {dimension}), got {matrix.shape}.")
    if not np.allclose(matrix, matrix.T):
        raise ValueError(f"{name} must be symmetric.")
    if np.any(np.linalg.eigvalsh(matrix) <= 0.0):
        raise ValueError(f"{name} must be positive definite.")
    return matrix


@dataclass(frozen=True)
class VirtualVelocityGains:
    """Block-diagonal translational and rotational descent gains."""

    translation: FloatArray
    rotation: FloatArray

    def __post_init__(self) -> None:
        translation = _positive_definite_matrix(
            self.translation,
            3,
            name="translation",
        )
        rotation = _positive_definite_matrix(
            self.rotation,
            3,
            name="rotation",
        )

        object.__setattr__(self, "translation", translation)
        object.__setattr__(self, "rotation", rotation)

    @classmethod
    def isotropic(
        cls,
        *,
        translation: float,
        rotation: float,
    ) -> VirtualVelocityGains:
        """Construct isotropic translational and rotational gains."""
        if translation <= 0.0:
            raise ValueError("translation must be positive.")
        if rotation <= 0.0:
            raise ValueError("rotation must be positive.")

        return cls(
            translation=translation * np.eye(3),
            rotation=rotation * np.eye(3),
        )

    @property
    def matrix(self) -> FloatArray:
        """Return the complete block-diagonal gain matrix."""
        gain = np.zeros((6, 6))
        gain[:3, :3] = self.translation
        gain[3:, 3:] = self.rotation
        return gain


@dataclass(frozen=True)
class VirtualVelocityEvaluation:
    """Generalized configuration gradient and desired body velocity."""

    generalized_gradient_body: FloatArray
    desired_velocity_body: FloatArray

    def __post_init__(self) -> None:
        gradient = _vector(
            self.generalized_gradient_body,
            6,
            name="generalized_gradient_body",
        )
        command = _vector(
            self.desired_velocity_body,
            6,
            name="desired_velocity_body",
        )
        object.__setattr__(self, "generalized_gradient_body", gradient)
        object.__setattr__(self, "desired_velocity_body", command)


def generalized_configuration_gradient(
    rotation_body_to_inertial: FloatArray,
    position_gradient_inertial: FloatArray,
    orientation_gradient_body: FloatArray,
) -> FloatArray:
    """Return the body-coordinate generalized gradient.

    If

        V_dot = grad_p(V).T R v + g_R.T omega + offset,

    then

        zeta = [R.T grad_p(V), g_R]

    satisfies ``V_dot = zeta.T nu + offset``.
    """
    rotation = np.asarray(rotation_body_to_inertial, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError("rotation_body_to_inertial must have shape (3, 3).")

    position_gradient = _vector(
        position_gradient_inertial,
        3,
        name="position_gradient_inertial",
    )
    orientation_gradient = _vector(
        orientation_gradient_body,
        3,
        name="orientation_gradient_body",
    )

    return np.concatenate(
        (
            rotation.T @ position_gradient,
            orientation_gradient,
        )
    )


def virtual_velocity_command(
    rotation_body_to_inertial: FloatArray,
    position_gradient_inertial: FloatArray,
    orientation_gradient_body: FloatArray,
    gains: VirtualVelocityGains,
    *,
    feedforward_velocity_body: FloatArray | None = None,
) -> VirtualVelocityEvaluation:
    """Generate a block-diagonal gradient-descent generalized velocity.

    The command is

        nu_d = nu_ff - K_eta zeta,

    where ``zeta = [R.T grad_p(V), g_R]``.

    The supplied gradients may come from the complete configuration potential
    or from a chosen nominal objective.  This keeps nominal command generation
    independent from the potential used later in the CLF.
    """
    generalized_gradient = generalized_configuration_gradient(
        rotation_body_to_inertial,
        position_gradient_inertial,
        orientation_gradient_body,
    )

    if feedforward_velocity_body is None:
        feedforward = np.zeros(6)
    else:
        feedforward = _vector(
            feedforward_velocity_body,
            6,
            name="feedforward_velocity_body",
        )

    desired_velocity = feedforward - gains.matrix @ generalized_gradient

    return VirtualVelocityEvaluation(
        generalized_gradient_body=generalized_gradient,
        desired_velocity_body=desired_velocity,
    )
