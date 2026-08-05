"""Formation-regulation potentials."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import FloatArray, PotentialEvaluation


def _vector3(value: FloatArray, *, name: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}.")
    return array


@dataclass(frozen=True)
class RelativePositionPotential:
    """Quadratic potential for a desired relative position.

    The relative-position convention is

        p_ij = p_j - p_i.

    For error ``e_ij = p_ij - p_ij^d``,

        V = 1/2 e_ij^T K e_ij.
    """

    desired_relative_position: FloatArray
    gain: FloatArray

    def __post_init__(self) -> None:
        desired = _vector3(
            self.desired_relative_position,
            name="desired_relative_position",
        )
        gain = np.asarray(self.gain, dtype=float)

        if gain.shape != (3, 3):
            raise ValueError(f"gain must have shape (3, 3), got {gain.shape}.")
        if not np.allclose(gain, gain.T):
            raise ValueError("gain must be symmetric.")
        if np.any(np.linalg.eigvalsh(gain) <= 0.0):
            raise ValueError("gain must be positive definite.")

        object.__setattr__(self, "desired_relative_position", desired)
        object.__setattr__(self, "gain", gain)

    @classmethod
    def isotropic(
        cls,
        desired_relative_position: FloatArray,
        gain: float,
    ) -> RelativePositionPotential:
        """Create a potential with gain matrix ``gain * I``."""
        if gain <= 0.0:
            raise ValueError("gain must be positive.")
        return cls(
            desired_relative_position=desired_relative_position,
            gain=gain * np.eye(3),
        )

    def evaluate(self, relative_position: FloatArray) -> PotentialEvaluation:
        relative_position = _vector3(
            relative_position,
            name="relative_position",
        )
        error = relative_position - self.desired_relative_position

        return PotentialEvaluation(
            value=0.5 * float(error @ self.gain @ error),
            gradient=self.gain @ error,
        )
