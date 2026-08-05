"""Absolute position-tracking potentials."""

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
class PositionTrackingPotential:
    """Quadratic absolute-position tracking potential.

    For ``e = p - p_d``,

        V = 1/2 e.T K e.
    """

    desired_position: FloatArray
    gain: FloatArray

    def __post_init__(self) -> None:
        desired = _vector3(self.desired_position, name="desired_position")
        gain = np.asarray(self.gain, dtype=float)

        if gain.shape != (3, 3):
            raise ValueError(f"gain must have shape (3, 3), got {gain.shape}.")
        if not np.allclose(gain, gain.T):
            raise ValueError("gain must be symmetric.")
        if np.any(np.linalg.eigvalsh(gain) <= 0.0):
            raise ValueError("gain must be positive definite.")

        object.__setattr__(self, "desired_position", desired)
        object.__setattr__(self, "gain", gain)

    @classmethod
    def isotropic(
        cls,
        desired_position: FloatArray,
        gain: float,
    ) -> PositionTrackingPotential:
        """Create a potential with gain matrix ``gain * I``."""
        if gain <= 0.0:
            raise ValueError("gain must be positive.")

        return cls(
            desired_position=desired_position,
            gain=gain * np.eye(3),
        )

    def evaluate(self, position: FloatArray) -> PotentialEvaluation:
        position = _vector3(position, name="position")
        error = position - self.desired_position

        return PotentialEvaluation(
            value=0.5 * float(error @ self.gain @ error),
            gradient=self.gain @ error,
        )
