"""Distance constraints for relative robot positions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import ConstraintEvaluation, FloatArray, ScalarConstraint


def _relative_position(relative_position: FloatArray) -> FloatArray:
    value = np.asarray(relative_position, dtype=float)
    if value.shape != (3,):
        raise ValueError(f"relative_position must have shape (3,), got {value.shape}.")
    if not np.all(np.isfinite(value)):
        raise ValueError("relative_position must contain only finite values.")
    return value


def _distance_and_direction(relative_position: FloatArray) -> tuple[float, FloatArray]:
    relative_position = _relative_position(relative_position)
    distance = float(np.linalg.norm(relative_position))
    if distance <= np.finfo(float).eps:
        raise ValueError(
            "the Euclidean-distance constraint gradient is undefined at zero "
            "relative position."
        )
    return distance, relative_position / distance


@dataclass(frozen=True)
class MinimumDistanceConstraint(ScalarConstraint[FloatArray]):
    """Collision-avoidance constraint.

    By default the paper form ``||p_ij|| - d_min > 0`` is used.  Set
    ``squared=True`` to recover the legacy form
    ``||p_ij||^2 - d_min^2 > 0``.
    """

    minimum_distance: float
    squared: bool = False

    def __post_init__(self) -> None:
        if self.minimum_distance < 0.0:
            raise ValueError("minimum_distance must be nonnegative.")

    def value(self, relative_position: FloatArray) -> float:
        relative_position = _relative_position(relative_position)
        if self.squared:
            return float(
                relative_position @ relative_position - self.minimum_distance**2
            )
        return float(np.linalg.norm(relative_position) - self.minimum_distance)

    def evaluate(self, relative_position: FloatArray) -> ConstraintEvaluation:
        relative_position = _relative_position(relative_position)
        if self.squared:
            return ConstraintEvaluation(
                value=self.value(relative_position),
                gradient=2.0 * relative_position,
            )
        distance, direction = _distance_and_direction(relative_position)
        return ConstraintEvaluation(
            value=distance - self.minimum_distance,
            gradient=direction,
        )


@dataclass(frozen=True)
class MaximumDistanceConstraint(ScalarConstraint[FloatArray]):
    """Sensing/range constraint.

    By default the paper form ``d_max - ||p_ij|| > 0`` is used.  Set
    ``squared=True`` to recover the legacy form
    ``d_max^2 - ||p_ij||^2 > 0``.
    """

    maximum_distance: float
    squared: bool = False

    def __post_init__(self) -> None:
        if self.maximum_distance <= 0.0:
            raise ValueError("maximum_distance must be positive.")

    def value(self, relative_position: FloatArray) -> float:
        relative_position = _relative_position(relative_position)
        if self.squared:
            return float(
                self.maximum_distance**2 - relative_position @ relative_position
            )
        return float(self.maximum_distance - np.linalg.norm(relative_position))

    def evaluate(self, relative_position: FloatArray) -> ConstraintEvaluation:
        relative_position = _relative_position(relative_position)
        if self.squared:
            return ConstraintEvaluation(
                value=self.value(relative_position),
                gradient=-2.0 * relative_position,
            )
        distance, direction = _distance_and_direction(relative_position)
        return ConstraintEvaluation(
            value=self.maximum_distance - distance,
            gradient=-direction,
        )
