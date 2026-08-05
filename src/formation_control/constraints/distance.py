"""Distance constraints for relative robot positions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import ConstraintEvaluation, FloatArray, ScalarConstraint


def _relative_position(relative_position: FloatArray) -> FloatArray:
    value = np.asarray(relative_position, dtype=float)
    if value.shape != (3,):
        raise ValueError(f"relative_position must have shape (3,), got {value.shape}.")
    return value


@dataclass(frozen=True)
class MinimumDistanceConstraint(ScalarConstraint[FloatArray]):
    """Collision-avoidance constraint ``||p_ij||² - d_min² > 0``."""

    minimum_distance: float

    def __post_init__(self) -> None:
        if self.minimum_distance < 0.0:
            raise ValueError("minimum_distance must be nonnegative.")

    def value(self, relative_position: FloatArray) -> float:
        relative_position = _relative_position(relative_position)
        return float(relative_position @ relative_position - self.minimum_distance**2)

    def evaluate(self, relative_position: FloatArray) -> ConstraintEvaluation:
        relative_position = _relative_position(relative_position)
        return ConstraintEvaluation(
            value=self.value(relative_position),
            gradient=2.0 * relative_position,
        )


@dataclass(frozen=True)
class MaximumDistanceConstraint(ScalarConstraint[FloatArray]):
    """Sensing/range constraint ``d_max² - ||p_ij||² > 0``."""

    maximum_distance: float

    def __post_init__(self) -> None:
        if self.maximum_distance <= 0.0:
            raise ValueError("maximum_distance must be positive.")

    def value(self, relative_position: FloatArray) -> float:
        relative_position = _relative_position(relative_position)
        return float(self.maximum_distance**2 - relative_position @ relative_position)

    def evaluate(self, relative_position: FloatArray) -> ConstraintEvaluation:
        relative_position = _relative_position(relative_position)
        return ConstraintEvaluation(
            value=self.value(relative_position),
            gradient=-2.0 * relative_position,
        )
