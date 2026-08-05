"""Normalized camera field-of-view constraints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from formation_control.geometry import NormalizedImagePoint

from .base import ConstraintEvaluation, FloatArray, ScalarConstraint


@dataclass(frozen=True)
class HorizontalFieldOfViewConstraint(ScalarConstraint[NormalizedImagePoint]):
    """Horizontal FoV constraint ``alpha_limit² - alpha_h² > 0``."""

    alpha_limit: float

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha_limit <= 1.0:
            raise ValueError("alpha_limit must lie in (0, 1].")

    def value(self, point: NormalizedImagePoint) -> float:
        return float(self.alpha_limit**2 - point.alpha_h**2)

    def evaluate(self, point: NormalizedImagePoint) -> ConstraintEvaluation:
        return ConstraintEvaluation(
            value=self.value(point),
            gradient=np.array([-2.0 * point.alpha_h, 0.0]),
        )


@dataclass(frozen=True)
class VerticalFieldOfViewConstraint(ScalarConstraint[NormalizedImagePoint]):
    """Vertical FoV constraint ``alpha_limit² - alpha_v² > 0``."""

    alpha_limit: float

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha_limit <= 1.0:
            raise ValueError("alpha_limit must lie in (0, 1].")

    def value(self, point: NormalizedImagePoint) -> float:
        return float(self.alpha_limit**2 - point.alpha_v**2)

    def evaluate(self, point: NormalizedImagePoint) -> ConstraintEvaluation:
        return ConstraintEvaluation(
            value=self.value(point),
            gradient=np.array([0.0, -2.0 * point.alpha_v]),
        )


@dataclass(frozen=True)
class PositiveDepthConstraint(ScalarConstraint[FloatArray]):
    """Pinhole-camera positive-depth constraint ``p_x^C > 0``."""

    def value(self, point_camera: FloatArray) -> float:
        point_camera = np.asarray(point_camera, dtype=float)
        if point_camera.shape != (3,):
            raise ValueError(f"point_camera must have shape (3,), got {point_camera.shape}.")
        return float(point_camera[0])

    def evaluate(self, point_camera: FloatArray) -> ConstraintEvaluation:
        value = self.value(point_camera)
        return ConstraintEvaluation(
            value=value,
            gradient=np.array([1.0, 0.0, 0.0]),
        )
