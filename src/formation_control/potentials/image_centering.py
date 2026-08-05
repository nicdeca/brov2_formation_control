"""Smooth image-centering objectives independent of FoV barriers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from formation_control.geometry import NormalizedImagePoint

from .base import PotentialEvaluation


@dataclass(frozen=True)
class ImageCenteringPotential:
    """Quadratic objective for desired normalized image coordinates.

    This is intentionally separate from the FoV barrier.  It provides a smooth
    nominal tendency to center (or otherwise place) the observed target in the
    image, while the barrier is reserved for boundary repulsion.
    """

    alpha_h_desired: float = 0.0
    alpha_v_desired: float = 0.0
    horizontal_gain: float = 1.0
    vertical_gain: float = 1.0

    def __post_init__(self) -> None:
        if self.horizontal_gain <= 0.0:
            raise ValueError("horizontal_gain must be positive.")
        if self.vertical_gain <= 0.0:
            raise ValueError("vertical_gain must be positive.")

    def evaluate(self, point: NormalizedImagePoint) -> PotentialEvaluation:
        error_h = point.alpha_h - self.alpha_h_desired
        error_v = point.alpha_v - self.alpha_v_desired

        return PotentialEvaluation(
            value=0.5 * (self.horizontal_gain * error_h**2 + self.vertical_gain * error_v**2),
            gradient=np.array(
                [
                    self.horizontal_gain * error_h,
                    self.vertical_gain * error_v,
                ],
                dtype=float,
            ),
        )
