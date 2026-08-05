"""Class-K functions used in CLF decay conditions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class ClassKFunction(Protocol):
    """Callable interface for a class-K function on the nonnegative reals."""

    def __call__(self, value: float) -> float:
        """Evaluate the function at a nonnegative scalar."""


def _validate_argument(value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("class-K argument must be finite.")
    if value < 0.0:
        raise ValueError("class-K argument must be nonnegative.")
    return value


@dataclass(frozen=True)
class LinearClassK:
    """Linear class-K function ``alpha(s) = gain * s``."""

    gain: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.gain) or self.gain <= 0.0:
            raise ValueError("gain must be finite and strictly positive.")

    def __call__(self, value: float) -> float:
        value = _validate_argument(value)
        return self.gain * value


@dataclass(frozen=True)
class SaturatingClassK:
    """Bounded class-K function based on the hyperbolic tangent.

    The function is

        alpha(s) = maximum * tanh(gain * s / maximum).

    Its slope at the origin is ``gain`` and its supremum is ``maximum``.
    """

    gain: float
    maximum: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.gain) or self.gain <= 0.0:
            raise ValueError("gain must be finite and strictly positive.")
        if not np.isfinite(self.maximum) or self.maximum <= 0.0:
            raise ValueError("maximum must be finite and strictly positive.")

    def __call__(self, value: float) -> float:
        value = _validate_argument(value)
        return float(self.maximum * np.tanh(self.gain * value / self.maximum))
