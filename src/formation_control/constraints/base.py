"""Generic scalar inequality constraints.

All constraints use the convention

    h(z) > 0

for strict admissibility.  Concrete constraints expose values and, when useful
for control design, analytic gradients with respect to their natural variables.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

StateT = TypeVar("StateT")


class ScalarConstraint(Generic[StateT], ABC):
    """Interface for a scalar strict inequality ``h(z) > 0``."""

    @abstractmethod
    def value(self, state: StateT) -> float:
        """Return the scalar constraint value ``h(state)``."""

    def is_satisfied(self, state: StateT, *, margin: float = 0.0) -> bool:
        """Return whether ``h(state)`` is strictly larger than ``margin``."""
        if margin < 0.0:
            raise ValueError("margin must be nonnegative.")
        return self.value(state) > margin


@dataclass(frozen=True)
class ConstraintEvaluation:
    """Value and gradient of a scalar constraint."""

    value: float
    gradient: FloatArray

    def __post_init__(self) -> None:
        gradient = np.asarray(self.gradient, dtype=float)
        if gradient.ndim != 1:
            raise ValueError("gradient must be a one-dimensional array.")
        object.__setattr__(self, "gradient", gradient)
