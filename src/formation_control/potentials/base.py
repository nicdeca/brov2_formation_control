"""Generic Euclidean potential primitives."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

StateT = TypeVar("StateT")


@dataclass(frozen=True)
class PotentialEvaluation:
    """Value and Euclidean gradient of a differentiable potential."""

    value: float
    gradient: FloatArray

    def __post_init__(self) -> None:
        gradient = np.asarray(self.gradient, dtype=float)
        if gradient.ndim != 1:
            raise ValueError("gradient must be a one-dimensional array.")
        if not np.isfinite(self.value):
            raise ValueError("potential value must be finite.")
        if not np.all(np.isfinite(gradient)):
            raise ValueError("potential gradient must be finite.")
        object.__setattr__(self, "gradient", gradient)


class EuclideanPotential(Protocol[StateT]):
    """Protocol for differentiable potentials over Euclidean variables."""

    def evaluate(self, state: StateT) -> PotentialEvaluation:
        """Return the potential value and gradient at ``state``."""


@dataclass(frozen=True)
class WeightedPotential(Generic[StateT]):
    """Positive scalar weighting of another potential."""

    potential: EuclideanPotential[StateT]
    weight: float

    def __post_init__(self) -> None:
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative.")

    def evaluate(self, state: StateT) -> PotentialEvaluation:
        evaluation = self.potential.evaluate(state)
        return PotentialEvaluation(
            value=self.weight * evaluation.value,
            gradient=self.weight * evaluation.gradient,
        )


@dataclass(frozen=True)
class SumPotential(Generic[StateT]):
    """Sum of potentials defined over the same Euclidean variable."""

    terms: Sequence[EuclideanPotential[StateT]]

    def __post_init__(self) -> None:
        if not self.terms:
            raise ValueError("SumPotential requires at least one term.")

    def evaluate(self, state: StateT) -> PotentialEvaluation:
        evaluations = [term.evaluate(state) for term in self.terms]
        shape = evaluations[0].gradient.shape

        if any(evaluation.gradient.shape != shape for evaluation in evaluations):
            raise ValueError("all potential gradients must have the same shape.")

        return PotentialEvaluation(
            value=sum(evaluation.value for evaluation in evaluations),
            gradient=np.sum(
                [evaluation.gradient for evaluation in evaluations],
                axis=0,
            ),
        )
