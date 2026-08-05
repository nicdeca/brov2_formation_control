"""Logarithmic barrier potentials.

The recentered barrier is constructed in the scalar constraint value ``h``,
rather than directly in the underlying state variable.  For ``h > 0`` and
reference ``h_d > 0``,

    beta_bar(h; h_d) = -log(h / h_d) + h / h_d - 1.

This construction is nonnegative, vanishes at ``h = h_d``, has zero derivative
there, and diverges as ``h -> 0+``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from formation_control.constraints import ConstraintEvaluation

from .base import PotentialEvaluation


class DifferentiableConstraint[StateT](Protocol):
    """Constraint exposing both scalar value and Euclidean gradient."""

    def evaluate(self, state: StateT) -> ConstraintEvaluation:
        """Return ``h(state)`` and its gradient."""


@dataclass(frozen=True)
class RecenteredLogBarrier:
    """Scalar recentered logarithmic barrier."""

    reference_value: float

    def __post_init__(self) -> None:
        if self.reference_value <= 0.0:
            raise ValueError("reference_value must be strictly positive.")

    def value(self, constraint_value: float) -> float:
        """Return the recentered barrier value."""
        if constraint_value <= 0.0:
            raise ValueError("constraint_value must be strictly positive.")

        ratio = constraint_value / self.reference_value
        return float(-np.log(ratio) + ratio - 1.0)

    def derivative(self, constraint_value: float) -> float:
        """Return the derivative with respect to the scalar constraint value."""
        if constraint_value <= 0.0:
            raise ValueError("constraint_value must be strictly positive.")

        return 1.0 / self.reference_value - 1.0 / constraint_value


@dataclass(frozen=True)
class ConstraintBarrierPotential[StateT]:
    """Compose a differentiable scalar constraint with a recentered barrier."""

    constraint: DifferentiableConstraint[StateT]
    barrier: RecenteredLogBarrier
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative.")

    @classmethod
    def from_reference(
        cls,
        constraint: DifferentiableConstraint[StateT],
        reference_state: StateT,
        *,
        weight: float = 1.0,
    ) -> ConstraintBarrierPotential[StateT]:
        """Construct the barrier using ``h(reference_state)`` as ``h_d``."""
        evaluation = constraint.evaluate(reference_state)
        return cls(
            constraint=constraint,
            barrier=RecenteredLogBarrier(evaluation.value),
            weight=weight,
        )

    def evaluate(self, state: StateT) -> PotentialEvaluation:
        constraint_evaluation = self.constraint.evaluate(state)
        multiplier = self.weight * self.barrier.derivative(constraint_evaluation.value)

        return PotentialEvaluation(
            value=self.weight * self.barrier.value(constraint_evaluation.value),
            gradient=multiplier * constraint_evaluation.gradient,
        )
