"""Recentered barriers for additively enlarged scalar constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from formation_control.constraints import ConstraintEvaluation

from .barriers import RecenteredLogBarrier
from .base import FloatArray, PotentialEvaluation


class DifferentiableConstraint[StateT](Protocol):
    """Constraint exposing scalar value and Euclidean gradient."""

    def evaluate(self, state: StateT) -> ConstraintEvaluation:
        """Return ``h_c(state)`` and its state gradient."""


@dataclass(frozen=True)
class AdaptiveBarrierEvaluation:
    """Potential value, state gradient, and derivative w.r.t. enlargement."""

    value: float
    gradient: FloatArray
    enlargement_derivative: float
    constraint_value: float
    reference_constraint_value: float

    def __post_init__(self) -> None:
        gradient = np.asarray(self.gradient, dtype=float)
        if gradient.ndim != 1:
            raise ValueError("gradient must be one-dimensional.")
        if not np.all(np.isfinite(gradient)):
            raise ValueError("gradient must contain only finite values.")
        for name in (
            "value",
            "enlargement_derivative",
            "constraint_value",
            "reference_constraint_value",
        ):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite.")
        object.__setattr__(self, "gradient", gradient)


@dataclass(frozen=True)
class AdaptiveConstraintBarrierPotential[StateT]:
    """Barrier for ``h_a(state, rho) = h_c(state) + rho``.

    The reference constraint value is enlarged by the same ``rho``:

        h_a^d(rho) = h_c(reference_state) + rho.

    Consequently the recentered barrier remains zero with zero state gradient
    at the desired state for every admissible enlargement.
    """

    constraint: DifferentiableConstraint[StateT]
    reference_state: StateT
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.weight) or self.weight < 0.0:
            raise ValueError("weight must be finite and nonnegative.")

        reference = self.constraint.evaluate(self.reference_state)
        if reference.value <= 0.0:
            raise ValueError("reference_state must lie strictly inside the conservative domain.")
        object.__setattr__(self, "_reference_base_value", float(reference.value))

    @property
    def reference_base_value(self) -> float:
        """Return the conservative constraint value at the desired state."""
        return self._reference_base_value

    def evaluate(
        self,
        state: StateT,
        enlargement: float,
    ) -> AdaptiveBarrierEvaluation:
        """Evaluate the adaptive barrier for a frozen enlargement ``rho``."""
        enlargement = float(enlargement)
        if not np.isfinite(enlargement) or enlargement < 0.0:
            raise ValueError("enlargement must be finite and nonnegative.")

        base = self.constraint.evaluate(state)
        constraint_value = float(base.value + enlargement)
        reference_value = float(self.reference_base_value + enlargement)

        barrier = RecenteredLogBarrier(reference_value)
        value = self.weight * barrier.value(constraint_value)
        multiplier = self.weight * barrier.derivative(constraint_value)
        gradient = multiplier * base.gradient

        # Both h and h_d shift by rho.  For
        # beta_bar = -log(h/h_d) + h/h_d - 1,
        # d beta_bar / d rho = -(h - h_d)^2 / (h_d^2 h).
        enlargement_derivative = -self.weight * (
            (constraint_value - reference_value) ** 2 / (reference_value**2 * constraint_value)
        )

        return AdaptiveBarrierEvaluation(
            value=value,
            gradient=gradient,
            enlargement_derivative=float(enlargement_derivative),
            constraint_value=constraint_value,
            reference_constraint_value=reference_value,
        )

    def bind(
        self,
        enlargement: float,
    ) -> BoundAdaptiveConstraintBarrierPotential[StateT]:
        """Freeze ``rho`` and return a standard Euclidean potential."""
        return BoundAdaptiveConstraintBarrierPotential(
            potential=self,
            enlargement=enlargement,
        )


@dataclass(frozen=True)
class BoundAdaptiveConstraintBarrierPotential[StateT]:
    """Adaptive barrier with a frozen enlargement parameter."""

    potential: AdaptiveConstraintBarrierPotential[StateT]
    enlargement: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.enlargement) or self.enlargement < 0.0:
            raise ValueError("enlargement must be finite and nonnegative.")

    def evaluate(self, state: StateT) -> PotentialEvaluation:
        """Return value and state gradient with ``rho`` held fixed."""
        evaluation = self.potential.evaluate(state, self.enlargement)
        return PotentialEvaluation(
            value=evaluation.value,
            gradient=evaluation.gradient,
        )

    def enlargement_derivative(self, state: StateT) -> float:
        """Return ``partial V / partial rho`` at the frozen enlargement."""
        return self.potential.evaluate(
            state,
            self.enlargement,
        ).enlargement_derivative
