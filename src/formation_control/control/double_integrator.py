"""Agent-level CLF-QP assembly for the double-integrator model.

The double-integrator adapter connects an edge potential to the generic
second-order controller.  It contains no numerical integration and no mutable
controller state.

An edge ``(i, j)`` means that follower ``i`` observes parent ``j`` and uses

    p_ij = p_j - p_i.

The caller may supply a different ``EdgePotential`` at every evaluation.  This
is intentional: adaptive barriers can be bound to the current enlargement
state externally and passed in without making the dynamic controller aware of
the adaptation mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from formation_control.models import DoubleIntegratorModel
from formation_control.models.base import FloatArray
from formation_control.potentials import EdgePotential, EdgePotentialEvaluation

from .second_order import (
    SecondOrderCLFQPController,
    SecondOrderControllerEvaluation,
)


def _vector3(value: FloatArray, *, name: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


@dataclass(frozen=True)
class DoubleIntegratorAgentEvaluation:
    """Potential and dynamic-control results for one follower."""

    potential: EdgePotentialEvaluation
    controller: SecondOrderControllerEvaluation

    @property
    def acceleration(self) -> FloatArray:
        """Return the optimized double-integrator acceleration."""
        return self.controller.control

    @property
    def slack(self) -> float:
        """Return the optimal CLF-QP relaxation."""
        return self.controller.slack


@dataclass(frozen=True)
class DoubleIntegratorAgentController:
    """Thin adapter from edge potentials to a 3-D second-order controller."""

    dynamics_controller: SecondOrderCLFQPController
    model: DoubleIntegratorModel = DoubleIntegratorModel(dimension=3)

    def __post_init__(self) -> None:
        if self.model.dimension != 3:
            raise ValueError("DoubleIntegratorAgentController currently represents 3-D agents.")
        if self.dynamics_controller.velocity_dim != 3:
            raise ValueError("dynamics_controller must have three velocity coordinates.")
        if self.dynamics_controller.control_dim != 3:
            raise ValueError("double-integrator acceleration must have dimension three.")

    @staticmethod
    def _evaluate_potential(
        edge_potential: EdgePotential,
        follower_position: FloatArray,
        parent_position: FloatArray,
    ) -> EdgePotentialEvaluation:
        if edge_potential.uses_camera:
            raise ValueError(
                "double-integrator agents have no orientation; "
                "camera-dependent edge potentials are not supported."
            )

        return edge_potential.evaluate(
            observer_position=follower_position,
            observer_rotation=np.eye(3),
            target_position=parent_position,
        )

    def initialize_filter(
        self,
        *,
        follower_state: FloatArray,
        parent_position: FloatArray,
        edge_potential: EdgePotential,
        feedforward_velocity: FloatArray | None = None,
    ) -> FloatArray:
        """Initialize the filter at the initial virtual follower velocity."""
        follower_position, _ = self.model.split_state(follower_state)
        parent_position = _vector3(parent_position, name="parent_position")

        potential = self._evaluate_potential(
            edge_potential,
            follower_position,
            parent_position,
        )

        return self.dynamics_controller.initialize_filter(
            potential.observer_position_gradient,
            feedforward_velocity=feedforward_velocity,
        )

    def evaluate(
        self,
        *,
        follower_state: FloatArray,
        parent_position: FloatArray,
        edge_potential: EdgePotential,
        filter_state: FloatArray,
        parent_velocity: FloatArray | None = None,
        feedforward_velocity: FloatArray | None = None,
    ) -> DoubleIntegratorAgentEvaluation:
        """Evaluate one follower control cycle.

        If ``parent_velocity`` is supplied, its exact contribution

            chi = grad_{p_j}(V).T v_j

        is included in the modeled CLF derivative.  If it is omitted, the
        implemented QP sets ``chi = 0``; the actual parent-motion contribution
        can then be treated as an interconnection perturbation.
        """
        follower_position, follower_velocity = self.model.split_state(follower_state)
        parent_position = _vector3(parent_position, name="parent_position")

        potential = self._evaluate_potential(
            edge_potential,
            follower_position,
            parent_position,
        )

        if parent_velocity is None:
            configuration_rate_offset = 0.0
        else:
            parent_velocity = _vector3(
                parent_velocity,
                name="parent_velocity",
            )
            configuration_rate_offset = float(potential.target_position_gradient @ parent_velocity)

        controller = self.dynamics_controller.evaluate(
            configuration_value=potential.value,
            configuration_gradient=potential.observer_position_gradient,
            generalized_velocity=follower_velocity,
            filter_state=filter_state,
            dynamics_bias=np.zeros(3),
            configuration_rate_offset=configuration_rate_offset,
            feedforward_velocity=feedforward_velocity,
        )

        return DoubleIntegratorAgentEvaluation(
            potential=potential,
            controller=controller,
        )
