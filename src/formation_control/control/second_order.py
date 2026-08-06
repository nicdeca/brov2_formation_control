"""Reusable second-order command-filtered CLF-QP controller.

This module contains the model-independent dynamic-control layer.  It assumes

    M nu_dot + h = B u,

with constant positive-definite ``M``.  Configuration geometry and potentials
are evaluated outside this class and supplied through the scalar potential
value and its generalized gradient.

The controller performs, in order,

1. virtual generalized-velocity generation,
2. command-filter evaluation,
3. composite backstepping-CLF evaluation,
4. CLF-QP solution.

All dynamic controller states remain external.  In particular, evaluating the
controller never advances the command filter, which prevents accidental
multiple filter updates within one control cycle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from formation_control.models.base import FloatArray

from .backstepping_clf import BacksteppingCLF, BacksteppingCLFEvaluation
from .clf_qp import CLFQP, CLFQPResult
from .filters import (
    CommandFilter,
    CommandFilterEvaluation,
)


def _vector(value: FloatArray, dimension: int, *, name: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.shape != (dimension,):
        raise ValueError(f"{name} must have shape ({dimension},), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _positive_definite_matrix(
    value: FloatArray,
    dimension: int,
    *,
    name: str,
) -> FloatArray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (dimension, dimension):
        raise ValueError(f"{name} must have shape ({dimension}, {dimension}), got {matrix.shape}.")
    if not np.allclose(matrix, matrix.T):
        raise ValueError(f"{name} must be symmetric.")
    if np.any(np.linalg.eigvalsh(matrix) <= 0.0):
        raise ValueError(f"{name} must be positive definite.")
    return matrix


@dataclass(frozen=True)
class SecondOrderControllerEvaluation:
    """Complete result of one controller evaluation."""

    desired_velocity: FloatArray
    filter: CommandFilterEvaluation
    clf: BacksteppingCLFEvaluation
    qp: CLFQPResult

    def __post_init__(self) -> None:
        desired_velocity = np.asarray(self.desired_velocity, dtype=float)
        if desired_velocity.ndim != 1:
            raise ValueError("desired_velocity must be one-dimensional.")
        object.__setattr__(self, "desired_velocity", desired_velocity)

    @property
    def control(self) -> FloatArray:
        """Return the optimized input."""
        return self.qp.control

    @property
    def slack(self) -> float:
        """Return the optimal CLF-QP relaxation."""
        return self.qp.slack

    @property
    def required_slack(self) -> float | None:
        """Return the minimum relaxation required by actuator limits."""
        return self.qp.required_slack

    @property
    def actuation_margin(self) -> float | None:
        """Return the zero-slack CLF feasibility margin."""
        return self.qp.actuation_margin


@dataclass(frozen=True)
class SecondOrderCLFQPController:
    """Command-filtered backstepping controller with a soft CLF-QP.

    Parameters
    ----------
    virtual_gain:
        Positive-definite gain ``K_eta`` in

            nu_d = nu_ff - K_eta zeta.

    command_filter:
        First- or second-order filter for ``nu_d``.
    clf:
        Composite backstepping CLF.  Its input matrix determines whether the
        optimized variable is a generalized wrench, an acceleration, or e.g.
        a thruster-force vector.
    qp:
        Soft CLF-QP operating on the control coordinates exposed by ``clf``.
    """

    virtual_gain: FloatArray
    command_filter: CommandFilter
    clf: BacksteppingCLF
    qp: CLFQP

    def __post_init__(self) -> None:
        dimension = self.clf.velocity_dim
        virtual_gain = _positive_definite_matrix(
            self.virtual_gain,
            dimension,
            name="virtual_gain",
        )

        if self.command_filter.signal_dim != dimension:
            raise ValueError("command-filter signal dimension must match CLF velocity dimension.")
        if self.qp.control_dim != self.clf.input_dim:
            raise ValueError("QP control dimension must match the CLF input dimension.")

        object.__setattr__(self, "virtual_gain", virtual_gain)

    @property
    def velocity_dim(self) -> int:
        """Dimension of the generalized velocity."""
        return self.clf.velocity_dim

    @property
    def control_dim(self) -> int:
        """Dimension of the optimized input."""
        return self.qp.control_dim

    def desired_velocity(
        self,
        configuration_gradient: FloatArray,
        *,
        feedforward_velocity: FloatArray | None = None,
    ) -> FloatArray:
        """Return ``nu_d = nu_ff - K_eta zeta``."""
        gradient = _vector(
            configuration_gradient,
            self.velocity_dim,
            name="configuration_gradient",
        )

        if feedforward_velocity is None:
            feedforward = np.zeros(self.velocity_dim)
        else:
            feedforward = _vector(
                feedforward_velocity,
                self.velocity_dim,
                name="feedforward_velocity",
            )

        return feedforward - self.virtual_gain @ gradient

    def initialize_filter(
        self,
        configuration_gradient: FloatArray,
        *,
        feedforward_velocity: FloatArray | None = None,
    ) -> FloatArray:
        """Initialize the command filter at the current virtual command."""
        desired_velocity = self.desired_velocity(
            configuration_gradient,
            feedforward_velocity=feedforward_velocity,
        )
        return self.command_filter.initialize(desired_velocity)

    def evaluate(
        self,
        *,
        configuration_value: float,
        configuration_gradient: FloatArray,
        generalized_velocity: FloatArray,
        filter_state: FloatArray,
        dynamics_bias: FloatArray,
        configuration_rate_offset: float = 0.0,
        feedforward_velocity: FloatArray | None = None,
        control_gradient_offset: FloatArray | None = None,
        control_reference: FloatArray | None = None,
        control_lower: FloatArray | None = None,
        control_upper: FloatArray | None = None,
    ) -> SecondOrderControllerEvaluation:
        """Evaluate one control cycle without advancing any dynamic state."""
        gradient = _vector(
            configuration_gradient,
            self.velocity_dim,
            name="configuration_gradient",
        )
        velocity = _vector(
            generalized_velocity,
            self.velocity_dim,
            name="generalized_velocity",
        )
        dynamics_bias = _vector(
            dynamics_bias,
            self.velocity_dim,
            name="dynamics_bias",
        )

        desired_velocity = self.desired_velocity(
            gradient,
            feedforward_velocity=feedforward_velocity,
        )
        filter_evaluation = self.command_filter.evaluate(
            filter_state,
            desired_velocity,
        )

        clf_evaluation = self.clf.evaluate(
            configuration_value=configuration_value,
            generalized_configuration_gradient=gradient,
            generalized_velocity=velocity,
            filtered_velocity=filter_evaluation.output,
            filtered_velocity_derivative=filter_evaluation.output_derivative,
            dynamics_bias=dynamics_bias,
            configuration_rate_offset=configuration_rate_offset,
            control_gradient_offset=control_gradient_offset,
        )

        qp_result = self.qp.solve(
            clf_evaluation,
            control_reference=control_reference,
            lower_override=control_lower,
            upper_override=control_upper,
        )

        return SecondOrderControllerEvaluation(
            desired_velocity=desired_velocity,
            filter=filter_evaluation,
            clf=clf_evaluation,
            qp=qp_result,
        )
