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


def _positive_vector(
    value: FloatArray,
    dimension: int,
    *,
    name: str,
) -> FloatArray:
    array = _vector(value, dimension, name=name)
    if np.any(array <= 0.0):
        raise ValueError(f"{name} must be strictly positive.")
    return array


def _smooth_norm_saturation(
    value: FloatArray,
    limit: float,
) -> FloatArray:
    """Smoothly saturate a vector norm while preserving its direction.

    The map is

        ssat(x, limit)
        = limit * tanh(||x|| / limit) * x / ||x||

    for nonzero ``x``, and zero at the origin.
    """
    vector = np.asarray(value, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return np.zeros_like(vector)
    return limit * np.tanh(norm / limit) * vector / norm


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
    unlimited_desired_velocity: FloatArray
    filter_command: FloatArray
    filter: CommandFilterEvaluation
    clf: BacksteppingCLFEvaluation
    qp: CLFQPResult

    def __post_init__(self) -> None:
        desired_velocity = np.asarray(self.desired_velocity, dtype=float)
        unlimited_desired_velocity = np.asarray(
            self.unlimited_desired_velocity,
            dtype=float,
        )
        if desired_velocity.ndim != 1:
            raise ValueError("desired_velocity must be one-dimensional.")
        filter_command = np.asarray(self.filter_command, dtype=float)
        if unlimited_desired_velocity.shape != desired_velocity.shape:
            raise ValueError("unlimited_desired_velocity must match desired_velocity shape.")
        if filter_command.shape != desired_velocity.shape:
            raise ValueError("filter_command must match desired_velocity shape.")
        object.__setattr__(self, "desired_velocity", desired_velocity)
        object.__setattr__(
            self,
            "unlimited_desired_velocity",
            unlimited_desired_velocity,
        )
        object.__setattr__(self, "filter_command", filter_command)

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
        Positive-definite gain ``K_eta`` in the unsaturated command

            nu_d,raw = nu_ff - K_eta zeta.

    virtual_velocity_norm_limits:
        Optional positive norm limits for smooth saturation of groups of the
        virtual feedback correction.

    virtual_velocity_group_sizes:
        Sizes of the groups associated with ``virtual_velocity_norm_limits``.
        Each group is saturated radially, preserving its direction exactly.
        For a 6-DoF body twist, use ``(3, 3)`` to saturate the translational
        and rotational corrections independently.

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
    virtual_velocity_norm_limits: FloatArray | None = None
    virtual_velocity_group_sizes: tuple[int, ...] | None = None

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

        norm_limits = self.virtual_velocity_norm_limits
        group_sizes = self.virtual_velocity_group_sizes

        if (norm_limits is None) != (group_sizes is None):
            raise ValueError(
                "virtual_velocity_norm_limits and "
                "virtual_velocity_group_sizes must be provided together."
            )

        if norm_limits is not None:
            norm_limits = np.asarray(norm_limits, dtype=float)
            if norm_limits.ndim != 1 or norm_limits.size == 0:
                raise ValueError(
                    "virtual_velocity_norm_limits must be a nonempty one-dimensional array."
                )
            if not np.all(np.isfinite(norm_limits)) or np.any(norm_limits <= 0.0):
                raise ValueError("virtual_velocity_norm_limits must be finite and positive.")

            if not group_sizes or any(size <= 0 for size in group_sizes):
                raise ValueError("virtual_velocity_group_sizes must contain positive sizes.")
            if sum(group_sizes) != dimension:
                raise ValueError("virtual_velocity_group_sizes must sum to velocity_dim.")
            if len(group_sizes) != norm_limits.size:
                raise ValueError("one virtual_velocity_norm_limit is required per group.")

            # Radial saturation of each block preserves the correction
            # direction within that block. To preserve the gradient-descent
            # sign, K_eta must not couple different saturation blocks.
            start = 0
            for size in group_sizes:
                stop = start + size
                if np.any(np.abs(virtual_gain[start:stop, :start]) > 1e-12) or np.any(
                    np.abs(virtual_gain[start:stop, stop:]) > 1e-12
                ):
                    raise ValueError(
                        "virtual_gain must be block diagonal with respect to "
                        "virtual_velocity_group_sizes."
                    )
                start = stop

        object.__setattr__(self, "virtual_gain", virtual_gain)
        object.__setattr__(
            self,
            "virtual_velocity_norm_limits",
            norm_limits,
        )
        object.__setattr__(
            self,
            "virtual_velocity_group_sizes",
            group_sizes,
        )

    @property
    def velocity_dim(self) -> int:
        """Dimension of the generalized velocity."""
        return self.clf.velocity_dim

    @property
    def control_dim(self) -> int:
        """Dimension of the optimized input."""
        return self.qp.control_dim

    def unlimited_desired_velocity(
        self,
        configuration_gradient: FloatArray,
        *,
        feedforward_velocity: FloatArray | None = None,
    ) -> FloatArray:
        """Return the unsaturated command ``nu_ff - K_eta zeta``."""
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

    def desired_velocity(
        self,
        configuration_gradient: FloatArray,
        *,
        feedforward_velocity: FloatArray | None = None,
    ) -> FloatArray:
        """Return the possibly saturated virtual generalized velocity."""
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

        correction = self.virtual_gain @ gradient
        if self.virtual_velocity_norm_limits is not None:
            saturated = np.empty_like(correction)
            start = 0
            for size, limit in zip(
                self.virtual_velocity_group_sizes,
                self.virtual_velocity_norm_limits,
                strict=True,
            ):
                stop = start + size
                saturated[start:stop] = _smooth_norm_saturation(
                    correction[start:stop],
                    float(limit),
                )
                start = stop
            correction = saturated

        return feedforward - correction

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

    def initialize_feedback_filter(
        self,
        configuration_gradient: FloatArray,
    ) -> FloatArray:
        """Initialize a filter used only for the feedback correction.

        This is useful when an exact feedforward velocity and its derivative
        are available. The total filtered velocity is then formed as

            nu_c = nu_ff + nu_fb,c,

        while only ``nu_fb,d`` is command-filtered.
        """
        feedback_command = self.desired_velocity(
            configuration_gradient,
            feedforward_velocity=np.zeros(self.velocity_dim),
        )
        return self.command_filter.initialize(feedback_command)

    def evaluate_with_feedforward_derivative(
        self,
        *,
        configuration_value: float,
        configuration_gradient: FloatArray,
        generalized_velocity: FloatArray,
        filter_state: FloatArray,
        dynamics_bias: FloatArray,
        feedforward_velocity: FloatArray,
        feedforward_velocity_derivative: FloatArray,
        configuration_rate_offset: float = 0.0,
        control_gradient_offset: FloatArray | None = None,
        control_reference: FloatArray | None = None,
        control_lower: FloatArray | None = None,
        control_upper: FloatArray | None = None,
    ) -> SecondOrderControllerEvaluation:
        """Evaluate using exact feedforward velocity and derivative.

        The virtual command is split as

            nu_d = nu_ff + nu_fb,d,

        where ``nu_fb,d`` is the smoothly saturated backstepping correction.
        Only the feedback part is passed through the command filter:

            nu_c     = nu_ff     + nu_fb,c,
            nu_c_dot = nu_ff_dot + nu_fb,c_dot.

        This avoids adding command-filter phase lag to a known trajectory
        feedforward while retaining command filtering of the nonlinear
        backstepping correction.
        """
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
        feedforward = _vector(
            feedforward_velocity,
            self.velocity_dim,
            name="feedforward_velocity",
        )
        feedforward_derivative = _vector(
            feedforward_velocity_derivative,
            self.velocity_dim,
            name="feedforward_velocity_derivative",
        )

        zero_feedforward = np.zeros(self.velocity_dim)
        unlimited_feedback_command = self.unlimited_desired_velocity(
            gradient,
            feedforward_velocity=zero_feedforward,
        )
        feedback_command = self.desired_velocity(
            gradient,
            feedforward_velocity=zero_feedforward,
        )
        feedback_filter = self.command_filter.evaluate(
            filter_state,
            feedback_command,
        )

        desired_velocity = feedforward + feedback_command
        unlimited_desired_velocity = feedforward + unlimited_feedback_command
        total_filter = CommandFilterEvaluation(
            output=feedforward + feedback_filter.output,
            output_derivative=(feedforward_derivative + feedback_filter.output_derivative),
        )

        clf_evaluation = self.clf.evaluate(
            configuration_value=configuration_value,
            generalized_configuration_gradient=gradient,
            generalized_velocity=velocity,
            filtered_velocity=total_filter.output,
            filtered_velocity_derivative=total_filter.output_derivative,
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
            unlimited_desired_velocity=unlimited_desired_velocity,
            filter_command=feedback_command,
            filter=total_filter,
            clf=clf_evaluation,
            qp=qp_result,
        )

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

        unlimited_desired_velocity = self.unlimited_desired_velocity(
            gradient,
            feedforward_velocity=feedforward_velocity,
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
            unlimited_desired_velocity=unlimited_desired_velocity,
            filter_command=desired_velocity,
            filter=filter_evaluation,
            clf=clf_evaluation,
            qp=qp_result,
        )
