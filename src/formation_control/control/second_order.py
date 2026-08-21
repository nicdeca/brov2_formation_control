"""Reusable second-order command-filtered CLF-QP controller.

This module implements the command-filtered backstepping dynamic-control layer
for

    M nu_dot + h = B u,

with constant positive-definite ``M``.  The updated paper inequality is

    a_c + b.T u <= -e_nu.T K_e e_nu + delta,

where

    e_nu = nu - nu_c,
    a_c  = e_nu.T (zeta - h - M nu_c_dot),
    b    = B.T e_nu

(up to any explicitly supplied control-gradient offset).

The complete modeled CLF derivative is still retained for diagnostics.  The QP
therefore enforces the paper's backstepping dissipation condition without
canceling the full marine drift.

An optional trim allocator can provide a control satisfying the stationary
restoring wrench.  Its reference is activated smoothly only as the local CLF
input direction approaches zero.
"""

from __future__ import annotations

from collections.abc import Callable
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


def _descending_quintic_activation(
    value: float,
    *,
    on: float,
    off: float,
) -> float:
    """C2 activation equal to one near zero and zero above ``off``."""
    if value <= on:
        return 1.0
    if value >= off:
        return 0.0
    xi = (off - value) / (off - on)
    return float(6.0 * xi**5 - 15.0 * xi**4 + 10.0 * xi**3)


@dataclass(frozen=True)
class SecondOrderControllerEvaluation:
    """Complete result of one controller evaluation."""

    desired_velocity: FloatArray
    unlimited_desired_velocity: FloatArray
    filter_command: FloatArray
    filter: CommandFilterEvaluation
    clf: BacksteppingCLFEvaluation
    qp: CLFQPResult
    constraint_drift: float
    dissipation_rate: float
    control_reference: FloatArray
    trim_reference: FloatArray
    trim_activation: float
    trim_metric: float

    def __post_init__(self) -> None:
        desired_velocity = np.asarray(self.desired_velocity, dtype=float)
        unlimited_desired_velocity = np.asarray(
            self.unlimited_desired_velocity,
            dtype=float,
        )
        filter_command = np.asarray(self.filter_command, dtype=float)
        control_reference = np.asarray(self.control_reference, dtype=float)
        trim_reference = np.asarray(self.trim_reference, dtype=float)

        if desired_velocity.ndim != 1:
            raise ValueError("desired_velocity must be one-dimensional.")
        if unlimited_desired_velocity.shape != desired_velocity.shape:
            raise ValueError("unlimited_desired_velocity must match desired_velocity shape.")
        if filter_command.shape != desired_velocity.shape:
            raise ValueError("filter_command must match desired_velocity shape.")
        if control_reference.shape != self.qp.control.shape:
            raise ValueError("control_reference must match optimized-control shape.")
        if trim_reference.shape != self.qp.control.shape:
            raise ValueError("trim_reference must match optimized-control shape.")
        if not np.isfinite(self.constraint_drift):
            raise ValueError("constraint_drift must be finite.")
        if not np.isfinite(self.dissipation_rate) or self.dissipation_rate < -1e-12:
            raise ValueError("dissipation_rate must be finite and nonnegative.")
        if not np.isfinite(self.trim_activation) or not 0.0 <= self.trim_activation <= 1.0:
            raise ValueError("trim_activation must lie in [0, 1].")
        if not np.isfinite(self.trim_metric) or self.trim_metric < 0.0:
            raise ValueError("trim_metric must be finite and nonnegative.")

        object.__setattr__(self, "desired_velocity", desired_velocity)
        object.__setattr__(
            self,
            "unlimited_desired_velocity",
            unlimited_desired_velocity,
        )
        object.__setattr__(self, "filter_command", filter_command)
        object.__setattr__(self, "control_reference", control_reference)
        object.__setattr__(self, "trim_reference", trim_reference)

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
    velocity_error_gain:
        Positive-definite matrix ``K_e`` in the paper inequality.  If omitted,
        the controller retains the legacy generic ``-alpha(W)`` QP condition;
        the canonical BlueROV2 design always supplies ``K_e``.
    trim_control_allocator:
        Optional map from a desired body restoring wrench to an admissible
        optimized-input vector.  It is called only when trim activation is
        nonzero.
    trim_activation_on, trim_activation_off:
        Thresholds for the metric
        ``sqrt(b.T @ H^{-1} @ b)``.  Trim is fully active below ``on``, fully
        inactive above ``off``, and connected by a C2 quintic smoothstep.
    """

    virtual_gain: FloatArray
    command_filter: CommandFilter
    clf: BacksteppingCLF
    qp: CLFQP
    virtual_velocity_norm_limits: FloatArray | None = None
    virtual_velocity_group_sizes: tuple[int, ...] | None = None
    velocity_error_gain: FloatArray | None = None
    trim_control_allocator: Callable[[FloatArray], FloatArray | None] | None = None
    trim_activation_on: float = 0.5
    trim_activation_off: float = 2.0

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

        velocity_error_gain = self.velocity_error_gain
        if velocity_error_gain is not None:
            velocity_error_gain = _positive_definite_matrix(
                velocity_error_gain,
                dimension,
                name="velocity_error_gain",
            )

        if self.trim_control_allocator is not None and not callable(self.trim_control_allocator):
            raise ValueError("trim_control_allocator must be callable or None.")

        if not np.isfinite(self.trim_activation_on) or self.trim_activation_on < 0.0:
            raise ValueError("trim_activation_on must be finite and nonnegative.")
        if not np.isfinite(self.trim_activation_off):
            raise ValueError("trim_activation_off must be finite.")
        if self.trim_activation_off <= self.trim_activation_on:
            raise ValueError("trim_activation_off must be greater than trim_activation_on.")

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
        object.__setattr__(
            self,
            "velocity_error_gain",
            velocity_error_gain,
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

    def _paper_constraint_terms(
        self,
        *,
        clf_evaluation: BacksteppingCLFEvaluation,
        configuration_gradient: FloatArray,
        dynamics_bias: FloatArray,
        filtered_velocity_derivative: FloatArray,
    ) -> tuple[float, float]:
        """Return ``(a_c, e_nu.T K_e e_nu)`` from the paper formulation."""
        error = np.asarray(clf_evaluation.velocity_error, dtype=float)
        gradient = _vector(
            configuration_gradient,
            self.velocity_dim,
            name="configuration_gradient",
        )
        bias = _vector(
            dynamics_bias,
            self.velocity_dim,
            name="dynamics_bias",
        )
        filtered_derivative = _vector(
            filtered_velocity_derivative,
            self.velocity_dim,
            name="filtered_velocity_derivative",
        )

        constraint_drift = float(
            error
            @ (
                gradient
                - bias
                - self.clf.inertia @ filtered_derivative
            )
        )
        if self.velocity_error_gain is None:
            raise RuntimeError(
                "paper constraint terms require velocity_error_gain."
            )
        dissipation_rate = float(
            error @ self.velocity_error_gain @ error
        )
        return constraint_drift, max(dissipation_rate, 0.0)

    def _trim_metric(
        self,
        clf_evaluation: BacksteppingCLFEvaluation,
    ) -> float:
        """Return ``sqrt(b.T H^{-1} b)`` in the QP control metric."""
        gradient = np.asarray(clf_evaluation.control_gradient, dtype=float)
        dual = np.linalg.solve(self.qp.control_weight, gradient)
        metric_squared = float(gradient @ dual)
        return float(np.sqrt(max(metric_squared, 0.0)))

    def _trim_control_reference(
        self,
        *,
        clf_evaluation: BacksteppingCLFEvaluation,
        trim_wrench: FloatArray | None,
        explicit_reference: FloatArray | None,
        control_lower: FloatArray | None,
        control_upper: FloatArray | None,
    ) -> tuple[FloatArray, FloatArray, float, float]:
        """Return actual reference, full trim candidate, weight, and metric.

        An explicitly supplied control reference retains its previous semantics
        and bypasses trim activation.  Otherwise the restoring-wrench trim is
        introduced only close to ``b = 0``.
        """
        if explicit_reference is not None:
            reference = np.asarray(explicit_reference, dtype=float)
            if reference.shape != (self.control_dim,):
                raise ValueError(
                    f"control_reference must have shape ({self.control_dim},), "
                    f"got {reference.shape}."
                )
            if not np.all(np.isfinite(reference)):
                raise ValueError("control_reference must contain only finite values.")
            return (
                reference.copy(),
                np.zeros(self.control_dim),
                0.0,
                self._trim_metric(clf_evaluation),
            )

        metric = self._trim_metric(clf_evaluation)
        if trim_wrench is None or self.trim_control_allocator is None:
            return (
                np.zeros(self.control_dim),
                np.zeros(self.control_dim),
                0.0,
                metric,
            )

        activation = _descending_quintic_activation(
            metric,
            on=self.trim_activation_on,
            off=self.trim_activation_off,
        )
        if activation <= 0.0:
            return (
                np.zeros(self.control_dim),
                np.zeros(self.control_dim),
                0.0,
                metric,
            )

        wrench = np.asarray(trim_wrench, dtype=float)
        if wrench.ndim != 1 or not np.all(np.isfinite(wrench)):
            raise ValueError("trim_wrench must be a finite one-dimensional array.")

        candidate_raw = self.trim_control_allocator(wrench)
        if candidate_raw is None:
            return (
                np.zeros(self.control_dim),
                np.zeros(self.control_dim),
                0.0,
                metric,
            )

        candidate = np.asarray(candidate_raw, dtype=float)
        if candidate.shape != (self.control_dim,):
            raise ValueError(
                "trim_control_allocator returned a vector with the wrong dimension."
            )
        if not np.all(np.isfinite(candidate)):
            raise ValueError("trim_control_allocator returned a non-finite vector.")

        # A trim point outside a dynamic box override is not an admissible
        # stationary input for this control instant.  In that case disable the
        # trim bias rather than silently clipping it and destroying B f_tr = g.
        lower = np.full(self.control_dim, -np.inf)
        upper = np.full(self.control_dim, np.inf)
        if self.qp.control_set.lower is not None:
            lower = np.asarray(self.qp.control_set.lower, dtype=float)
        if self.qp.control_set.upper is not None:
            upper = np.asarray(self.qp.control_set.upper, dtype=float)
        if control_lower is not None:
            lower_override = np.asarray(control_lower, dtype=float)
            if lower_override.shape != (self.control_dim,):
                raise ValueError(
                    f"control_lower must have shape ({self.control_dim},), "
                    f"got {lower_override.shape}."
                )
            if np.any(np.isnan(lower_override)):
                raise ValueError("control_lower must not contain NaN.")
            lower = np.maximum(lower, lower_override)
        if control_upper is not None:
            upper_override = np.asarray(control_upper, dtype=float)
            if upper_override.shape != (self.control_dim,):
                raise ValueError(
                    f"control_upper must have shape ({self.control_dim},), "
                    f"got {upper_override.shape}."
                )
            if np.any(np.isnan(upper_override)):
                raise ValueError("control_upper must not contain NaN.")
            upper = np.minimum(upper, upper_override)
        if np.any(lower > upper):
            raise ValueError("control_lower and control_upper define an empty box.")

        if (
            np.any(candidate < lower - 1e-9)
            or np.any(candidate > upper + 1e-9)
            or not self.qp.control_set.contains(candidate, tolerance=1e-8)
        ):
            return (
                np.zeros(self.control_dim),
                candidate,
                0.0,
                metric,
            )

        return (
            activation * candidate,
            candidate,
            activation,
            metric,
        )

    def _solve_paper_qp(
        self,
        *,
        clf_evaluation: BacksteppingCLFEvaluation,
        configuration_gradient: FloatArray,
        dynamics_bias: FloatArray,
        filtered_velocity_derivative: FloatArray,
        control_reference: FloatArray | None,
        trim_wrench: FloatArray | None,
        control_lower: FloatArray | None,
        control_upper: FloatArray | None,
    ) -> tuple[CLFQPResult, float, float, FloatArray, FloatArray, float, float]:
        if self.velocity_error_gain is None:
            # Backward-compatible path for generic second-order controllers
            # that have not opted into the updated paper inequality.
            reference = (
                np.zeros(self.control_dim)
                if control_reference is None
                else np.asarray(control_reference, dtype=float)
            )
            result = self.qp.solve(
                clf_evaluation,
                control_reference=reference,
                lower_override=control_lower,
                upper_override=control_upper,
            )
            return (
                result,
                float(clf_evaluation.drift),
                float(self.qp.alpha(clf_evaluation.value)),
                reference.copy(),
                np.zeros(self.control_dim),
                0.0,
                self._trim_metric(clf_evaluation),
            )

        constraint_drift, dissipation_rate = self._paper_constraint_terms(
            clf_evaluation=clf_evaluation,
            configuration_gradient=configuration_gradient,
            dynamics_bias=dynamics_bias,
            filtered_velocity_derivative=filtered_velocity_derivative,
        )
        (
            reference,
            trim_reference,
            trim_activation,
            trim_metric,
        ) = self._trim_control_reference(
            clf_evaluation=clf_evaluation,
            trim_wrench=trim_wrench,
            explicit_reference=control_reference,
            control_lower=control_lower,
            control_upper=control_upper,
        )

        result = self.qp.solve(
            clf_evaluation,
            control_reference=reference,
            constraint_drift=constraint_drift,
            dissipation_rate=dissipation_rate,
            lower_override=control_lower,
            upper_override=control_upper,
        )
        return (
            result,
            constraint_drift,
            dissipation_rate,
            reference,
            trim_reference,
            trim_activation,
            trim_metric,
        )

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
        trim_wrench: FloatArray | None = None,
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

        (
            qp_result,
            constraint_drift,
            dissipation_rate,
            resolved_reference,
            trim_reference,
            trim_activation,
            trim_metric,
        ) = self._solve_paper_qp(
            clf_evaluation=clf_evaluation,
            configuration_gradient=gradient,
            dynamics_bias=dynamics_bias,
            filtered_velocity_derivative=total_filter.output_derivative,
            control_reference=control_reference,
            trim_wrench=trim_wrench,
            control_lower=control_lower,
            control_upper=control_upper,
        )

        return SecondOrderControllerEvaluation(
            desired_velocity=desired_velocity,
            unlimited_desired_velocity=unlimited_desired_velocity,
            filter_command=feedback_command,
            filter=total_filter,
            clf=clf_evaluation,
            qp=qp_result,
            constraint_drift=constraint_drift,
            dissipation_rate=dissipation_rate,
            control_reference=resolved_reference,
            trim_reference=trim_reference,
            trim_activation=trim_activation,
            trim_metric=trim_metric,
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
        trim_wrench: FloatArray | None = None,
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

        (
            qp_result,
            constraint_drift,
            dissipation_rate,
            resolved_reference,
            trim_reference,
            trim_activation,
            trim_metric,
        ) = self._solve_paper_qp(
            clf_evaluation=clf_evaluation,
            configuration_gradient=gradient,
            dynamics_bias=dynamics_bias,
            filtered_velocity_derivative=filter_evaluation.output_derivative,
            control_reference=control_reference,
            trim_wrench=trim_wrench,
            control_lower=control_lower,
            control_upper=control_upper,
        )

        return SecondOrderControllerEvaluation(
            desired_velocity=desired_velocity,
            unlimited_desired_velocity=unlimited_desired_velocity,
            filter_command=desired_velocity,
            filter=filter_evaluation,
            clf=clf_evaluation,
            qp=qp_result,
            constraint_drift=constraint_drift,
            dissipation_rate=dissipation_rate,
            control_reference=resolved_reference,
            trim_reference=trim_reference,
            trim_activation=trim_activation,
            trim_metric=trim_metric,
        )
