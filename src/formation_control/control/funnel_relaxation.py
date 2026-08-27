"""Smooth barrier-potential enlargement of conservative sensing domains.

For each scalar constraint the adaptive domain is

    h_a(x, s) = h_c(x) + rho_max s,
    s >= 0,

where ``s = 0`` recovers the conservative domain and ``s = 1`` places the
zero level set at the corresponding physical limit.  The enlargement state is
not artificially capped at one: values above one quantify how much additional
domain enlargement the auxiliary dynamics request.  Physical admissibility is
reported separately.

The auxiliary dynamics use only the current constraint value,

    s_dot = -k_s s + k_b sigma(y) rho_max / y,
    y = h_a - h_margin,

with a C2 smoothstep ``sigma``.  The barrier action is zero away from the
adaptive boundary, turns on smoothly, and becomes singular as ``y -> 0+``.
Consequently no parent velocity or derivative of ``h_c`` is required by the
adaptation law.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from formation_control.models.base import FloatArray

FUNNEL_CHANNELS = (
    "collision",
    "range",
    "horizontal_fov",
    "vertical_fov",
)


def _vector4(value: float | FloatArray, *, name: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(4, float(array))
    if array.shape != (4,):
        raise ValueError(f"{name} must be scalar or have shape (4,).")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


class FunnelRelaxationInfeasibleError(RuntimeError):
    """Raised when a channel has no enlargement reserve near singularity."""


@dataclass(frozen=True)
class FunnelRelaxationEvaluation:
    """One evaluation of the smooth auxiliary enlargement dynamics."""

    state: FloatArray
    enlargement: FloatArray
    reference_rate: FloatArray
    selected_rate: FloatArray
    adaptive_constraint_values: FloatArray
    residual_margin: FloatArray
    activation: FloatArray
    barrier_rate: FloatArray
    expansion_required: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "state",
            "enlargement",
            "reference_rate",
            "selected_rate",
            "adaptive_constraint_values",
            "residual_margin",
            "activation",
            "barrier_rate",
        ):
            array = np.asarray(getattr(self, name), dtype=float)
            if array.shape != (4,):
                raise ValueError(f"{name} must have shape (4,).")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must contain only finite values.")
            object.__setattr__(self, name, array.copy())

        required = np.asarray(self.expansion_required, dtype=bool)
        if required.shape != (4,):
            raise ValueError("expansion_required must have shape (4,).")
        object.__setattr__(self, "expansion_required", required.copy())


@dataclass(frozen=True)
class FunnelRelaxationPolicy:
    """Four-channel smooth barrier-gradient enlargement controller."""

    maximum_enlargement: FloatArray
    recovery_gain: FloatArray | float = 0.8
    barrier_gain: FloatArray | float = 0.20
    domain_margin_ratio: float = 0.10
    activation_on_ratio: float = 0.10
    activation_off_ratio: float = 0.30
    minimum_constraint_margin: float = 1e-5

    def __post_init__(self) -> None:
        maximum = _vector4(self.maximum_enlargement, name="maximum_enlargement")
        recovery = _vector4(self.recovery_gain, name="recovery_gain")
        barrier = _vector4(self.barrier_gain, name="barrier_gain")

        if np.any(maximum < 0.0):
            raise ValueError("maximum_enlargement must be nonnegative.")
        if np.any(recovery < 0.0):
            raise ValueError("recovery_gain must be nonnegative.")
        if np.any(barrier < 0.0):
            raise ValueError("barrier_gain must be nonnegative.")
        if not np.isfinite(self.domain_margin_ratio) or not 0.0 < self.domain_margin_ratio < 1.0:
            raise ValueError("domain_margin_ratio must lie strictly in (0, 1).")
        if not np.isfinite(self.activation_on_ratio) or self.activation_on_ratio <= 0.0:
            raise ValueError("activation_on_ratio must be finite and positive.")
        if not np.isfinite(self.activation_off_ratio) or (
            self.activation_off_ratio <= self.activation_on_ratio
        ):
            raise ValueError("activation_off_ratio must exceed activation_on_ratio.")
        if not np.isfinite(self.minimum_constraint_margin) or (
            self.minimum_constraint_margin <= 0.0
        ):
            raise ValueError("minimum_constraint_margin must be finite and positive.")

        object.__setattr__(self, "maximum_enlargement", maximum)
        object.__setattr__(self, "recovery_gain", recovery)
        object.__setattr__(self, "barrier_gain", barrier)

    @property
    def domain_margin(self) -> FloatArray:
        """Positive adaptive-domain margin h_margin."""
        return np.maximum(
            self.minimum_constraint_margin,
            self.domain_margin_ratio * self.maximum_enlargement,
        )

    @property
    def activation_on_margin(self) -> FloatArray:
        """Residual margin below which the barrier action is fully active."""
        return np.maximum(
            self.minimum_constraint_margin,
            self.activation_on_ratio * self.maximum_enlargement,
        )

    @property
    def activation_off_margin(self) -> FloatArray:
        """Residual margin above which the barrier action is exactly zero."""
        return np.maximum(
            2.0 * self.minimum_constraint_margin,
            self.activation_off_ratio * self.maximum_enlargement,
        )

    def initialize(self) -> FloatArray:
        return np.zeros(4, dtype=float)

    def validate_state(self, state: FloatArray) -> FloatArray:
        state = _vector4(state, name="state")
        if np.any(state < -1e-9):
            raise ValueError("normalized funnel state must be nonnegative.")
        # Remove only tiny integration roundoff below zero.  There is
        # intentionally no upper clipping.
        return np.maximum(state, 0.0)

    def enlargement(self, state: FloatArray) -> FloatArray:
        return self.maximum_enlargement * self.validate_state(state)

    def reference_rate(self, state: FloatArray) -> FloatArray:
        state = self.validate_state(state)
        return -self.recovery_gain * state

    @staticmethod
    def _smoothstep_activation(y: float, y_on: float, y_off: float) -> float:
        """C2 activation: one near the boundary and zero sufficiently far away."""
        if y <= y_on:
            return 1.0
        if y >= y_off:
            return 0.0
        xi = (y_off - y) / (y_off - y_on)
        return float(6.0 * xi**5 - 15.0 * xi**4 + 10.0 * xi**3)

    def project_to_current_domain(
        self,
        state: FloatArray,
        conservative_values: FloatArray,
        *,
        enabled: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        """Emergency initialization/roundoff repair for the logarithmic domain.

        This is not part of the nominal adaptation law.  If an adaptive
        constraint has already reached the numerical logarithm floor, increase
        ``s`` just enough to restore the prescribed positive margin.  No upper
        clipping is applied.
        """
        state = self.validate_state(state)
        base = _vector4(conservative_values, name="conservative_values")
        active = np.asarray(enabled, dtype=bool)
        if active.shape != (4,):
            raise ValueError("enabled must have shape (4,).")

        projected = state.copy()
        correction = np.zeros(4, dtype=float)
        margin = self.domain_margin

        for index in range(4):
            if not active[index]:
                continue
            maximum = self.maximum_enlargement[index]
            adaptive_value = base[index] + maximum * projected[index]
            if adaptive_value > self.minimum_constraint_margin:
                continue
            if maximum <= 0.0:
                raise FunnelRelaxationInfeasibleError(
                    f"{FUNNEL_CHANNELS[index]} has no relaxation reserve."
                )
            required_state = max(0.0, (margin[index] - base[index]) / maximum)
            correction[index] = max(0.0, required_state - projected[index])
            projected[index] = max(projected[index], required_state)

        return projected, correction

    def evaluate(
        self,
        state: FloatArray,
        *,
        conservative_values: FloatArray,
        enabled: FloatArray,
        # Accepted for backward compatibility with the previous sampled-data
        # implementation.  They are intentionally unused by this law.
        conservative_rates: FloatArray | None = None,
        sample_time: float | None = None,
    ) -> FunnelRelaxationEvaluation:
        """Evaluate the continuous-time smooth barrier-gradient law.

        For every enabled channel

            y = h_c + rho_max s - h_margin,
            s_dot = -k_s s + k_b sigma(y) rho_max / y.

        A tiny positive denominator is used only as a discrete-time numerical
        safeguard if a caller evaluates the law after an undershoot of the
        theoretical boundary.
        """
        del conservative_rates, sample_time

        state = self.validate_state(state)
        base = _vector4(conservative_values, name="conservative_values")
        active = np.asarray(enabled, dtype=bool)
        if active.shape != (4,):
            raise ValueError("enabled must have shape (4,).")

        rho = self.maximum_enlargement * state
        adaptive = base + rho
        residual = adaptive - self.domain_margin
        reference = self.reference_rate(state)
        selected = np.zeros(4, dtype=float)
        activation = np.zeros(4, dtype=float)
        barrier_rate = np.zeros(4, dtype=float)
        expansion_required = np.zeros(4, dtype=bool)

        y_on = self.activation_on_margin
        y_off = self.activation_off_margin

        for index in range(4):
            if not active[index]:
                continue

            maximum = self.maximum_enlargement[index]
            if maximum <= 0.0:
                if adaptive[index] <= self.minimum_constraint_margin:
                    raise FunnelRelaxationInfeasibleError(
                        f"{FUNNEL_CHANNELS[index]} has no relaxation reserve."
                    )
                continue

            sigma = self._smoothstep_activation(
                residual[index], y_on[index], y_off[index]
            )
            activation[index] = sigma

            # The continuous law is defined for y>0.  max(...) is only a
            # numerical safeguard for finite-step integration/measurement noise.
            y_safe = max(float(residual[index]), self.minimum_constraint_margin)
            barrier_rate[index] = (
                self.barrier_gain[index] * sigma * maximum / y_safe
            )
            selected[index] = reference[index] + barrier_rate[index]
            expansion_required[index] = selected[index] > 1e-12

        return FunnelRelaxationEvaluation(
            state=state,
            enlargement=rho,
            reference_rate=reference,
            selected_rate=selected,
            adaptive_constraint_values=adaptive,
            residual_margin=residual,
            activation=activation,
            barrier_rate=barrier_rate,
            expansion_required=expansion_required,
        )
