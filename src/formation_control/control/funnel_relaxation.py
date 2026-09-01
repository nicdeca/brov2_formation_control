"""Smooth barrier-potential enlargement of conservative sensing domains.

For each scalar constraint the adaptive domain is

    h_a(x, s) = h_c(x) + rho_max s,
    s in [0, 1],

where ``s = 0`` recovers the conservative domain and ``s = 1`` makes the
adaptive constraint coincide with the corresponding physical constraint.

The state-dependent inner margin is

    h_inner(s) = mu_s rho_max (1 - s),

so that

    y = h_a - h_inner
      = h_c - mu_s rho_max + (1 + mu_s) rho_max s.

The unconstrained enlargement direction is

    v = -k_s s + k_b (1 + mu_s) rho_max sigma(y) / y,

and the actual continuous-time dynamics apply the upper projection

    s_dot = Pi_{<=1}(s, v).

No lower projection is needed: at ``s = 0`` the unprojected vector field is
nonnegative.  The implementation uses an implicit sampled-data step to avoid
large explicit-Euler increments close to ``y = 0``.
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
    """Raised when the physical domain cannot accommodate a channel."""


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

    # Paper notation: mu_s.  The historical name is retained to avoid
    # unnecessary configuration/API changes elsewhere in the codebase.
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
        if not np.isfinite(self.domain_margin_ratio) or not (
            0.0 < self.domain_margin_ratio < 1.0
        ):
            raise ValueError("domain_margin_ratio must lie strictly in (0, 1).")
        if not np.isfinite(self.activation_on_ratio) or (
            self.activation_on_ratio <= 0.0
        ):
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
        """Inner margin at s=0, i.e. mu_s rho_max.

        Kept as a property for backward compatibility.  The current
        state-dependent margin is returned by :meth:`inner_margin`.
        """
        return self.domain_margin_ratio * self.maximum_enlargement

    def inner_margin(self, state: FloatArray) -> FloatArray:
        """State-dependent inner margin mu_s rho_max (1-s)."""
        state = self.validate_state(state)
        return (
            self.domain_margin_ratio
            * self.maximum_enlargement
            * (1.0 - state)
        )

    @property
    def adaptive_margin_state_gain(self) -> FloatArray:
        """Derivative dy/ds = (1 + mu_s) rho_max."""
        return (1.0 + self.domain_margin_ratio) * self.maximum_enlargement

    @property
    def activation_on_margin(self) -> FloatArray:
        """Adaptive margin below which the barrier action is fully active."""
        return np.maximum(
            self.minimum_constraint_margin,
            self.activation_on_ratio * self.maximum_enlargement,
        )

    @property
    def activation_off_margin(self) -> FloatArray:
        """Adaptive margin above which the barrier action is exactly zero."""
        return np.maximum(
            2.0 * self.minimum_constraint_margin,
            self.activation_off_ratio * self.maximum_enlargement,
        )

    def initialize(self) -> FloatArray:
        return np.zeros(4, dtype=float)

    def validate_state(self, state: FloatArray) -> FloatArray:
        state = _vector4(state, name="state")
        tolerance = 1e-9
        if np.any(state < -tolerance) or np.any(state > 1.0 + tolerance):
            raise ValueError("normalized funnel state must lie in [0, 1].")
        # Numerical cleanup only; the continuous dynamics themselves require
        # only the upper projection because the vector field is nonnegative
        # at s=0.
        return np.clip(state, 0.0, 1.0)

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

    def _adaptive_margin_value(
        self,
        index: int,
        state_value: float,
        conservative_value: float,
    ) -> float:
        maximum = float(self.maximum_enlargement[index])
        mu_s = float(self.domain_margin_ratio)
        return (
            conservative_value
            - mu_s * maximum
            + (1.0 + mu_s) * maximum * state_value
        )

    def _state_for_margin(
        self,
        index: int,
        conservative_value: float,
        target_margin: float,
    ) -> float:
        """Smallest s producing the requested adaptive margin.

        If the requested numerical margin cannot be attained before s=1 but
        the physical constraint is still strictly satisfied, the result is
        saturated at one.  If even the physical constraint is nonpositive,
        the channel is infeasible.
        """
        maximum = float(self.maximum_enlargement[index])
        mu_s = float(self.domain_margin_ratio)

        if maximum <= 0.0:
            if conservative_value <= 0.0:
                raise FunnelRelaxationInfeasibleError(
                    f"{FUNNEL_CHANNELS[index]} has no relaxation reserve."
                )
            return 0.0

        physical_value = conservative_value + maximum
        if physical_value <= 0.0:
            raise FunnelRelaxationInfeasibleError(
                f"{FUNNEL_CHANNELS[index]} physical constraint is not satisfied."
            )

        coefficient = (1.0 + mu_s) * maximum
        required = (
            target_margin
            - conservative_value
            + mu_s * maximum
        ) / coefficient

        return min(1.0, max(0.0, required))

    def project_to_current_domain(
        self,
        state: FloatArray,
        conservative_values: FloatArray,
        *,
        enabled: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        """Emergency sampled-data repair of the adaptive-margin domain.

        This is not part of the nominal adaptation law.  If a sampled
        measurement has already reached the numerical neighborhood of

            y = h_c - mu_s rho_max + (1 + mu_s) rho_max s = 0,

        ``s`` is increased only enough to recover the numerical margin when
        possible, while always respecting ``s <= 1``.  If the requested
        numerical floor cannot be reached but the physical constraint remains
        strictly positive, the repair saturates at ``s = 1``.  A nonpositive
        physical constraint is reported as infeasible.
        """
        state = self.validate_state(state)
        base = _vector4(conservative_values, name="conservative_values")
        active = np.asarray(enabled, dtype=bool)
        if active.shape != (4,):
            raise ValueError("enabled must have shape (4,).")

        projected = state.copy()
        correction = np.zeros(4, dtype=float)
        residual_floor = self.minimum_constraint_margin

        for index in range(4):
            if not active[index]:
                continue

            residual = self._adaptive_margin_value(
                index,
                float(projected[index]),
                float(base[index]),
            )
            if residual > residual_floor:
                continue

            required_state = self._state_for_margin(
                index,
                float(base[index]),
                residual_floor,
            )
            correction[index] = max(
                0.0,
                required_state - projected[index],
            )
            projected[index] = max(projected[index], required_state)

        projected = self.validate_state(projected)
        return projected, correction

    def advance(
        self,
        state: FloatArray,
        *,
        conservative_values: FloatArray,
        enabled: FloatArray,
        sample_time: float,
    ) -> tuple[FloatArray, FloatArray]:
        """Advance the projected auxiliary dynamics with an implicit step.

        For ``s < 1`` the continuous law is

            s_dot = -k_s s
                    + k_b (1 + mu_s) rho_max sigma(y) / y,

        while at ``s = 1`` its positive component is removed by the upper
        projection.  The current sampled ``h_c`` is held fixed during the
        implicit step.  The returned rate is the effective sampled-data rate
        ``(s_{k+1} - s_k) / dt``.
        """
        if not np.isfinite(sample_time) or sample_time <= 0.0:
            raise ValueError("sample_time must be finite and positive.")

        base = _vector4(conservative_values, name="conservative_values")
        active = np.asarray(enabled, dtype=bool)
        if active.shape != (4,):
            raise ValueError("enabled must have shape (4,).")

        state, _ = self.project_to_current_domain(
            state,
            base,
            enabled=active,
        )
        next_state = state.copy()

        y_on = self.activation_on_margin
        y_off = self.activation_off_margin
        residual_floor = self.minimum_constraint_margin
        mu_s = float(self.domain_margin_ratio)

        for index in range(4):
            if not active[index]:
                continue

            maximum = float(self.maximum_enlargement[index])
            if maximum <= 0.0:
                continue

            recovery = float(self.recovery_gain[index])
            barrier = float(self.barrier_gain[index])
            old_state = float(state[index])
            state_gain = (1.0 + mu_s) * maximum

            minimum_state = self._state_for_margin(
                index,
                float(base[index]),
                residual_floor,
            )

            if barrier <= 0.0:
                candidate = old_state / (1.0 + sample_time * recovery)
                next_state[index] = min(
                    1.0,
                    max(candidate, minimum_state),
                )
                continue

            def equation(candidate: float) -> float:
                y = self._adaptive_margin_value(
                    index,
                    candidate,
                    float(base[index]),
                )
                y_safe = max(y, residual_floor)
                sigma = self._smoothstep_activation(
                    y_safe,
                    float(y_on[index]),
                    float(y_off[index]),
                )
                barrier_rate = barrier * sigma * state_gain / y_safe
                return (
                    (1.0 + sample_time * recovery) * candidate
                    - old_state
                    - sample_time * barrier_rate
                )

            lower = minimum_state
            upper = 1.0

            # If even the numerical floor is only attainable at the physical
            # limit, remain at that limit.
            if lower >= upper:
                next_state[index] = upper
                continue

            f_lower = equation(lower)
            if f_lower >= 0.0:
                next_state[index] = lower
                continue

            f_upper = equation(upper)

            # The unconstrained implicit step would lie above one.  This is
            # precisely the outward direction removed by Pi_{<=1}.
            if f_upper <= 0.0:
                next_state[index] = upper
                continue

            for _ in range(60):
                middle = 0.5 * (lower + upper)
                if equation(middle) <= 0.0:
                    lower = middle
                else:
                    upper = middle

            next_state[index] = 0.5 * (lower + upper)

        next_state = self.validate_state(next_state)
        effective_rate = (next_state - state) / sample_time
        return next_state, effective_rate

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
        """Evaluate the continuous-time projected adaptation law.

        For every enabled channel,

            y = h_c - mu_s rho_max + (1 + mu_s) rho_max s,

            v = -k_s s
                + k_b (1 + mu_s) rho_max sigma(y) / y,

            s_dot = Pi_{<=1}(s, v).

        A tiny positive denominator is used only as a discrete-time numerical
        safeguard if a caller evaluates the law after a finite-step undershoot.
        """
        del conservative_rates, sample_time

        state = self.validate_state(state)
        base = _vector4(conservative_values, name="conservative_values")
        active = np.asarray(enabled, dtype=bool)
        if active.shape != (4,):
            raise ValueError("enabled must have shape (4,).")

        rho = self.maximum_enlargement * state
        adaptive = base + rho
        residual = adaptive - self.inner_margin(state)
        reference = self.reference_rate(state)
        selected = np.zeros(4, dtype=float)
        activation = np.zeros(4, dtype=float)
        barrier_rate = np.zeros(4, dtype=float)
        expansion_required = np.zeros(4, dtype=bool)

        y_on = self.activation_on_margin
        y_off = self.activation_off_margin
        state_gain = self.adaptive_margin_state_gain

        for index in range(4):
            if not active[index]:
                continue

            maximum = self.maximum_enlargement[index]
            if maximum <= 0.0:
                if adaptive[index] <= 0.0:
                    raise FunnelRelaxationInfeasibleError(
                        f"{FUNNEL_CHANNELS[index]} has no relaxation reserve."
                    )
                continue

            sigma = self._smoothstep_activation(
                residual[index],
                y_on[index],
                y_off[index],
            )
            activation[index] = sigma

            # The continuous law is defined for y>0.  max(...) is only a
            # numerical safeguard for finite-step integration/measurement noise.
            y_safe = max(
                float(residual[index]),
                self.minimum_constraint_margin,
            )
            barrier_rate[index] = (
                self.barrier_gain[index]
                * sigma
                * state_gain[index]
                / y_safe
            )

            unprojected_rate = reference[index] + barrier_rate[index]

            # Upper projection Pi_{<=1}.  No lower projection is necessary:
            # at s=0, reference=0 and the barrier term is nonnegative.
            if state[index] >= 1.0 - 1e-12:
                selected[index] = min(0.0, unprojected_rate)
            else:
                selected[index] = unprojected_rate

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
