"""Adaptive enlargement of conservative sensing domains.

This module implements the relaxation law used in the paper.  For each
constraint channel ``ell``

    h_a = h_c + rho_max * s,                0 <= s <= 1,
    h_d_a = h_d_c + rho_max * s,

and the common shift preserves ``h_a - h_d_a = h_c - h_d_c``.  The
recentered-barrier contribution satisfies

    D = dV/ds
      = -mu * rho_max * (h_a - h_d_a)^2 / (h_a * h_d_a^2) <= 0.

The adaptive state follows

    s_dot = Pi_{<=1}(s,
        -k_s (1 - sigma(h_a)) s
        -k_b gamma sigma(h_a) D),

where ``sigma`` activates near the adaptive boundary.  The gate ``gamma`` is built only from the minimum CLF relaxation required by
the actuator box, exactly as in the paper.  The state is capped at one, where
the adaptive boundary coincides with the physical limit.

The sampled-data implementation adds numerical devices that do not alter the
continuous-time law: an implicit update for ``s``, a one-step predictive guard
based on finite differences of the conservative constraint values, and a small
emergency projection if an unexpected inter-sample crossing still occurs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

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
    """Raised when the physical enlargement budget cannot restore h_a > 0."""


@dataclass(frozen=True)
class FunnelRelaxationEvaluation:
    """One evaluation of the paper's adaptive-domain dynamics."""

    state: FloatArray
    enlargement: FloatArray
    selected_rate: FloatArray
    adaptive_constraint_values: FloatArray
    desired_adaptive_constraint_values: FloatArray
    activation: FloatArray
    barrier_derivative: FloatArray
    recovery_rate: FloatArray
    enlargement_rate: FloatArray
    infeasibility_activation: float
    expansion_required: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "state",
            "enlargement",
            "selected_rate",
            "adaptive_constraint_values",
            "desired_adaptive_constraint_values",
            "activation",
            "barrier_derivative",
            "recovery_rate",
            "enlargement_rate",
        ):
            array = np.asarray(getattr(self, name), dtype=float)
            if array.shape != (4,):
                raise ValueError(f"{name} must have shape (4,).")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must contain only finite values.")
            object.__setattr__(self, name, array.copy())

        gamma = float(self.infeasibility_activation)
        if not np.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
            raise ValueError("infeasibility_activation must lie in [0, 1].")
        object.__setattr__(self, "infeasibility_activation", gamma)

        required = np.asarray(self.expansion_required, dtype=bool)
        if required.shape != (4,):
            raise ValueError("expansion_required must have shape (4,).")
        object.__setattr__(self, "expansion_required", required.copy())

    # Backward-compatible aliases used by some diagnostic code.
    @property
    def reference_rate(self) -> FloatArray:
        return self.recovery_rate.copy()

    @property
    def barrier_rate(self) -> FloatArray:
        return self.enlargement_rate.copy()


@dataclass(frozen=True)
class FunnelRelaxationPolicy:
    """Four-channel adaptive-domain controller matching Sec. III-D."""

    maximum_enlargement: FloatArray
    desired_conservative_values: FloatArray
    barrier_weights: FloatArray
    recovery_gain: FloatArray | float = 0.8
    barrier_gain: FloatArray | float = 0.20
    activation_on_ratio: float = 0.10
    activation_off_ratio: float = 0.30
    infeasibility_epsilon: float = 1e-3
    minimum_constraint_margin: float = 1e-8
    sampled_guard_margin_ratio: float = 0.02

    def __post_init__(self) -> None:
        maximum = _vector4(self.maximum_enlargement, name="maximum_enlargement")
        desired = _vector4(
            self.desired_conservative_values,
            name="desired_conservative_values",
        )
        weights = _vector4(self.barrier_weights, name="barrier_weights")
        recovery = _vector4(self.recovery_gain, name="recovery_gain")
        barrier = _vector4(self.barrier_gain, name="barrier_gain")

        if np.any(maximum < 0.0):
            raise ValueError("maximum_enlargement must be nonnegative.")
        if np.any(desired <= 0.0):
            raise ValueError("desired_conservative_values must be positive.")
        if np.any(weights <= 0.0):
            raise ValueError("barrier_weights must be positive.")
        if np.any(recovery < 0.0):
            raise ValueError("recovery_gain must be nonnegative.")
        if np.any(barrier < 0.0):
            raise ValueError("barrier_gain must be nonnegative.")
        if not np.isfinite(self.activation_on_ratio) or not (
            0.0 < self.activation_on_ratio < 1.0
        ):
            raise ValueError("activation_on_ratio must lie strictly in (0, 1).")
        if not np.isfinite(self.activation_off_ratio) or not (
            self.activation_on_ratio < self.activation_off_ratio < 1.0
        ):
            raise ValueError(
                "activation_off_ratio must satisfy activation_on_ratio < "
                "activation_off_ratio < 1."
            )
        if not np.isfinite(self.infeasibility_epsilon) or (
            self.infeasibility_epsilon <= 0.0
        ):
            raise ValueError("infeasibility_epsilon must be finite and positive.")
        if not np.isfinite(self.minimum_constraint_margin) or (
            self.minimum_constraint_margin <= 0.0
        ):
            raise ValueError("minimum_constraint_margin must be finite and positive.")
        if not np.isfinite(self.sampled_guard_margin_ratio) or not (
            0.0 <= self.sampled_guard_margin_ratio < 1.0
        ):
            raise ValueError("sampled_guard_margin_ratio must lie in [0, 1).")

        object.__setattr__(self, "maximum_enlargement", maximum)
        object.__setattr__(self, "desired_conservative_values", desired)
        object.__setattr__(self, "barrier_weights", weights)
        object.__setattr__(self, "recovery_gain", recovery)
        object.__setattr__(self, "barrier_gain", barrier)

    @property
    def activation_on_margin(self) -> FloatArray:
        """h_on; by construction 0 < h_on < h_off < h_d,c."""
        return self.activation_on_ratio * self.desired_conservative_values

    @property
    def activation_off_margin(self) -> FloatArray:
        """h_off; by construction 0 < h_on < h_off < h_d,c."""
        return self.activation_off_ratio * self.desired_conservative_values

    def with_desired_conservative_values(
        self,
        desired_conservative_values: FloatArray,
    ) -> "FunnelRelaxationPolicy":
        """Return the same policy retargeted to a new formation reference."""
        return replace(
            self,
            desired_conservative_values=_vector4(
                desired_conservative_values,
                name="desired_conservative_values",
            ),
        )

    def initialize(self) -> FloatArray:
        return np.zeros(4, dtype=float)

    def validate_state(self, state: FloatArray) -> FloatArray:
        state = _vector4(state, name="state")
        tolerance = 1e-9
        if np.any(state < -tolerance) or np.any(state > 1.0 + tolerance):
            raise ValueError("normalized relaxation state must lie in [0, 1].")
        return np.clip(state, 0.0, 1.0)

    def enlargement(self, state: FloatArray) -> FloatArray:
        return self.maximum_enlargement * self.validate_state(state)

    @staticmethod
    def _smoothstep_activation(h: float, h_on: float, h_off: float) -> float:
        """C2 activation: one near h_a=0 and zero away from the boundary."""
        if h <= h_on:
            return 1.0
        if h >= h_off:
            return 0.0
        xi = (h_off - h) / (h_off - h_on)
        return float(6.0 * xi**5 - 15.0 * xi**4 + 10.0 * xi**3)

    def infeasibility_activation(self, required_slack: float) -> float:
        """Return the paper gate ``gamma(delta_req)``."""
        required = float(required_slack)
        if not np.isfinite(required):
            raise ValueError("required_slack must be finite.")
        delta = max(required, 0.0)
        epsilon = float(self.infeasibility_epsilon)
        return float(delta * delta / (delta * delta + epsilon * epsilon))

    @property
    def sampled_guard_margin(self) -> FloatArray:
        """Implementation-only positive margin used after sampled crossings."""
        return np.maximum(
            self.minimum_constraint_margin,
            self.sampled_guard_margin_ratio * self.maximum_enlargement,
        )

    def initialize_for_constraint_values(
        self,
        conservative_values: FloatArray,
        *,
        enabled: FloatArray,
    ) -> FloatArray:
        """Choose s(0) in [0,1] so every enabled adaptive barrier has h_a>0.

        This implements the initialization choice stated in the paper.  The
        smallest state that leaves the numerical margin
        ``minimum_constraint_margin`` is selected.  Existence is guaranteed
        for a state strictly inside the physical domain, up to the numerical
        floor used here.
        """
        base = _vector4(conservative_values, name="conservative_values")
        active = np.asarray(enabled, dtype=bool)
        if active.shape != (4,):
            raise ValueError("enabled must have shape (4,).")

        state = np.zeros(4, dtype=float)
        floor = float(self.minimum_constraint_margin)
        for index in range(4):
            if not active[index] or base[index] > floor:
                continue

            maximum = float(self.maximum_enlargement[index])
            if maximum <= 0.0:
                raise FunnelRelaxationInfeasibleError(
                    f"{FUNNEL_CHANNELS[index]} has no relaxation reserve."
                )

            required = (floor - float(base[index])) / maximum
            if required > 1.0 + 1e-12:
                physical_value = base[index] + maximum
                raise FunnelRelaxationInfeasibleError(
                    f"{FUNNEL_CHANNELS[index]} is outside the physical domain: "
                    f"h_c + rho_max = {physical_value:.6g}."
                )
            state[index] = float(np.clip(required, 0.0, 1.0))

        return state

    def project_to_current_domain(
        self,
        state: FloatArray,
        conservative_values: FloatArray,
        *,
        enabled: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        """Sampled-data safeguard: restore a small positive adaptive margin.

        The continuous-time paper law is unchanged.  This projection is used
        only at controller sampling instants if plant motion between samples
        has moved an enabled adaptive constraint below the implementation guard.
        """
        state = self.validate_state(state)
        base = _vector4(conservative_values, name="conservative_values")
        active = np.asarray(enabled, dtype=bool)
        if active.shape != (4,):
            raise ValueError("enabled must have shape (4,).")

        repaired = state.copy()
        guard = self.sampled_guard_margin
        for index in range(4):
            if not active[index]:
                continue
            h_a = base[index] + self.maximum_enlargement[index] * repaired[index]
            target_margin = float(guard[index])
            if h_a >= target_margin:
                continue

            maximum = float(self.maximum_enlargement[index])
            if maximum <= 0.0:
                raise FunnelRelaxationInfeasibleError(
                    f"{FUNNEL_CHANNELS[index]} has no relaxation reserve."
                )
            required = (target_margin - float(base[index])) / maximum
            if required > 1.0 + 1e-12:
                raise FunnelRelaxationInfeasibleError(
                    f"{FUNNEL_CHANNELS[index]} cannot be restored before the "
                    "physical limit."
                )
            repaired[index] = max(repaired[index], float(np.clip(required, 0.0, 1.0)))

        return repaired, repaired - state

    def project_to_predicted_domain(
        self,
        state: FloatArray,
        conservative_values: FloatArray,
        conservative_rates: FloatArray,
        *,
        sample_time: float,
        enabled: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        """Anticipate one sample of threatening constraint motion.

        Using a finite-difference estimate ``h_c_dot``, enforce the
        implementation-only condition

            h_c + dt * min(h_c_dot, 0) + rho_max * s >= h_guard.

        Only negative rates are extrapolated: motion away from a boundary does
        not reduce the current margin.  The correction is the smallest increase
        of ``s`` satisfying the one-step forecast, capped by the physical
        enlargement ``s <= 1``.  This is a sampled-data safeguard and does not
        modify the continuous-time adaptive law.
        """
        if not np.isfinite(sample_time) or sample_time <= 0.0:
            raise ValueError("sample_time must be finite and positive.")

        state = self.validate_state(state)
        base = _vector4(conservative_values, name="conservative_values")
        rates = _vector4(conservative_rates, name="conservative_rates")
        active = np.asarray(enabled, dtype=bool)
        if active.shape != (4,):
            raise ValueError("enabled must have shape (4,).")

        predicted_base = base + sample_time * np.minimum(rates, 0.0)
        repaired = state.copy()
        guard = self.sampled_guard_margin
        for index in range(4):
            if not active[index]:
                continue

            maximum = float(self.maximum_enlargement[index])
            if maximum <= 0.0:
                continue

            predicted_h_a = predicted_base[index] + maximum * repaired[index]
            target_margin = float(guard[index])
            if predicted_h_a >= target_margin:
                continue

            required = (target_margin - float(predicted_base[index])) / maximum
            if required > 1.0 + 1e-12:
                physical_value = predicted_base[index] + maximum
                raise FunnelRelaxationInfeasibleError(
                    f"{FUNNEL_CHANNELS[index]} cannot maintain the sampled "
                    f"guard over the next sample: predicted physical margin "
                    f"{physical_value:.6g}."
                )
            repaired[index] = max(
                repaired[index],
                float(np.clip(required, 0.0, 1.0)),
            )

        return repaired, repaired - state

    def evaluate(
        self,
        state: FloatArray,
        *,
        conservative_values: FloatArray,
        required_slack: float,
        enabled: FloatArray,
        # Backward-compatible sampled-data arguments; the paper law does not use
        # h_dot or dt in the continuous right-hand side.
        conservative_rates: FloatArray | None = None,
        sample_time: float | None = None,
    ) -> FunnelRelaxationEvaluation:
        """Evaluate the continuous-time adaptive-domain law."""
        del conservative_rates, sample_time

        state = self.validate_state(state)
        base = _vector4(conservative_values, name="conservative_values")
        active = np.asarray(enabled, dtype=bool)
        if active.shape != (4,):
            raise ValueError("enabled must have shape (4,).")

        enlargement = self.maximum_enlargement * state
        adaptive = base + enlargement
        desired_adaptive = self.desired_conservative_values + enlargement
        gamma = self.infeasibility_activation(required_slack)

        selected = np.zeros(4, dtype=float)
        activation = np.zeros(4, dtype=float)
        derivative = np.zeros(4, dtype=float)
        recovery_rate = np.zeros(4, dtype=float)
        enlargement_rate = np.zeros(4, dtype=float)
        expansion_required = np.zeros(4, dtype=bool)

        h_on = self.activation_on_margin
        h_off = self.activation_off_margin
        floor = float(self.minimum_constraint_margin)

        for index in range(4):
            if not active[index]:
                continue

            h_a = float(adaptive[index])
            h_d_a = float(desired_adaptive[index])
            if h_a <= 0.0:
                raise FunnelRelaxationInfeasibleError(
                    f"{FUNNEL_CHANNELS[index]} adaptive barrier is outside its "
                    f"domain (h_a={h_a:.6g})."
                )

            sigma = self._smoothstep_activation(
                h_a,
                float(h_on[index]),
                float(h_off[index]),
            )
            activation[index] = sigma

            # D_{ell,i} = dV_i/ds_{ell,i}.  The floor only prevents floating-
            # point overflow arbitrarily close to the logarithmic singularity.
            h_safe = max(h_a, floor)
            difference = h_a - h_d_a
            derivative[index] = (
                -self.barrier_weights[index]
                * self.maximum_enlargement[index]
                * difference**2
                / (h_safe * h_d_a**2)
            )

            recovery_rate[index] = (
                -self.recovery_gain[index] * (1.0 - sigma) * state[index]
            )
            enlargement_rate[index] = (
                -self.barrier_gain[index]
                * gamma
                * sigma
                * derivative[index]
            )
            raw_rate = recovery_rate[index] + enlargement_rate[index]

            # Pi_{<=1}: outward motion is blocked at the physical-limit state.
            if state[index] >= 1.0 - 1e-12 and raw_rate > 0.0:
                raw_rate = 0.0
            selected[index] = raw_rate
            expansion_required[index] = enlargement_rate[index] > 1e-12

        return FunnelRelaxationEvaluation(
            state=state,
            enlargement=enlargement,
            selected_rate=selected,
            adaptive_constraint_values=adaptive,
            desired_adaptive_constraint_values=desired_adaptive,
            activation=activation,
            barrier_derivative=derivative,
            recovery_rate=recovery_rate,
            enlargement_rate=enlargement_rate,
            infeasibility_activation=gamma,
            expansion_required=expansion_required,
        )

    def advance(
        self,
        state: FloatArray,
        *,
        conservative_values: FloatArray,
        required_slack: float,
        enabled: FloatArray,
        sample_time: float,
    ) -> tuple[FloatArray, FloatArray]:
        """Advance the paper law with an implicit sampled-data step.

        For each enabled channel this solves

            s_{k+1} = s_k + dt f(s_{k+1}; h_{c,k}),

        with the sampled conservative constraint held fixed.  The continuous
        right-hand side ``f`` is exactly the one returned by :meth:`evaluate`;
        only its numerical integration is implicit.  This prevents the large
        explicit-Euler overshoot that can otherwise cross ``h_a=0`` in one
        sample near the logarithmic singularity.
        """
        if not np.isfinite(sample_time) or sample_time <= 0.0:
            raise ValueError("sample_time must be finite and positive.")

        state = self.validate_state(state)
        base = _vector4(conservative_values, name="conservative_values")
        active = np.asarray(enabled, dtype=bool)
        if active.shape != (4,):
            raise ValueError("enabled must have shape (4,).")

        gamma = self.infeasibility_activation(required_slack)
        next_state = state.copy()
        floor = float(self.minimum_constraint_margin)
        h_on = self.activation_on_margin
        h_off = self.activation_off_margin

        for index in range(4):
            if not active[index]:
                continue

            rho = float(self.maximum_enlargement[index])
            if rho <= 0.0:
                continue

            h_c = float(base[index])
            h_d_c = float(self.desired_conservative_values[index])
            mu = float(self.barrier_weights[index])
            k_s = float(self.recovery_gain[index])
            k_b = float(self.barrier_gain[index])
            old_state = float(state[index])

            # Smallest admissible state for the frozen sampled h_c.  If this
            # exceeds one, even the physical boundary cannot restore h_a>0.
            lower = max(0.0, (floor - h_c) / rho)
            if lower > 1.0 + 1e-12:
                physical_value = h_c + rho
                raise FunnelRelaxationInfeasibleError(
                    f"{FUNNEL_CHANNELS[index]} is outside the physical domain: "
                    f"h_c + rho_max = {physical_value:.6g}."
                )
            lower = min(lower, 1.0)

            def raw_rate(candidate: float) -> float:
                h_a = h_c + rho * candidate
                h_d_a = h_d_c + rho * candidate
                if h_a <= 0.0:
                    # The root search never intentionally enters this region;
                    # the floor merely makes the boundary value numerically
                    # evaluable for bracketing.
                    h_a = floor
                sigma = self._smoothstep_activation(
                    h_a,
                    float(h_on[index]),
                    float(h_off[index]),
                )
                difference = h_a - h_d_a
                derivative = (
                    -mu * rho * difference**2 / (max(h_a, floor) * h_d_a**2)
                )
                rate = (
                    -k_s * (1.0 - sigma) * candidate
                    -k_b * gamma * sigma * derivative
                )
                if candidate >= 1.0 - 1e-12 and rate > 0.0:
                    return 0.0
                return rate

            def equation(candidate: float) -> float:
                return candidate - old_state - sample_time * raw_rate(candidate)

            f_lower = equation(lower)
            f_upper = equation(1.0)

            # If the sampled plant motion has already moved the frozen h_c so
            # far that no implicit root remains below the admissible lower
            # bound, use the minimum admissible state.  This is a sampled-data
            # numerical safeguard, not an additional continuous-time term.
            if f_lower >= 0.0:
                next_state[index] = lower
                continue

            if f_upper <= 0.0:
                # At s=1 the projected continuous law cannot move farther
                # outward, so a nonpositive residual means the physical-limit
                # state is the only admissible sampled update.
                next_state[index] = 1.0
                continue

            lo, hi = lower, 1.0
            for _ in range(60):
                middle = 0.5 * (lo + hi)
                if equation(middle) <= 0.0:
                    lo = middle
                else:
                    hi = middle
            next_state[index] = 0.5 * (lo + hi)

        next_state = self.validate_state(next_state)
        effective_rate = (next_state - state) / sample_time
        return next_state, effective_rate

