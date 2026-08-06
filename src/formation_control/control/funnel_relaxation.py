"""Independent normalized relaxation of conservative sensing funnels.

Each conservative scalar constraint is enlarged according to

    h_a(x, s) = h_c(x) + rho_max s,
    0 <= s <= 1,

where ``s = 0`` is the conservative domain and ``s = 1`` is the full physical
admissible domain. The artificial dynamics are

    s_dot = v.

The auxiliary input ``v`` is intentionally not a CLF-QP input. The physical
CLF-QP treats ``s`` as a frozen scheduling parameter and optimizes only the
physical actuator input.

The funnel controller has only two objectives:

1. recover the conservative domain with ``v_ref = -K_s s``;
2. keep every logarithmic barrier well inside its domain by preserving a
   small positive channel-wise margin ``h_a >= h_margin``.

With zero-order hold over one controller sample,

    s^+ = s + dt v,

and with the first-order prediction

    h_a^+ = h_a + dt (h_c_dot + rho_max v),

the auxiliary controller imposes

    0 <= s^+ <= 1,
    h_a^+ >= h_margin,

where

    h_margin = max(epsilon_abs, margin_ratio * rho_max).

Thus the practical margin scales naturally with the available
conservative-to-physical reserve in each constraint coordinate. The much
smaller ``epsilon_abs`` is retained only as an emergency safeguard against
evaluating the logarithm at or beyond its singularity.

The selected rate is the projection of ``v_ref`` onto these admissible bounds.
Consequently, positive funnel expansion is used only when it is required to
keep the adaptive barrier domain well defined; otherwise the funnel contracts
toward ``s = 0``.
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
    """Raised when the physical relaxation reserve cannot preserve the domain."""


@dataclass(frozen=True)
class FunnelRelaxationEvaluation:
    """One evaluation of the independent auxiliary funnel controller."""

    state: FloatArray
    enlargement: FloatArray
    reference_rate: FloatArray
    lower_rate: FloatArray
    upper_rate: FloatArray
    selected_rate: FloatArray
    domain_lower_rate: FloatArray
    adaptive_constraint_values: FloatArray
    expansion_required: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "state",
            "enlargement",
            "reference_rate",
            "lower_rate",
            "upper_rate",
            "selected_rate",
            "domain_lower_rate",
            "adaptive_constraint_values",
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
    """Four-channel domain-preserving funnel controller."""

    maximum_enlargement: FloatArray
    recovery_gain: FloatArray | float = 0.8
    domain_margin_ratio: float = 0.1
    minimum_constraint_margin: float = 1e-5

    def __post_init__(self) -> None:
        maximum = _vector4(
            self.maximum_enlargement,
            name="maximum_enlargement",
        )
        if np.any(maximum < 0.0):
            raise ValueError("maximum_enlargement must be nonnegative.")

        recovery_gain = _vector4(
            self.recovery_gain,
            name="recovery_gain",
        )
        if np.any(recovery_gain < 0.0):
            raise ValueError("recovery_gain must be nonnegative.")

        if not np.isfinite(self.domain_margin_ratio) or (self.domain_margin_ratio < 0.0):
            raise ValueError("domain_margin_ratio must be finite and nonnegative.")

        if not np.isfinite(self.minimum_constraint_margin) or (
            self.minimum_constraint_margin <= 0.0
        ):
            raise ValueError("minimum_constraint_margin must be finite and positive.")

        object.__setattr__(self, "maximum_enlargement", maximum)
        object.__setattr__(self, "recovery_gain", recovery_gain)

    @property
    def domain_margin(self) -> FloatArray:
        """Return the practical channel-wise adaptive-domain margin.

        The margin is expressed in the same scalar ``h`` coordinate used by
        each logarithmic barrier. Scaling it by ``maximum_enlargement`` makes
        the margin dimensionally consistent and comparable across collision,
        range, and FoV channels.
        """
        return np.maximum(
            self.minimum_constraint_margin,
            self.domain_margin_ratio * self.maximum_enlargement,
        )

    def initialize(self) -> FloatArray:
        """Return the conservative funnel state ``s = 0``."""
        return np.zeros(4)

    def validate_state(self, state: FloatArray) -> FloatArray:
        state = _vector4(state, name="state")
        if np.any(state < -1e-9) or np.any(state > 1.0 + 1e-9):
            raise ValueError("normalized funnel state must lie in [0, 1].")
        return np.clip(state, 0.0, 1.0)

    def enlargement(self, state: FloatArray) -> FloatArray:
        """Return ``rho = rho_max * s``."""
        return self.maximum_enlargement * self.validate_state(state)

    def reference_rate(self, state: FloatArray) -> FloatArray:
        """Return the nominal recovery law ``v_ref = -K_s s``."""
        state = self.validate_state(state)
        return -self.recovery_gain * state

    def project_to_current_domain(
        self,
        state: FloatArray,
        conservative_values: FloatArray,
        *,
        enabled: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        """Repair only an actual near-singularity of the logarithmic domain.

        The sampled auxiliary controller targets the larger practical
        ``domain_margin``. This safeguard is deliberately less intrusive: it
        activates only if the realized adaptive constraint reaches the tiny
        absolute floor ``minimum_constraint_margin``. If activated, it projects
        the state back to the practical margin so that the following logarithmic
        evaluation is also numerically well conditioned.

        Under the nominal sampled-data controller this should remain inactive.
        """
        state = self.validate_state(state)
        base = _vector4(conservative_values, name="conservative_values")
        active = np.asarray(enabled, dtype=bool)
        if active.shape != (4,):
            raise ValueError("enabled must have shape (4,).")

        projected = state.copy()
        correction = np.zeros(4)

        for index in range(4):
            if not active[index]:
                continue

            maximum = self.maximum_enlargement[index]
            absolute_floor = self.minimum_constraint_margin
            practical_margin = self.domain_margin[index]
            physical_value = base[index] + maximum
            adaptive_value = base[index] + maximum * projected[index]

            if physical_value <= absolute_floor:
                raise FunnelRelaxationInfeasibleError(
                    f"physical {FUNNEL_CHANNELS[index]} constraint exhausted: "
                    f"h_physical={physical_value:.6g}, "
                    f"h_floor={absolute_floor:.6g}."
                )

            if maximum <= 0.0:
                if adaptive_value <= absolute_floor:
                    raise FunnelRelaxationInfeasibleError(
                        f"{FUNNEL_CHANNELS[index]} has no relaxation reserve."
                    )
                continue

            # Do not trigger the safeguard merely because the practical CBF
            # margin was undershot slightly. The auxiliary controller will
            # recover that margin at the next sample. Intervene only when the
            # logarithmic barrier itself is close to becoming undefined.
            if adaptive_value > absolute_floor:
                continue

            if physical_value < practical_margin:
                raise FunnelRelaxationInfeasibleError(
                    f"no reserve to restore the practical "
                    f"{FUNNEL_CHANNELS[index]} domain margin: "
                    f"h_physical={physical_value:.6g}, "
                    f"h_margin={practical_margin:.6g}."
                )

            required_state = (practical_margin - base[index]) / maximum
            required_state = float(np.clip(required_state, 0.0, 1.0))

            correction[index] = max(
                0.0,
                required_state - projected[index],
            )
            projected[index] = max(projected[index], required_state)

        return projected, correction

    def evaluate(
        self,
        state: FloatArray,
        *,
        conservative_values: FloatArray,
        conservative_rates: FloatArray,
        enabled: FloatArray,
        sample_time: float,
    ) -> FunnelRelaxationEvaluation:
        """Return the domain-preserving recovery rate.

        The physical CLF-QP is not involved in this calculation. For each
        enabled channel the auxiliary controller solves

            minimize_v  1/2 (v - v_ref)^2

        subject to

            0 <= s + dt v <= 1,

            h_a + dt (h_c_dot + rho_max v) >= h_margin.

        The solution is the projection

            v* = clip(v_ref, v_lower, v_upper).
        """
        state = self.validate_state(state)
        base = _vector4(conservative_values, name="conservative_values")
        base_rate = _vector4(conservative_rates, name="conservative_rates")
        active = np.asarray(enabled, dtype=bool)
        if active.shape != (4,):
            raise ValueError("enabled must have shape (4,).")
        if not np.isfinite(sample_time) or sample_time <= 0.0:
            raise ValueError("sample_time must be finite and strictly positive.")

        rho = self.maximum_enlargement * state
        adaptive_value = base + rho
        reference_rate = self.reference_rate(state)

        lower = np.zeros(4)
        upper = np.zeros(4)
        selected = np.zeros(4)
        domain_lower = np.zeros(4)
        expansion_required = np.zeros(4, dtype=bool)

        for index in range(4):
            if not active[index]:
                reference_rate[index] = 0.0
                continue

            maximum = self.maximum_enlargement[index]
            practical_margin = self.domain_margin[index]
            physical_value = base[index] + maximum

            if physical_value < practical_margin:
                raise FunnelRelaxationInfeasibleError(
                    f"no reserve to maintain the practical "
                    f"{FUNNEL_CHANNELS[index]} domain margin: "
                    f"h_physical={physical_value:.6g}, "
                    f"h_margin={practical_margin:.6g}."
                )

            if maximum <= 0.0:
                if base[index] <= self.minimum_constraint_margin:
                    raise FunnelRelaxationInfeasibleError(
                        f"{FUNNEL_CHANNELS[index]} has no relaxation reserve."
                    )
                reference_rate[index] = 0.0
                continue

            state_lower = -state[index] / sample_time
            state_upper = (1.0 - state[index]) / sample_time

            domain_lower[index] = (
                practical_margin - adaptive_value[index] - sample_time * base_rate[index]
            ) / (sample_time * maximum)

            lower[index] = max(state_lower, domain_lower[index])
            upper[index] = state_upper

            if lower[index] > upper[index] + 1e-10:
                raise FunnelRelaxationInfeasibleError(
                    f"no admissible rate for {FUNNEL_CHANNELS[index]}: "
                    f"s={state[index]:.6g}, h_a={adaptive_value[index]:.6g}, "
                    f"h_c_dot={base_rate[index]:.6g}, "
                    f"h_margin={practical_margin:.6g}, "
                    f"predicted one-step domain requires "
                    f"v_lower={lower[index]:.6g}, "
                    f"v_upper={upper[index]:.6g}."
                )

            selected[index] = float(
                np.clip(
                    reference_rate[index],
                    lower[index],
                    upper[index],
                )
            )
            expansion_required[index] = selected[index] > 1e-12

        return FunnelRelaxationEvaluation(
            state=state,
            enlargement=rho,
            reference_rate=reference_rate,
            lower_rate=lower,
            upper_rate=upper,
            selected_rate=selected,
            domain_lower_rate=domain_lower,
            adaptive_constraint_values=adaptive_value,
            expansion_required=expansion_required,
        )
