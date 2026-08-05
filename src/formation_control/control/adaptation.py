"""Feasibility-driven enlargement of conservative admissible domains.

For each scalar constraint, the adaptive construction uses

    h_a = h_c + rho,
    0 <= rho <= rho_max,

where ``rho = 0`` recovers the conservative domain and ``rho = rho_max``
recovers exactly the physical domain.

The enlargement dynamics are driven by a nonnegative CLF-relaxation
signal.  For actuator-limited controllers this should preferably be the
minimum relaxation required by the actuator set, rather than the optimal soft
QP slack:

    sigma(delta_req) = max(delta_req - delta_threshold, 0),
    rho_dot = Pi_[0,rho_max](k_plus sigma - k_minus rho).

The state is kept external and this class implements ``ContinuousTimeModel``,
so the same generic simulation integrators used elsewhere in the package can
advance the adaptive state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from formation_control.constraints import DistanceDomain, FieldOfViewDomain
from formation_control.models.base import ContinuousTimeModel, FloatArray


@dataclass(frozen=True)
class AdaptiveDomainState:
    """Named enlargement variables for the four admissibility constraints."""

    collision: float = 0.0
    range: float = 0.0
    horizontal_fov: float = 0.0
    vertical_fov: float = 0.0

    def as_array(self) -> FloatArray:
        """Return ``[rho_delta, rho_Delta, rho_h, rho_v]``."""
        return np.array(
            [
                self.collision,
                self.range,
                self.horizontal_fov,
                self.vertical_fov,
            ],
            dtype=float,
        )

    @classmethod
    def from_array(cls, value: FloatArray) -> AdaptiveDomainState:
        """Construct the named state from a four-dimensional vector."""
        value = np.asarray(value, dtype=float)
        if value.shape != (4,):
            raise ValueError(f"value must have shape (4,), got {value.shape}.")
        if not np.all(np.isfinite(value)):
            raise ValueError("value must contain only finite entries.")

        return cls(
            collision=float(value[0]),
            range=float(value[1]),
            horizontal_fov=float(value[2]),
            vertical_fov=float(value[3]),
        )


@dataclass(frozen=True)
class AdaptiveEnlargementLaw:
    """Projected scalar enlargement law for one constraint."""

    maximum: float
    expansion_gain: float
    recovery_gain: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.maximum) or self.maximum < 0.0:
            raise ValueError("maximum must be finite and nonnegative.")
        if not np.isfinite(self.expansion_gain) or self.expansion_gain < 0.0:
            raise ValueError("expansion_gain must be finite and nonnegative.")
        if not np.isfinite(self.recovery_gain) or self.recovery_gain < 0.0:
            raise ValueError("recovery_gain must be finite and nonnegative.")

    def rate(self, enlargement: float, activation: float) -> float:
        """Return the projected enlargement rate."""
        enlargement = float(enlargement)
        activation = float(activation)

        if not np.isfinite(enlargement):
            raise ValueError("enlargement must be finite.")
        if not np.isfinite(activation) or activation < 0.0:
            raise ValueError("activation must be finite and nonnegative.")
        if self.maximum == 0.0:
            return 0.0

        bounded_enlargement = float(np.clip(enlargement, 0.0, self.maximum))
        raw_rate = self.expansion_gain * activation - self.recovery_gain * bounded_enlargement

        if enlargement <= 0.0 and raw_rate < 0.0:
            return 0.0
        if enlargement >= self.maximum and raw_rate > 0.0:
            return 0.0

        return float(raw_rate)

    def project(self, enlargement: float) -> float:
        """Project an enlargement onto ``[0, maximum]``."""
        enlargement = float(enlargement)
        if not np.isfinite(enlargement):
            raise ValueError("enlargement must be finite.")
        return float(np.clip(enlargement, 0.0, self.maximum))


@dataclass(frozen=True)
class AdaptiveDomainDynamics(ContinuousTimeModel):
    """Four-channel domain-enlargement dynamics driven by one CLF-relaxation signal."""

    collision: AdaptiveEnlargementLaw
    range: AdaptiveEnlargementLaw
    horizontal_fov: AdaptiveEnlargementLaw
    vertical_fov: AdaptiveEnlargementLaw
    slack_threshold: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.slack_threshold) or self.slack_threshold < 0.0:
            raise ValueError("slack_threshold must be finite and nonnegative.")

    @classmethod
    def from_domains(
        cls,
        distance_domain: DistanceDomain,
        fov_domain: FieldOfViewDomain,
        *,
        expansion_gain: float,
        recovery_gain: float,
        slack_threshold: float = 0.0,
    ) -> AdaptiveDomainDynamics:
        """Create equal-gain adaptation laws from conservative/physical domains."""
        return cls(
            collision=AdaptiveEnlargementLaw(
                maximum=distance_domain.collision_enlargement_max,
                expansion_gain=expansion_gain,
                recovery_gain=recovery_gain,
            ),
            range=AdaptiveEnlargementLaw(
                maximum=distance_domain.range_enlargement_max,
                expansion_gain=expansion_gain,
                recovery_gain=recovery_gain,
            ),
            horizontal_fov=AdaptiveEnlargementLaw(
                maximum=fov_domain.horizontal_enlargement_max,
                expansion_gain=expansion_gain,
                recovery_gain=recovery_gain,
            ),
            vertical_fov=AdaptiveEnlargementLaw(
                maximum=fov_domain.vertical_enlargement_max,
                expansion_gain=expansion_gain,
                recovery_gain=recovery_gain,
            ),
            slack_threshold=slack_threshold,
        )

    @property
    def state_dim(self) -> int:
        return 4

    @property
    def input_dim(self) -> int:
        return 1

    @property
    def maximum_state(self) -> AdaptiveDomainState:
        """Return the physical-domain enlargement limits."""
        return AdaptiveDomainState(
            collision=self.collision.maximum,
            range=self.range.maximum,
            horizontal_fov=self.horizontal_fov.maximum,
            vertical_fov=self.vertical_fov.maximum,
        )

    def initialize(self) -> FloatArray:
        """Initialize at the conservative domain ``rho = 0``."""
        return np.zeros(4)

    def activation(self, slack: float) -> float:
        """Return ``max(slack - slack_threshold, 0)``."""
        slack = float(slack)
        if not np.isfinite(slack) or slack < 0.0:
            raise ValueError("slack must be finite and nonnegative.")
        return max(slack - self.slack_threshold, 0.0)

    def split_state(self, state: FloatArray) -> AdaptiveDomainState:
        """Return a named view of the enlargement state."""
        return AdaptiveDomainState.from_array(self.validate_state(state))

    def dynamics(self, state: FloatArray, control: FloatArray) -> FloatArray:
        state = self.split_state(state)
        control = self.validate_control(control)
        slack = float(control[0])
        activation = self.activation(slack)

        return np.array(
            [
                self.collision.rate(state.collision, activation),
                self.range.rate(state.range, activation),
                self.horizontal_fov.rate(state.horizontal_fov, activation),
                self.vertical_fov.rate(state.vertical_fov, activation),
            ],
            dtype=float,
        )

    def project_state(self, state: FloatArray) -> FloatArray:
        """Project all enlargement states onto their physical limits."""
        state = AdaptiveDomainState.from_array(self.validate_state(state))
        return np.array(
            [
                self.collision.project(state.collision),
                self.range.project(state.range),
                self.horizontal_fov.project(state.horizontal_fov),
                self.vertical_fov.project(state.vertical_fov),
            ],
            dtype=float,
        )
