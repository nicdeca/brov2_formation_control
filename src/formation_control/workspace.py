"""Axis-aligned workspace barriers and bounded conservative-domain relaxation.

The workspace uses six one-sided constraints ordered as

    (x_min, x_max, y_min, y_max, z_min, z_max).

For each face, the conservative constraint ``h_c > 0`` may be enlarged by
``rho = rho_max * s``, with normalized state ``s in [0, 1]``.  At ``s = 1``
the corresponding boundary coincides with the physical safe robot-center
boundary.  The physical safe bounds are themselves inset from the actual tank
surfaces and are never relaxed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from formation_control.potentials import RecenteredLogBarrier


WORKSPACE_CHANNELS = (
    "x_min",
    "x_max",
    "y_min",
    "y_max",
    "z_min",
    "z_max",
)


def _vector3(value, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.shape != (3,):
        raise ValueError(f"{name} must contain three values.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _vector6(value, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.shape != (6,):
        raise ValueError(f"{name} must contain six values.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


@dataclass(frozen=True, slots=True)
class AxisAlignedWorkspaceDomain:
    """Nested conservative and physical safe robot-center boxes."""

    physical_lower: np.ndarray
    physical_upper: np.ndarray
    conservative_lower: np.ndarray
    conservative_upper: np.ndarray

    def __post_init__(self) -> None:
        physical_lower = _vector3(self.physical_lower, name="physical_lower")
        physical_upper = _vector3(self.physical_upper, name="physical_upper")
        conservative_lower = _vector3(
            self.conservative_lower,
            name="conservative_lower",
        )
        conservative_upper = _vector3(
            self.conservative_upper,
            name="conservative_upper",
        )

        if np.any(physical_lower >= conservative_lower):
            raise ValueError(
                "physical_lower must lie strictly outside conservative_lower."
            )
        if np.any(conservative_lower >= conservative_upper):
            raise ValueError(
                "conservative_lower must lie strictly below conservative_upper."
            )
        if np.any(conservative_upper >= physical_upper):
            raise ValueError(
                "conservative_upper must lie strictly inside physical_upper."
            )

        object.__setattr__(self, "physical_lower", physical_lower.copy())
        object.__setattr__(self, "physical_upper", physical_upper.copy())
        object.__setattr__(
            self,
            "conservative_lower",
            conservative_lower.copy(),
        )
        object.__setattr__(
            self,
            "conservative_upper",
            conservative_upper.copy(),
        )

    @property
    def maximum_enlargement(self) -> np.ndarray:
        """Face-wise conservative-to-physical reserve in metres."""
        lower = self.conservative_lower - self.physical_lower
        upper = self.physical_upper - self.conservative_upper
        return np.array(
            [
                lower[0],
                upper[0],
                lower[1],
                upper[1],
                lower[2],
                upper[2],
            ],
            dtype=float,
        )

    @property
    def constraint_gradients(self) -> np.ndarray:
        """Rows ``grad h_k`` for the six conservative constraints."""
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=float,
        )

    def conservative_values(self, position) -> np.ndarray:
        position = _vector3(position, name="position")
        lower = position - self.conservative_lower
        upper = self.conservative_upper - position
        return np.array(
            [lower[0], upper[0], lower[1], upper[1], lower[2], upper[2]],
            dtype=float,
        )

    def physical_values(self, position) -> np.ndarray:
        position = _vector3(position, name="position")
        lower = position - self.physical_lower
        upper = self.physical_upper - position
        return np.array(
            [lower[0], upper[0], lower[1], upper[1], lower[2], upper[2]],
            dtype=float,
        )

    def conservative_rates(self, inertial_velocity) -> np.ndarray:
        velocity = _vector3(inertial_velocity, name="inertial_velocity")
        return self.constraint_gradients @ velocity

    def adaptive_bounds(self, relaxation_state) -> tuple[np.ndarray, np.ndarray]:
        state = _vector6(relaxation_state, name="relaxation_state")
        if np.any(state < -1e-9) or np.any(state > 1.0 + 1e-9):
            raise ValueError("workspace relaxation state must lie in [0, 1].")
        state = np.clip(state, 0.0, 1.0)
        rho = self.maximum_enlargement * state
        lower = self.conservative_lower - rho[[0, 2, 4]]
        upper = self.conservative_upper + rho[[1, 3, 5]]
        return lower, upper

    def project_reference(
        self,
        position,
        relaxation_state,
        *,
        margin: float,
    ) -> np.ndarray:
        """Project a task reference into the current adaptive workspace."""
        if not np.isfinite(margin) or margin < 0.0:
            raise ValueError("reference margin must be finite and nonnegative.")
        lower, upper = self.adaptive_bounds(relaxation_state)
        if np.any(lower + margin >= upper - margin):
            raise ValueError("reference margin leaves no admissible workspace.")
        return np.clip(
            _vector3(position, name="reference_position"),
            lower + margin,
            upper - margin,
        )


@dataclass(frozen=True, slots=True)
class WorkspaceBarrierEvaluation:
    value: float
    position_gradient: np.ndarray
    conservative_values: np.ndarray
    physical_values: np.ndarray
    adaptive_values: np.ndarray
    reference_position: np.ndarray

    @property
    def minimum_physical_margin(self) -> float:
        return float(np.min(self.physical_values))


@dataclass(frozen=True, slots=True)
class WorkspaceBarrierPotential:
    """Sum of six recentered log barriers for an axis-aligned workspace."""

    domain: AxisAlignedWorkspaceDomain
    weight: float = 0.10
    reference_margin: float = 0.05

    def __post_init__(self) -> None:
        if not np.isfinite(self.weight) or self.weight < 0.0:
            raise ValueError("workspace barrier weight must be finite and nonnegative.")
        if not np.isfinite(self.reference_margin) or self.reference_margin <= 0.0:
            raise ValueError("workspace reference_margin must be finite and positive.")

    def evaluate(
        self,
        position,
        reference_position,
        relaxation_state,
    ) -> WorkspaceBarrierEvaluation:
        position = _vector3(position, name="position")
        state = _vector6(relaxation_state, name="relaxation_state")
        rho = self.domain.maximum_enlargement * np.clip(state, 0.0, 1.0)

        reference = self.domain.project_reference(
            reference_position,
            state,
            margin=self.reference_margin,
        )
        conservative = self.domain.conservative_values(position)
        physical = self.domain.physical_values(position)
        adaptive = conservative + rho
        reference_adaptive = self.domain.conservative_values(reference) + rho

        if np.any(adaptive <= 0.0):
            channel = WORKSPACE_CHANNELS[int(np.argmin(adaptive))]
            raise ValueError(
                "robot lies outside the current adaptive workspace on "
                f"{channel}: h={float(np.min(adaptive)):.6g}."
            )

        multipliers = np.zeros(6, dtype=float)
        total_value = 0.0
        for index in range(6):
            barrier = RecenteredLogBarrier(float(reference_adaptive[index]))
            total_value += self.weight * barrier.value(float(adaptive[index]))
            multipliers[index] = (
                self.weight * barrier.derivative(float(adaptive[index]))
            )

        gradient = self.domain.constraint_gradients.T @ multipliers
        return WorkspaceBarrierEvaluation(
            value=float(total_value),
            position_gradient=np.asarray(gradient, dtype=float),
            conservative_values=conservative,
            physical_values=physical,
            adaptive_values=adaptive,
            reference_position=reference,
        )


class WorkspaceRelaxationInfeasibleError(RuntimeError):
    """Raised when the physical workspace reserve cannot preserve the domain."""


@dataclass(frozen=True, slots=True)
class WorkspaceRelaxationEvaluation:
    state: np.ndarray
    enlargement: np.ndarray
    selected_rate: np.ndarray
    adaptive_values: np.ndarray


@dataclass(frozen=True, slots=True)
class WorkspaceRelaxationPolicy:
    """Six-channel sampled-data conservative-workspace relaxation policy.

    Positive expansion is selected only when required to preserve a positive
    adaptive-domain margin over the next control interval.  Otherwise the
    normalized state exponentially recovers toward the conservative box.
    """

    maximum_enlargement: np.ndarray
    recovery_gain: float = 0.8
    domain_margin_ratio: float = 0.10
    minimum_constraint_margin: float = 1e-3

    def __post_init__(self) -> None:
        maximum = _vector6(
            self.maximum_enlargement,
            name="maximum_enlargement",
        )
        if np.any(maximum <= 0.0):
            raise ValueError("workspace maximum_enlargement must be positive.")
        if not np.isfinite(self.recovery_gain) or self.recovery_gain < 0.0:
            raise ValueError("workspace recovery_gain must be finite and nonnegative.")
        if (
            not np.isfinite(self.domain_margin_ratio)
            or self.domain_margin_ratio < 0.0
        ):
            raise ValueError(
                "workspace domain_margin_ratio must be finite and nonnegative."
            )
        if (
            not np.isfinite(self.minimum_constraint_margin)
            or self.minimum_constraint_margin <= 0.0
        ):
            raise ValueError(
                "workspace minimum_constraint_margin must be finite and positive."
            )
        object.__setattr__(self, "maximum_enlargement", maximum.copy())

    @property
    def domain_margin(self) -> np.ndarray:
        return np.maximum(
            self.minimum_constraint_margin,
            self.domain_margin_ratio * self.maximum_enlargement,
        )

    def validate_state(self, state) -> np.ndarray:
        state = _vector6(state, name="workspace_relaxation_state")
        if np.any(state < -1e-9) or np.any(state > 1.0 + 1e-9):
            raise ValueError("workspace relaxation state must lie in [0, 1].")
        return np.clip(state, 0.0, 1.0)

    def enlargement(self, state) -> np.ndarray:
        return self.maximum_enlargement * self.validate_state(state)

    def project_to_current_domain(
        self,
        state,
        conservative_values,
    ) -> tuple[np.ndarray, bool]:
        state = self.validate_state(state)
        values = _vector6(conservative_values, name="conservative_values")
        required_rho = np.maximum(self.domain_margin - values, 0.0)
        if np.any(required_rho > self.maximum_enlargement + 1e-10):
            index = int(np.argmax(required_rho - self.maximum_enlargement))
            raise WorkspaceRelaxationInfeasibleError(
                "physical workspace reserve exhausted on "
                f"{WORKSPACE_CHANNELS[index]}."
            )
        required_state = required_rho / self.maximum_enlargement
        projected = np.maximum(state, required_state)
        projected = np.clip(projected, 0.0, 1.0)
        changed = not np.allclose(projected, state)
        return projected, changed

    def evaluate(
        self,
        state,
        *,
        conservative_values,
        conservative_rates,
        sample_time: float,
    ) -> WorkspaceRelaxationEvaluation:
        if not np.isfinite(sample_time) or sample_time <= 0.0:
            raise ValueError("sample_time must be finite and positive.")
        state = self.validate_state(state)
        values = _vector6(conservative_values, name="conservative_values")
        rates = _vector6(conservative_rates, name="conservative_rates")
        maximum = self.maximum_enlargement
        adaptive = values + maximum * state

        reference_rate = -self.recovery_gain * state
        lower_domain = (
            self.domain_margin
            - adaptive
            - sample_time * rates
        ) / (sample_time * maximum)
        lower_state = -state / sample_time
        upper_state = (1.0 - state) / sample_time
        lower = np.maximum(lower_domain, lower_state)

        if np.any(lower > upper_state + 1e-10):
            index = int(np.argmax(lower - upper_state))
            raise WorkspaceRelaxationInfeasibleError(
                "physical workspace reserve cannot preserve the sampled-data "
                f"domain on {WORKSPACE_CHANNELS[index]}."
            )

        selected = np.minimum(np.maximum(reference_rate, lower), upper_state)
        return WorkspaceRelaxationEvaluation(
            state=state.copy(),
            enlargement=(maximum * state).copy(),
            selected_rate=np.asarray(selected, dtype=float),
            adaptive_values=np.asarray(adaptive, dtype=float),
        )
