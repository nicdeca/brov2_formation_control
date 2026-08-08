"""Simulation-only robustness perturbations."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from formation_control.models import BlueROV2Model, BlueROV2Parameters
from formation_control.models.base import FloatArray


@dataclass(frozen=True, slots=True)
class RealismConfig:
    """Moderate perturbations for pre-ROS robustness validation."""

    parameter_variation: float = 0.10
    water_current_inertial: tuple[float, float, float] = (0.05, -0.02, 0.0)
    relative_position_noise_std: float = 0.015
    linear_velocity_noise_std: float = 0.01
    angular_velocity_noise_std: float = 0.005
    relative_measurement_delay: float = 0.04

    def __post_init__(self) -> None:
        if not 0.0 <= self.parameter_variation < 1.0:
            raise ValueError("parameter_variation must lie in [0, 1).")
        for name in (
            "relative_position_noise_std",
            "linear_velocity_noise_std",
            "angular_velocity_noise_std",
            "relative_measurement_delay",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        current = np.asarray(self.water_current_inertial, dtype=float)
        if current.shape != (3,) or not np.all(np.isfinite(current)):
            raise ValueError("water_current_inertial must contain three finite values.")


def _scale(
    value: float,
    rng: np.random.Generator,
    variation: float,
) -> float:
    return float(value * rng.uniform(1.0 - variation, 1.0 + variation))


def perturb_parameters(
    nominal: BlueROV2Parameters,
    *,
    rng: np.random.Generator,
    variation: float,
) -> BlueROV2Parameters:
    """Perturb inertia, added mass, and hydrodynamic damping."""
    if variation == 0.0:
        return nominal

    volume_variation = min(0.25 * variation, 0.03)
    return replace(
        nominal,
        mass=_scale(nominal.mass, rng, variation),
        volume=_scale(nominal.volume, rng, volume_variation),
        inertia_x=_scale(nominal.inertia_x, rng, variation),
        inertia_y=_scale(nominal.inertia_y, rng, variation),
        inertia_z=_scale(nominal.inertia_z, rng, variation),
        added_mass_u=_scale(nominal.added_mass_u, rng, variation),
        added_mass_v=_scale(nominal.added_mass_v, rng, variation),
        added_mass_w=_scale(nominal.added_mass_w, rng, variation),
        added_mass_p=_scale(nominal.added_mass_p, rng, variation),
        added_mass_q=_scale(nominal.added_mass_q, rng, variation),
        added_mass_r=_scale(nominal.added_mass_r, rng, variation),
        linear_damping_u=_scale(nominal.linear_damping_u, rng, variation),
        linear_damping_v=_scale(nominal.linear_damping_v, rng, variation),
        linear_damping_w=_scale(nominal.linear_damping_w, rng, variation),
        linear_damping_p=_scale(nominal.linear_damping_p, rng, variation),
        linear_damping_q=_scale(nominal.linear_damping_q, rng, variation),
        linear_damping_r=_scale(nominal.linear_damping_r, rng, variation),
        quadratic_damping_u=_scale(nominal.quadratic_damping_u, rng, variation),
        quadratic_damping_v=_scale(nominal.quadratic_damping_v, rng, variation),
        quadratic_damping_w=_scale(nominal.quadratic_damping_w, rng, variation),
        quadratic_damping_p=_scale(nominal.quadratic_damping_p, rng, variation),
        quadratic_damping_q=_scale(nominal.quadratic_damping_q, rng, variation),
        quadratic_damping_r=_scale(nominal.quadratic_damping_r, rng, variation),
    )


def build_perturbed_plant_model(
    nominal_model: BlueROV2Model,
    *,
    config: RealismConfig,
    rng: np.random.Generator,
) -> BlueROV2Model:
    """Build a plant unknown to the nominal controller."""
    return BlueROV2Model(
        perturb_parameters(
            nominal_model.parameters,
            rng=rng,
            variation=config.parameter_variation,
        ),
        current_velocity_inertial=np.asarray(
            config.water_current_inertial,
            dtype=float,
        ),
    )


def noisy_twist_state(
    true_state: FloatArray,
    *,
    rng: np.random.Generator,
    linear_std: float,
    angular_std: float,
) -> FloatArray:
    measured = np.asarray(true_state, dtype=float).copy()
    measured[7:10] += rng.normal(0.0, linear_std, size=3)
    measured[10:13] += rng.normal(0.0, angular_std, size=3)
    return measured


def delayed_noisy_parent_position(
    current_observer_position: FloatArray,
    delayed_observer_position: FloatArray,
    delayed_parent_position: FloatArray,
    *,
    rng: np.random.Generator,
    noise_std: float,
) -> FloatArray:
    """Encode a delayed noisy relative-position measurement."""
    delayed_relative = np.asarray(delayed_parent_position, dtype=float) - np.asarray(
        delayed_observer_position, dtype=float
    )
    noise = rng.normal(0.0, noise_std, size=3)
    return np.asarray(current_observer_position, dtype=float) + delayed_relative + noise


def delay_steps(delay: float, dt: float) -> int:
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    return max(0, int(round(delay / dt)))
