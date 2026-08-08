"""Tests for the pre-ROS robustness perturbations."""

import numpy as np

from formation_control.models import BlueROV2Model
from formation_control.simulation.realism import (
    RealismConfig,
    build_perturbed_plant_model,
    delay_steps,
    delayed_noisy_parent_position,
)


def test_delay_steps():
    assert delay_steps(0.04, 0.02) == 2
    assert delay_steps(0.06, 0.02) == 3


def test_delayed_relative_position_is_reconstructed():
    measured_parent = delayed_noisy_parent_position(
        np.array([10.0, 0.0, 0.0]),
        np.array([1.0, 2.0, 3.0]),
        np.array([3.0, 1.0, 4.0]),
        rng=np.random.default_rng(0),
        noise_std=0.0,
    )
    np.testing.assert_allclose(
        measured_parent - np.array([10.0, 0.0, 0.0]),
        np.array([2.0, -1.0, 1.0]),
    )


def test_perturbed_plant_is_reproducible():
    nominal = BlueROV2Model()
    config = RealismConfig()
    plant_a = build_perturbed_plant_model(
        nominal,
        config=config,
        rng=np.random.default_rng(4),
    )
    plant_b = build_perturbed_plant_model(
        nominal,
        config=config,
        rng=np.random.default_rng(4),
    )
    np.testing.assert_allclose(
        plant_a.mass_matrix,
        plant_b.mass_matrix,
    )
    np.testing.assert_allclose(
        plant_a.current_velocity_inertial,
        np.array([0.05, -0.02, 0.0]),
    )
