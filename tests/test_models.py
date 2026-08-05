import numpy as np
import pytest

from formation_control.models import (
    BlueROV2Model,
    DoubleIntegratorModel,
    SingleIntegratorModel,
)


def test_single_integrator_dynamics() -> None:
    model = SingleIntegratorModel(dimension=3)
    state = np.array([1.0, 2.0, 3.0])
    control = np.array([0.2, -0.1, 0.5])

    np.testing.assert_allclose(model.dynamics(state, control), control)


def test_double_integrator_dynamics() -> None:
    model = DoubleIntegratorModel(dimension=3)
    state = np.array([1.0, 2.0, 3.0, 0.2, -0.1, 0.5])
    acceleration = np.array([1.0, 2.0, 3.0])

    expected = np.array([0.2, -0.1, 0.5, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(model.dynamics(state, acceleration), expected)


def test_model_dimension_validation() -> None:
    with pytest.raises(ValueError):
        SingleIntegratorModel(dimension=0)

    with pytest.raises(ValueError):
        DoubleIntegratorModel(dimension=-1)


def test_bluerov2_mass_matrix_is_block_diagonal_and_positive_definite() -> None:
    model = BlueROV2Model()
    mass = model.mass_matrix

    np.testing.assert_allclose(mass[:3, 3:], 0.0)
    np.testing.assert_allclose(mass[3:, :3], 0.0)
    assert np.all(np.linalg.eigvalsh(mass) > 0.0)


def test_bluerov2_coriolis_matrix_is_skew_symmetric() -> None:
    model = BlueROV2Model()
    velocity = np.array([0.4, -0.2, 0.1, 0.03, -0.04, 0.05])
    coriolis = model.coriolis_matrix(velocity)

    np.testing.assert_allclose(coriolis + coriolis.T, 0.0, atol=1e-12)


def test_bluerov2_identity_attitude_kinematics() -> None:
    model = BlueROV2Model()
    state = np.zeros(13)
    state[3] = 1.0  # scalar-first identity quaternion
    state[7:10] = np.array([0.4, -0.2, 0.1])
    state[10:13] = np.array([0.02, -0.04, 0.06])

    derivative = model.dynamics(state, np.zeros(6))

    np.testing.assert_allclose(derivative[:3], state[7:10])
    np.testing.assert_allclose(
        derivative[3:7],
        np.array([0.0, 0.01, -0.02, 0.03]),
        atol=1e-12,
    )


def test_bluerov2_projection_normalizes_quaternion() -> None:
    model = BlueROV2Model()
    state = np.zeros(13)
    state[3:7] = np.array([2.0, 0.0, 0.0, 0.0])

    projected = model.project_state(state)
    np.testing.assert_allclose(projected[3:7], np.array([1.0, 0.0, 0.0, 0.0]))


def test_bluerov2_translational_and_rotational_partitions_match_full_drift() -> None:
    model = BlueROV2Model()
    state = np.zeros(13)
    state[3] = 1.0
    state[7:] = np.array([0.2, -0.1, 0.15, 0.02, -0.03, 0.04])

    drift = model.drift_wrench(state)
    mass_v, drift_v = model.translational_dynamics_terms(state)
    mass_w, drift_w = model.rotational_dynamics_terms(state)

    np.testing.assert_allclose(mass_v, model.mass_matrix[:3, :3])
    np.testing.assert_allclose(mass_w, model.mass_matrix[3:, 3:])
    np.testing.assert_allclose(drift_v, drift[:3])
    np.testing.assert_allclose(drift_w, drift[3:])
