import numpy as np
import pytest

from formation_control.control import BacksteppingCLF


def test_clf_value_contains_configuration_and_velocity_error():
    inertia = np.diag([2.0, 3.0, 4.0])
    clf = BacksteppingCLF(inertia)

    velocity = np.array([0.7, -0.2, 0.4])
    filtered = np.array([0.2, -0.1, 0.0])
    error = velocity - filtered

    evaluation = clf.evaluate(
        configuration_value=1.3,
        generalized_configuration_gradient=np.array([0.4, -0.2, 0.1]),
        generalized_velocity=velocity,
        filtered_velocity=filtered,
        filtered_velocity_derivative=np.zeros(3),
        dynamics_bias=np.zeros(3),
    )

    expected_kinetic = 0.5 * float(error @ inertia @ error)

    assert evaluation.configuration_value == pytest.approx(1.3)
    assert evaluation.velocity_error_value == pytest.approx(expected_kinetic)
    assert evaluation.value == pytest.approx(1.3 + expected_kinetic)
    np.testing.assert_allclose(evaluation.velocity_error, error)


def test_clf_affine_derivative_matches_direct_differentiation():
    inertia = np.diag([2.0, 3.0, 4.0])
    clf = BacksteppingCLF(inertia)

    position = np.array([0.8, -0.4, 0.2])
    gain = np.diag([1.5, 2.0, 0.7])
    gradient = gain @ position
    configuration_value = 0.5 * float(position @ gain @ position)

    velocity = np.array([0.5, -0.2, 0.3])
    filtered = np.array([0.1, -0.1, 0.2])
    filtered_derivative = np.array([0.05, -0.03, 0.02])
    dynamics_bias = np.array([0.4, -0.2, 0.1])
    control = np.array([1.2, -0.7, 0.5])
    rate_offset = 0.08

    evaluation = clf.evaluate(
        configuration_value=configuration_value,
        generalized_configuration_gradient=gradient,
        generalized_velocity=velocity,
        filtered_velocity=filtered,
        filtered_velocity_derivative=filtered_derivative,
        dynamics_bias=dynamics_bias,
        configuration_rate_offset=rate_offset,
    )

    acceleration = np.linalg.solve(
        inertia,
        control - dynamics_bias,
    )
    error = velocity - filtered

    direct_derivative = (
        float(gradient @ velocity)
        + rate_offset
        + float(error @ inertia @ (acceleration - filtered_derivative))
    )

    assert evaluation.derivative(control) == pytest.approx(direct_derivative)


def test_clf_derivative_matches_finite_difference():
    inertia = np.diag([2.0, 3.0, 4.0])
    gain = np.diag([1.5, 2.0, 0.7])
    clf = BacksteppingCLF(inertia)

    position = np.array([0.8, -0.4, 0.2])
    velocity = np.array([0.5, -0.2, 0.3])
    filtered = np.array([0.1, -0.1, 0.2])
    filtered_derivative = np.array([0.05, -0.03, 0.02])
    dynamics_bias = np.array([0.4, -0.2, 0.1])
    control = np.array([1.2, -0.7, 0.5])

    def value(q, nu, nu_c):
        configuration = 0.5 * float(q @ gain @ q)
        error = nu - nu_c
        return configuration + 0.5 * float(error @ inertia @ error)

    evaluation = clf.evaluate(
        configuration_value=0.5 * float(position @ gain @ position),
        generalized_configuration_gradient=gain @ position,
        generalized_velocity=velocity,
        filtered_velocity=filtered,
        filtered_velocity_derivative=filtered_derivative,
        dynamics_bias=dynamics_bias,
    )

    position_derivative = velocity
    velocity_derivative = np.linalg.solve(
        inertia,
        control - dynamics_bias,
    )

    epsilon = 1e-7
    value_plus = value(
        position + epsilon * position_derivative,
        velocity + epsilon * velocity_derivative,
        filtered + epsilon * filtered_derivative,
    )
    value_minus = value(
        position - epsilon * position_derivative,
        velocity - epsilon * velocity_derivative,
        filtered - epsilon * filtered_derivative,
    )
    numeric_derivative = (value_plus - value_minus) / (2.0 * epsilon)

    assert evaluation.derivative(control) == pytest.approx(
        numeric_derivative,
        abs=1e-8,
    )


def test_input_matrix_maps_clf_gradient_to_thruster_coordinates():
    inertia = np.diag([2.0, 3.0, 4.0])
    allocation = np.array(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, -1.0, 0.0, 0.0],
        ]
    )
    clf = BacksteppingCLF(
        inertia=inertia,
        input_matrix=allocation,
    )

    velocity = np.array([0.5, -0.2, 0.3])
    filtered = np.array([0.1, -0.1, 0.2])
    error = velocity - filtered

    evaluation = clf.evaluate(
        configuration_value=0.4,
        generalized_configuration_gradient=np.array([0.2, -0.1, 0.3]),
        generalized_velocity=velocity,
        filtered_velocity=filtered,
        filtered_velocity_derivative=np.zeros(3),
        dynamics_bias=np.zeros(3),
    )

    np.testing.assert_allclose(
        evaluation.control_gradient,
        allocation.T @ error,
    )


def test_clf_rejects_non_positive_definite_inertia():
    with pytest.raises(ValueError):
        BacksteppingCLF(np.diag([1.0, -1.0]))
