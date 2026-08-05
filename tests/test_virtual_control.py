import numpy as np

from formation_control.control import (
    VirtualVelocityGains,
    generalized_configuration_gradient,
    virtual_velocity_command,
)


def test_generalized_configuration_gradient():
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    position_gradient = np.array([2.0, 1.0, -0.5])
    orientation_gradient = np.array([0.2, -0.3, 0.4])

    gradient = generalized_configuration_gradient(
        rotation,
        position_gradient,
        orientation_gradient,
    )

    np.testing.assert_allclose(
        gradient,
        np.concatenate((rotation.T @ position_gradient, orientation_gradient)),
    )


def test_virtual_velocity_is_block_diagonal_gradient_descent():
    gains = VirtualVelocityGains(
        translation=np.diag([1.0, 2.0, 3.0]),
        rotation=np.diag([4.0, 5.0, 6.0]),
    )
    position_gradient = np.array([1.0, -2.0, 0.5])
    orientation_gradient = np.array([0.2, -0.1, 0.4])
    feedforward = np.array([0.1, 0.2, -0.3, 0.0, 0.1, -0.2])

    evaluation = virtual_velocity_command(
        np.eye(3),
        position_gradient,
        orientation_gradient,
        gains,
        feedforward_velocity_body=feedforward,
    )

    expected_gradient = np.concatenate((position_gradient, orientation_gradient))
    expected_command = feedforward - gains.matrix @ expected_gradient

    np.testing.assert_allclose(
        evaluation.generalized_gradient_body,
        expected_gradient,
    )
    np.testing.assert_allclose(
        evaluation.desired_velocity_body,
        expected_command,
    )


def test_virtual_command_decreases_potential_under_perfect_tracking():
    gains = VirtualVelocityGains.isotropic(
        translation=2.0,
        rotation=3.0,
    )
    evaluation = virtual_velocity_command(
        np.eye(3),
        np.array([1.0, -0.5, 0.2]),
        np.array([0.3, 0.1, -0.4]),
        gains,
    )

    modeled_potential_rate = float(
        evaluation.generalized_gradient_body @ evaluation.desired_velocity_body
    )

    expected = -float(
        evaluation.generalized_gradient_body @ gains.matrix @ evaluation.generalized_gradient_body
    )
    np.testing.assert_allclose(modeled_potential_rate, expected)
