import numpy as np

from formation_control_ros.reference_runtime import (
    VelocityCommandReference,
)


def test_velocity_reference_starts_with_consistent_acceleration():
    reference = VelocityCommandReference.initialize(
        np.array([1.0, 2.0, 3.0]),
        np.zeros(3),
        0.2,
    )
    sample = reference.sample(np.array([0.5, 0.0, -0.5]))
    np.testing.assert_allclose(
        sample.acceleration,
        np.array([0.1, 0.0, -0.1]),
    )


def test_velocity_reference_exact_update():
    reference = VelocityCommandReference.initialize(
        np.zeros(3),
        np.zeros(3),
        1.0,
    )
    command = np.ones(3)
    reference.advance(command, 1.0)

    expected_velocity = 1.0 - np.exp(-1.0)
    expected_position = np.exp(-1.0)
    np.testing.assert_allclose(
        reference.velocity,
        expected_velocity * np.ones(3),
    )
    np.testing.assert_allclose(
        reference.position,
        expected_position * np.ones(3),
    )
