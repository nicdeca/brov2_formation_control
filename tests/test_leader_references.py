"""Tests for leader trajectory and velocity-command references."""

import numpy as np

from formation_control.simulation.leader_references import (
    SmoothSpatialTrajectoryReference,
    VelocityCommandReferenceFilter,
)


def test_velocity_command_filter_supplies_position_velocity_acceleration():
    reference_filter = VelocityCommandReferenceFilter(bandwidth=0.5)
    state = reference_filter.initialize(
        np.array([1.0, -2.0, 0.5]),
        velocity=np.zeros(3),
    )
    command = np.array([0.4, -0.2, 0.1])

    sample = reference_filter.evaluate(state, command)

    np.testing.assert_allclose(
        sample.position,
        np.array([1.0, -2.0, 0.5]),
    )
    np.testing.assert_allclose(sample.velocity, np.zeros(3))
    np.testing.assert_allclose(sample.acceleration, 0.5 * command)


def test_velocity_command_filter_dynamics_integrate_filtered_velocity():
    reference_filter = VelocityCommandReferenceFilter(bandwidth=np.array([0.5, 1.0, 2.0]))
    state = reference_filter.initialize(
        np.zeros(3),
        velocity=np.array([0.1, -0.2, 0.3]),
    )
    command = np.array([0.4, 0.1, -0.1])

    derivative = reference_filter.dynamics(state, command)

    np.testing.assert_allclose(
        derivative[:3],
        np.array([0.1, -0.2, 0.3]),
    )
    np.testing.assert_allclose(
        derivative[3:],
        np.array([0.15, 0.3, -0.8]),
    )


def test_spatial_trajectory_supplies_consistent_finite_reference():
    reference = SmoothSpatialTrajectoryReference(
        initial_position=np.array([0.0, 0.0, -1.0]),
    )

    for time in (0.0, 5.0, 50.0, 200.0):
        sample = reference.evaluate(time)
        assert np.all(np.isfinite(sample.position))
        assert np.all(np.isfinite(sample.velocity))
        assert np.all(np.isfinite(sample.acceleration))
