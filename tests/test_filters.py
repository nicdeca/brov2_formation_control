import numpy as np
import pytest

from formation_control.control import (
    FirstOrderCommandFilter,
    SecondOrderCommandFilter,
)
from formation_control.simulation import RK4Integrator


def test_first_order_filter_initializes_at_command():
    command_filter = FirstOrderCommandFilter(signal_dim=3, bandwidth=4.0)
    command = np.array([1.0, -2.0, 0.5])

    state = command_filter.initialize(command)
    evaluation = command_filter.evaluate(state, command)

    np.testing.assert_allclose(state, command)
    np.testing.assert_allclose(evaluation.output, command)
    np.testing.assert_allclose(evaluation.output_derivative, np.zeros(3))


def test_first_order_filter_supports_per_axis_bandwidth():
    command_filter = FirstOrderCommandFilter(
        signal_dim=3,
        bandwidth=np.array([1.0, 2.0, 4.0]),
    )
    state = np.zeros(3)
    command = np.ones(3)

    np.testing.assert_allclose(
        command_filter.dynamics(state, command),
        np.array([1.0, 2.0, 4.0]),
    )


def test_second_order_filter_initialization():
    command_filter = SecondOrderCommandFilter(
        signal_dim=3,
        natural_frequency=5.0,
        damping_ratio=1.0,
    )
    command = np.array([1.0, -2.0, 0.5])

    state = command_filter.initialize(command)
    evaluation = command_filter.evaluate(state, command)

    np.testing.assert_allclose(evaluation.output, command)
    np.testing.assert_allclose(evaluation.output_derivative, np.zeros(3))
    np.testing.assert_allclose(
        command_filter.dynamics(state, command),
        np.zeros(6),
    )


def test_second_order_filter_dynamics():
    command_filter = SecondOrderCommandFilter(
        signal_dim=2,
        natural_frequency=np.array([2.0, 3.0]),
        damping_ratio=np.array([0.5, 1.0]),
    )
    state = np.array([0.0, 1.0, 0.2, -0.1])
    command = np.array([1.0, -1.0])

    derivative = command_filter.dynamics(state, command)

    expected_acceleration = (
        np.array([2.0, 3.0]) ** 2 * (command - state[:2])
        - 2.0 * np.array([0.5, 1.0]) * np.array([2.0, 3.0]) * state[2:]
    )
    np.testing.assert_allclose(
        derivative,
        np.concatenate((state[2:], expected_acceleration)),
    )


def test_filters_can_use_generic_rk4_integrator():
    command_filter = FirstOrderCommandFilter(signal_dim=1, bandwidth=2.0)
    integrator = RK4Integrator()
    state = np.array([0.0])
    command = np.array([1.0])

    next_state = integrator.step(
        command_filter,
        state,
        command,
        dt=0.1,
    )

    exact = 1.0 - np.exp(-0.2)
    assert next_state[0] == pytest.approx(exact, rel=2e-5)
