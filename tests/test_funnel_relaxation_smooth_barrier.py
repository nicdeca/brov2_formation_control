import numpy as np
import pytest

from formation_control.control import FunnelRelaxationPolicy


def make_policy() -> FunnelRelaxationPolicy:
    return FunnelRelaxationPolicy(
        maximum_enlargement=np.array([0.39, 3.96, 0.4816, 0.4816]),
        recovery_gain=0.8,
        barrier_gain=0.2,
        domain_margin_ratio=0.1,
        activation_on_ratio=0.1,
        activation_off_ratio=0.3,
    )


def test_far_from_boundary_recovers_exponentially():
    policy = make_policy()
    state = np.array([0.0, 0.4, 0.0, 0.0])
    evaluation = policy.evaluate(
        state,
        conservative_values=np.array([8.0, 4.0, 0.5, 0.5]),
        enabled=np.ones(4, dtype=bool),
    )
    assert evaluation.activation[1] == pytest.approx(0.0)
    assert evaluation.barrier_rate[1] == pytest.approx(0.0)
    assert evaluation.selected_rate[1] == pytest.approx(-0.32)


def test_barrier_is_fully_active_near_positive_margin():
    policy = make_policy()
    channel = 3
    maximum = policy.maximum_enlargement[channel]
    state = np.array([0.0, 0.0, 0.0, 0.4])
    residual = 0.05 * maximum
    adaptive_value = policy.domain_margin[channel] + residual
    conservative = adaptive_value - maximum * state[channel]

    evaluation = policy.evaluate(
        state,
        conservative_values=np.array([1.0, 1.0, 1.0, conservative]),
        enabled=np.array([False, False, False, True]),
    )

    assert evaluation.residual_margin[channel] == pytest.approx(residual)
    assert evaluation.activation[channel] == pytest.approx(1.0)
    assert evaluation.barrier_rate[channel] > 0.0


def test_smooth_transition_is_between_zero_and_one():
    policy = make_policy()
    channel = 3
    maximum = policy.maximum_enlargement[channel]
    state = np.zeros(4)
    y_on = policy.activation_on_margin[channel]
    y_off = policy.activation_off_margin[channel]
    residual = 0.5 * (y_on + y_off)
    conservative = policy.domain_margin[channel] + residual

    evaluation = policy.evaluate(
        state,
        conservative_values=np.array([1.0, 1.0, 1.0, conservative]),
        enabled=np.array([False, False, False, True]),
    )
    assert 0.0 < evaluation.activation[channel] < 1.0


def test_zero_state_cannot_be_driven_negative():
    policy = make_policy()
    state = np.zeros(4)
    evaluation = policy.evaluate(
        state,
        conservative_values=np.array([8.0, 4.0, 0.5, 0.5]),
        enabled=np.ones(4, dtype=bool),
    )
    assert np.all(evaluation.selected_rate >= -1e-12)


def test_state_above_one_is_allowed():
    policy = make_policy()
    state = np.array([0.0, 1.2, 0.0, 0.0])
    np.testing.assert_allclose(
        policy.enlargement(state)[1],
        1.2 * policy.maximum_enlargement[1],
    )


def test_rate_does_not_depend_on_constraint_derivative():
    policy = make_policy()
    state = np.array([0.0, 0.0, 0.0, 0.3])
    values = np.array([1.0, 1.0, 1.0, -0.04])
    enabled = np.array([False, False, False, True])

    first = policy.evaluate(
        state,
        conservative_values=values,
        enabled=enabled,
        conservative_rates=np.array([0.0, 0.0, 0.0, -100.0]),
        sample_time=0.01,
    )
    second = policy.evaluate(
        state,
        conservative_values=values,
        enabled=enabled,
        conservative_rates=np.array([0.0, 0.0, 0.0, 100.0]),
        sample_time=1.0,
    )
    np.testing.assert_allclose(first.selected_rate, second.selected_rate)
