import numpy as np
import pytest

from formation_control.control.funnel_relaxation import FunnelRelaxationPolicy


def _policy(**kwargs) -> FunnelRelaxationPolicy:
    defaults = dict(
        maximum_enlargement=np.array([0.30, 0.60, 0.4816, 0.4816]),
        desired_conservative_values=np.array([1.00, 1.20, 0.5184, 0.5184]),
        barrier_weights=np.array([0.18, 0.18, 0.25, 0.25]),
        recovery_gain=0.8,
        barrier_gain=0.20,
        activation_on_ratio=0.10,
        activation_off_ratio=0.30,
        infeasibility_epsilon=1e-3,
        minimum_constraint_margin=1e-8,
        sampled_guard_margin_ratio=0.02,
    )
    defaults.update(kwargs)
    return FunnelRelaxationPolicy(**defaults)


def test_implicit_advance_keeps_state_inside_unit_interval():
    policy = _policy()
    state = np.array([0.0, 0.0, 0.95, 0.0])
    conservative = np.array([1.0, 1.2, -0.44, 0.5184])
    next_state, effective_rate = policy.advance(
        state,
        conservative_values=conservative,
        required_slack=10.0,
        enabled=np.array([False, False, True, False]),
        sample_time=0.02,
    )
    assert np.all((0.0 <= next_state) & (next_state <= 1.0))
    np.testing.assert_allclose(effective_rate, (next_state - state) / 0.02)


def test_implicit_recovery_matches_backward_euler_when_sigma_zero():
    policy = _policy()
    state = np.array([0.4, 0.0, 0.0, 0.0])
    conservative = np.array([1.0, 1.2, 0.5184, 0.5184])
    dt = 0.02
    next_state, _ = policy.advance(
        state,
        conservative_values=conservative,
        required_slack=0.0,
        enabled=np.array([True, False, False, False]),
        sample_time=dt,
    )
    assert next_state[0] == pytest.approx(0.4 / (1.0 + 0.8 * dt))


def test_near_boundary_zero_required_slack_freezes_continuous_state():
    policy = _policy()
    state = np.array([0.0, 0.0, 0.5, 0.0])
    conservative = np.array([1.0, 1.2, -0.2, 0.5184])
    next_state, effective_rate = policy.advance(
        state,
        conservative_values=conservative,
        required_slack=0.0,
        enabled=np.array([False, False, True, False]),
        sample_time=0.02,
    )
    assert next_state[2] == pytest.approx(state[2])
    assert effective_rate[2] == pytest.approx(0.0)


def test_implicit_step_is_less_aggressive_than_explicit_near_singularity():
    policy = _policy(minimum_constraint_margin=1e-8)
    state = np.array([0.0, 0.0, 0.42, 0.0])
    conservative = np.array([1.0, 1.2, -0.201, 0.5184])
    dt = 0.02
    evaluation = policy.evaluate(
        state,
        conservative_values=conservative,
        required_slack=1.0,
        enabled=np.array([False, False, True, False]),
    )
    explicit = np.clip(state + dt * evaluation.selected_rate, 0.0, 1.0)
    implicit, _ = policy.advance(
        state,
        conservative_values=conservative,
        required_slack=1.0,
        enabled=np.array([False, False, True, False]),
        sample_time=dt,
    )
    assert implicit[2] >= state[2]
    assert implicit[2] <= explicit[2] + 1e-12
    assert implicit[2] <= 1.0
