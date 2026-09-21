import numpy as np
import pytest

from formation_control.control import (
    FunnelRelaxationInfeasibleError,
    FunnelRelaxationPolicy,
)


def make_policy(**kwargs) -> FunnelRelaxationPolicy:
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


def test_activation_thresholds_are_inside_desired_constraint_value():
    policy = make_policy()
    assert np.all(policy.activation_on_margin > 0.0)
    assert np.all(policy.activation_on_margin < policy.activation_off_margin)
    assert np.all(policy.activation_off_margin < policy.desired_conservative_values)


def test_zero_required_slack_disables_continuous_enlargement():
    policy = make_policy()
    state = np.array([0.0, 0.0, 0.4, 0.0])
    h_c = np.array([1.0, 1.2, -0.15, 0.5184])
    evaluation = policy.evaluate(
        state,
        conservative_values=h_c,
        required_slack=0.0,
        enabled=np.array([False, False, True, False]),
    )
    assert evaluation.infeasibility_activation == pytest.approx(0.0)
    assert evaluation.activation[2] == pytest.approx(1.0)
    assert evaluation.selected_rate[2] == pytest.approx(0.0)


def test_positive_required_slack_activates_enlargement_near_boundary():
    policy = make_policy(infeasibility_epsilon=0.01)
    state = np.array([0.0, 0.0, 0.4, 0.0])
    h_c = np.array([1.0, 1.2, -0.15, 0.5184])
    evaluation = policy.evaluate(
        state,
        conservative_values=h_c,
        required_slack=0.02,
        enabled=np.array([False, False, True, False]),
    )
    assert evaluation.infeasibility_activation == pytest.approx(0.8)
    assert evaluation.barrier_derivative[2] < 0.0
    assert evaluation.enlargement_rate[2] > 0.0


def test_common_shift_derivative_matches_paper_formula():
    policy = make_policy()
    state = np.array([0.25, 0.0, 0.0, 0.0])
    h_c = np.array([0.40, 1.20, 0.5184, 0.5184])
    evaluation = policy.evaluate(
        state,
        conservative_values=h_c,
        required_slack=0.01,
        enabled=np.array([True, False, False, False]),
    )
    rho = policy.maximum_enlargement[0]
    mu = policy.barrier_weights[0]
    h_a = h_c[0] + rho * state[0]
    h_d_a = policy.desired_conservative_values[0] + rho * state[0]
    expected = -mu * rho * (h_a - h_d_a) ** 2 / (h_a * h_d_a**2)
    assert evaluation.barrier_derivative[0] == pytest.approx(expected)


def test_far_from_boundary_only_recovery_remains():
    policy = make_policy()
    state = np.array([0.4, 0.0, 0.0, 0.0])
    h_c = np.array([1.0, 1.2, 0.5184, 0.5184])
    evaluation = policy.evaluate(
        state,
        conservative_values=h_c,
        required_slack=10.0,
        enabled=np.array([True, False, False, False]),
    )
    assert evaluation.activation[0] == pytest.approx(0.0)
    assert evaluation.selected_rate[0] == pytest.approx(-0.8 * 0.4)


def test_upper_projection_blocks_outward_motion_at_physical_limit():
    policy = make_policy()
    state = np.array([0.0, 0.0, 1.0, 0.0])
    h_c = np.array([1.0, 1.2, -0.47, 0.5184])
    evaluation = policy.evaluate(
        state,
        conservative_values=h_c,
        required_slack=1.0,
        enabled=np.array([False, False, True, False]),
    )
    assert evaluation.enlargement_rate[2] > 0.0
    assert evaluation.selected_rate[2] == pytest.approx(0.0)


def test_sampled_guard_repairs_to_configured_margin():
    policy = make_policy(sampled_guard_margin_ratio=0.02)
    state = np.zeros(4)
    h_c = np.array([1.0, 1.2, -0.0079, 0.5184])
    repaired, correction = policy.project_to_current_domain(
        state,
        h_c,
        enabled=np.array([False, False, True, False]),
    )
    h_a = h_c[2] + policy.maximum_enlargement[2] * repaired[2]
    assert correction[2] > 0.0
    assert h_a == pytest.approx(policy.sampled_guard_margin[2])
    assert repaired[2] <= 1.0


def test_initialization_rejects_state_outside_physical_domain():
    policy = make_policy(minimum_constraint_margin=1e-6)
    h_c = np.array([1.0, 1.2, -0.6, 0.5184])
    with pytest.raises(FunnelRelaxationInfeasibleError):
        policy.initialize_for_constraint_values(
            h_c,
            enabled=np.array([False, False, True, False]),
        )


def test_predictive_sampled_guard_anticipates_negative_finite_difference_rate():
    policy = make_policy(sampled_guard_margin_ratio=0.02)
    state = np.zeros(4)
    h_c = np.array([1.0, 1.2, 0.030, 0.5184])
    h_dot = np.array([0.0, 0.0, -2.2, 0.0])
    dt = 0.01
    repaired, correction = policy.project_to_predicted_domain(
        state,
        h_c,
        h_dot,
        sample_time=dt,
        enabled=np.array([False, False, True, False]),
    )
    predicted = h_c[2] + dt * min(h_dot[2], 0.0)
    predicted_h_a = predicted + policy.maximum_enlargement[2] * repaired[2]
    assert correction[2] > 0.0
    assert predicted_h_a == pytest.approx(policy.sampled_guard_margin[2])


def test_predictive_sampled_guard_ignores_positive_rate():
    policy = make_policy(sampled_guard_margin_ratio=0.02)
    state = np.zeros(4)
    h_c = np.array([1.0, 1.2, 0.030, 0.5184])
    h_dot = np.array([0.0, 0.0, 2.0, 0.0])
    repaired, correction = policy.project_to_predicted_domain(
        state,
        h_c,
        h_dot,
        sample_time=0.01,
        enabled=np.array([False, False, True, False]),
    )
    np.testing.assert_allclose(repaired, state)
    np.testing.assert_allclose(correction, 0.0)
