import numpy as np
import pytest

from formation_control.control import (
    FunnelRelaxationInfeasibleError,
    FunnelRelaxationPolicy,
)


def make_policy() -> FunnelRelaxationPolicy:
    return FunnelRelaxationPolicy(
        maximum_enlargement=np.array([0.39, 3.96, 0.4816, 0.4816]),
        recovery_gain=0.8,
    )


def test_safe_relaxed_funnel_recovers_exponentially():
    policy = make_policy()
    state = np.array([0.0, 0.4, 0.0, 0.0])

    evaluation = policy.evaluate(
        state,
        conservative_values=np.array([8.0, 4.0, 0.5, 0.5]),
        conservative_rates=np.array([1.0, 1.0, 0.0, 0.0]),
        enabled=np.array([True, True, False, False]),
        sample_time=0.01,
    )

    assert evaluation.reference_rate[1] == pytest.approx(-0.32)
    assert evaluation.selected_rate[1] == pytest.approx(-0.32)
    assert not evaluation.expansion_required[1]


def test_auxiliary_input_expands_only_when_domain_preservation_requires_it():
    policy = FunnelRelaxationPolicy(
        maximum_enlargement=np.array([1.0, 1.0, 1.0, 0.5]),
        recovery_gain=0.8,
    )
    state = np.array([0.0, 0.0, 0.0, 0.4])
    dt = 0.02
    h_c = np.array([1.0, 1.0, 1.0, -0.15])
    h_c_dot = np.array([0.0, 0.0, 0.0, -3.0])

    evaluation = policy.evaluate(
        state,
        conservative_values=h_c,
        conservative_rates=h_c_dot,
        enabled=np.array([False, False, False, True]),
        sample_time=dt,
    )

    h_a = h_c[3] + 0.5 * state[3]
    expected_lower = (policy.domain_margin[3] - h_a - dt * h_c_dot[3]) / (dt * 0.5)

    assert expected_lower > 0.0
    assert evaluation.lower_rate[3] == pytest.approx(expected_lower)
    assert evaluation.selected_rate[3] == pytest.approx(expected_lower)
    assert evaluation.expansion_required[3]

    predicted_next = h_a + dt * (h_c_dot[3] + 0.5 * evaluation.selected_rate[3])
    assert predicted_next == pytest.approx(policy.domain_margin[3])


def test_domain_condition_can_slow_recovery_without_causing_expansion():
    policy = FunnelRelaxationPolicy(
        maximum_enlargement=np.array([1.0, 1.0, 1.0, 0.5]),
        recovery_gain=2.0,
    )
    state = np.array([0.0, 0.0, 0.0, 0.5])

    # With the default 10% practical margin, the vertical channel has
    #
    #   h_margin = 0.1 * 0.5 = 0.05.
    #
    # Choose h_a = 0.0525. Over dt = 0.01, the domain constraint gives
    # v >= -0.5, so the nominal recovery v_ref = -1 is slowed without
    # requiring positive expansion.
    evaluation = policy.evaluate(
        state,
        conservative_values=np.array([1.0, 1.0, 1.0, -0.1975]),
        conservative_rates=np.array([0.0, 0.0, 0.0, 0.0]),
        enabled=np.array([False, False, False, True]),
        sample_time=0.01,
    )

    assert evaluation.reference_rate[3] == pytest.approx(-1.0)
    assert evaluation.lower_rate[3] == pytest.approx(-0.5)
    assert evaluation.selected_rate[3] == pytest.approx(-0.5)
    assert evaluation.selected_rate[3] < 0.0
    assert not evaluation.expansion_required[3]


def test_sampled_state_bounds_keep_next_state_inside_unit_interval():
    policy = FunnelRelaxationPolicy(
        maximum_enlargement=np.ones(4),
        recovery_gain=0.8,
    )
    state = np.array([0.2, 0.4, 0.6, 0.8])
    dt = 0.02

    evaluation = policy.evaluate(
        state,
        conservative_values=np.full(4, 0.5),
        conservative_rates=np.zeros(4),
        enabled=np.ones(4, dtype=bool),
        sample_time=dt,
    )

    next_at_lower = state + dt * evaluation.lower_rate
    next_at_upper = state + dt * evaluation.upper_rate

    assert np.all(next_at_lower >= -1e-12)
    assert np.all(next_at_upper <= 1.0 + 1e-12)


def test_selected_rate_is_projection_of_recovery_rate():
    policy = FunnelRelaxationPolicy(
        maximum_enlargement=np.ones(4),
        recovery_gain=np.array([0.2, 0.4, 0.6, 0.8]),
    )
    state = np.array([0.2, 0.4, 0.6, 0.8])

    evaluation = policy.evaluate(
        state,
        conservative_values=np.full(4, 0.5),
        conservative_rates=np.zeros(4),
        enabled=np.ones(4, dtype=bool),
        sample_time=0.01,
    )

    np.testing.assert_allclose(
        evaluation.selected_rate,
        np.clip(
            evaluation.reference_rate,
            evaluation.lower_rate,
            evaluation.upper_rate,
        ),
    )


def test_physical_limit_exhaustion_is_reported():
    policy = make_policy()

    with pytest.raises(FunnelRelaxationInfeasibleError):
        policy.evaluate(
            np.array([0.0, 1.0, 0.0, 0.0]),
            conservative_values=np.array([8.0, -3.97, 0.5, 0.5]),
            conservative_rates=np.array([1.0, -1.0, 0.0, 0.0]),
            enabled=np.array([True, True, False, False]),
            sample_time=0.01,
        )


def test_no_admissible_rate_is_reported_at_full_relaxation():
    policy = FunnelRelaxationPolicy(
        maximum_enlargement=np.array([1.0, 1.0, 1.0, 0.5]),
        recovery_gain=0.8,
    )

    with pytest.raises(FunnelRelaxationInfeasibleError):
        policy.evaluate(
            np.array([0.0, 0.0, 0.0, 1.0]),
            conservative_values=np.array([1.0, 1.0, 1.0, -0.49]),
            conservative_rates=np.array([0.0, 0.0, 0.0, -2.0]),
            enabled=np.array([False, False, False, True]),
            sample_time=0.01,
        )


def test_domain_margin_scales_with_available_relaxation_reserve():
    policy = FunnelRelaxationPolicy(
        maximum_enlargement=np.array([0.39, 3.96, 0.4816, 0.4816]),
        recovery_gain=0.8,
        domain_margin_ratio=0.01,
        minimum_constraint_margin=1e-5,
    )

    np.testing.assert_allclose(
        policy.domain_margin,
        0.01 * policy.maximum_enlargement,
    )


def test_absolute_floor_is_used_when_scaled_margin_would_be_too_small():
    policy = FunnelRelaxationPolicy(
        maximum_enlargement=np.array([0.0, 1e-6, 1.0, 1.0]),
        domain_margin_ratio=0.01,
        minimum_constraint_margin=1e-5,
    )

    assert policy.domain_margin[0] == pytest.approx(1e-5)
    assert policy.domain_margin[1] == pytest.approx(1e-5)
    assert policy.domain_margin[2] == pytest.approx(1e-2)


def test_safeguard_does_not_fire_for_small_practical_margin_undershoot():
    policy = FunnelRelaxationPolicy(
        maximum_enlargement=np.array([1.0, 1.0, 1.0, 0.5]),
        domain_margin_ratio=0.01,
        minimum_constraint_margin=1e-5,
    )

    # For the vertical channel, h_margin = 0.005.  Construct h_a = 0.002:
    # below the practical CBF margin, but still safely inside the logarithmic
    # domain.  The emergency safeguard should leave it untouched.
    state = np.array([0.0, 0.0, 0.0, 0.4])
    h_c = np.array([1.0, 1.0, 1.0, -0.198])

    projected, correction = policy.project_to_current_domain(
        state,
        h_c,
        enabled=np.array([False, False, False, True]),
    )

    np.testing.assert_allclose(projected, state)
    np.testing.assert_allclose(correction, np.zeros(4))


def test_safeguard_projects_near_log_singularity_back_to_practical_margin():
    policy = FunnelRelaxationPolicy(
        maximum_enlargement=np.array([1.0, 1.0, 1.0, 0.5]),
        domain_margin_ratio=0.01,
        minimum_constraint_margin=1e-5,
    )

    state = np.array([0.0, 0.0, 0.0, 0.4])
    # h_a = -0.200 -?  Use h_c = -0.199995 so h_a = 5e-6 < floor.
    h_c = np.array([1.0, 1.0, 1.0, -0.199995])

    projected, correction = policy.project_to_current_domain(
        state,
        h_c,
        enabled=np.array([False, False, False, True]),
    )

    assert correction[3] > 0.0
    repaired_h = h_c[3] + 0.5 * projected[3]
    assert repaired_h == pytest.approx(policy.domain_margin[3])


def test_sampled_controller_targets_practical_margin_not_absolute_floor():
    policy = FunnelRelaxationPolicy(
        maximum_enlargement=np.array([1.0, 1.0, 1.0, 0.5]),
        recovery_gain=0.8,
        domain_margin_ratio=0.01,
        minimum_constraint_margin=1e-5,
    )
    state = np.array([0.0, 0.0, 0.0, 0.4])
    h_c = np.array([1.0, 1.0, 1.0, -0.15])
    h_c_dot = np.array([0.0, 0.0, 0.0, -3.0])
    dt = 0.02

    evaluation = policy.evaluate(
        state,
        conservative_values=h_c,
        conservative_rates=h_c_dot,
        enabled=np.array([False, False, False, True]),
        sample_time=dt,
    )

    h_a = h_c[3] + 0.5 * state[3]
    predicted_next = h_a + dt * (h_c_dot[3] + 0.5 * evaluation.selected_rate[3])
    assert predicted_next == pytest.approx(policy.domain_margin[3])
    assert predicted_next > policy.minimum_constraint_margin
