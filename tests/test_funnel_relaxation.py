import numpy as np
import pytest

from formation_control.control import FunnelRelaxationPolicy


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


def test_safeguard_does_not_fire_for_small_adaptive_margin_undershoot():
    policy = FunnelRelaxationPolicy(
        maximum_enlargement=np.array([1.0, 1.0, 1.0, 0.5]),
        domain_margin_ratio=0.01,
        minimum_constraint_margin=1e-5,
    )

    # For the vertical channel, the prescribed adaptive margin is 0.005.
    # Construct h_a = 0.002: below this margin, but still safely inside
    # the logarithmic domain. The emergency safeguard should leave it untouched.
    state = np.array([0.0, 0.0, 0.0, 0.4])
    h_c = np.array([1.0, 1.0, 1.0, -0.198])

    projected, correction = policy.project_to_current_domain(
        state,
        h_c,
        enabled=np.array([False, False, False, True]),
    )

    np.testing.assert_allclose(projected, state)
    np.testing.assert_allclose(correction, np.zeros(4))


def test_safeguard_projects_near_log_singularity_to_admissible_margin():
    policy = FunnelRelaxationPolicy(
        maximum_enlargement=np.array([1.0, 1.0, 1.0, 0.5]),
        domain_margin_ratio=0.01,
        minimum_constraint_margin=1e-5,
    )

    state = np.array([0.0, 0.0, 0.0, 0.4])

    # h_a = 5e-6, i.e., below the numerical floor of the logarithmic domain.
    h_c = np.array([1.0, 1.0, 1.0, -0.199995])

    projected, correction = policy.project_to_current_domain(
        state,
        h_c,
        enabled=np.array([False, False, False, True]),
    )

    assert correction[3] > 0.0
    repaired_h = h_c[3] + 0.5 * projected[3]
    assert repaired_h == pytest.approx(policy.domain_margin[3])
