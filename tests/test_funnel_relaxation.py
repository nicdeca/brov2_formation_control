import numpy as np
import pytest

from formation_control.control import FunnelRelaxationPolicy
from formation_control.control.funnel_relaxation import (
    FunnelRelaxationInfeasibleError,
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


def test_inner_margin_decreases_with_state_and_vanishes_at_one():
    policy = FunnelRelaxationPolicy(
        maximum_enlargement=np.array([0.39, 3.96, 0.4816, 0.4816]),
        domain_margin_ratio=0.10,
    )
    state = np.array([0.0, 0.25, 0.5, 1.0])

    expected = (
        0.10
        * policy.maximum_enlargement
        * (1.0 - state)
    )
    np.testing.assert_allclose(policy.inner_margin(state), expected)
    assert policy.inner_margin(state)[3] == pytest.approx(0.0)


def test_adaptive_margin_equals_physical_constraint_at_s_one():
    policy = make_policy()
    state = np.ones(4)
    conservative = np.array([-0.1, 0.2, -0.05, 0.03])

    evaluation = policy.evaluate(
        state,
        conservative_values=conservative,
        enabled=np.zeros(4, dtype=bool),
    )

    physical = conservative + policy.maximum_enlargement
    np.testing.assert_allclose(evaluation.residual_margin, physical)


def test_barrier_rate_contains_one_plus_mu_factor():
    policy = FunnelRelaxationPolicy(
        maximum_enlargement=np.ones(4),
        recovery_gain=0.8,
        barrier_gain=0.20,
        domain_margin_ratio=0.20,
        activation_on_ratio=0.10,
        activation_off_ratio=0.30,
        minimum_constraint_margin=1e-5,
    )
    state = np.zeros(4)

    # y = h_c - mu_s rho_max = 0.05, hence sigma = 1.
    conservative = np.array([0.25, 1.0, 1.0, 1.0])
    evaluation = policy.evaluate(
        state,
        conservative_values=conservative,
        enabled=np.array([True, False, False, False]),
    )

    expected = 0.20 * (1.0 + 0.20) / 0.05
    assert evaluation.barrier_rate[0] == pytest.approx(expected)
    assert evaluation.selected_rate[0] == pytest.approx(expected)


def test_safeguard_repairs_state_dependent_adaptive_margin():
    policy = FunnelRelaxationPolicy(
        maximum_enlargement=np.array([1.0, 1.0, 1.0, 0.5]),
        domain_margin_ratio=0.10,
        minimum_constraint_margin=1e-5,
    )
    state = np.array([0.0, 0.0, 0.0, 0.4])

    maximum = policy.maximum_enlargement[3]
    mu_s = policy.domain_margin_ratio

    # Construct y = -1e-3 at the current state while keeping the physical
    # constraint strictly positive.
    h_c_vertical = (
        mu_s * maximum
        - (1.0 + mu_s) * maximum * state[3]
        - 1e-3
    )
    conservative = np.array([1.0, 1.0, 1.0, h_c_vertical])

    projected, correction = policy.project_to_current_domain(
        state,
        conservative,
        enabled=np.array([False, False, False, True]),
    )

    repaired_y = (
        conservative[3]
        - mu_s * maximum
        + (1.0 + mu_s) * maximum * projected[3]
    )

    assert correction[3] > 0.0
    assert 0.0 <= projected[3] <= 1.0
    assert repaired_y >= policy.minimum_constraint_margin - 1e-12


def test_safeguard_never_enlarges_beyond_physical_domain():
    policy = FunnelRelaxationPolicy(
        maximum_enlargement=np.array([1.0, 1.0, 1.0, 0.5]),
        domain_margin_ratio=0.10,
        minimum_constraint_margin=1e-5,
    )
    state = np.array([0.0, 0.0, 0.0, 0.9])

    # The physical margin is positive but smaller than the numerical floor.
    conservative = np.array(
        [1.0, 1.0, 1.0, -0.5 + 0.5e-5]
    )

    projected, correction = policy.project_to_current_domain(
        state,
        conservative,
        enabled=np.array([False, False, False, True]),
    )

    assert correction[3] > 0.0
    assert projected[3] == pytest.approx(1.0)


def test_safeguard_reports_nonpositive_physical_constraint():
    policy = FunnelRelaxationPolicy(
        maximum_enlargement=np.array([1.0, 1.0, 1.0, 0.5]),
        domain_margin_ratio=0.10,
    )

    with pytest.raises(FunnelRelaxationInfeasibleError):
        policy.project_to_current_domain(
            np.zeros(4),
            np.array([1.0, 1.0, 1.0, -0.6]),
            enabled=np.array([False, False, False, True]),
        )


def test_state_validation_rejects_values_above_one():
    policy = make_policy()

    with pytest.raises(ValueError):
        policy.validate_state(np.array([0.0, 1.01, 0.0, 0.0]))
