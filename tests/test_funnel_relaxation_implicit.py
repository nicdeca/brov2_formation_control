import numpy as np

from formation_control.control.funnel_relaxation import FunnelRelaxationPolicy


def _policy() -> FunnelRelaxationPolicy:
    return FunnelRelaxationPolicy(
        maximum_enlargement=np.array([0.51, 7.2, 0.7975, 0.7975]),
        recovery_gain=0.8,
        barrier_gain=0.20,
        domain_margin_ratio=0.02,
        activation_on_ratio=0.001,
        activation_off_ratio=0.15,
        minimum_constraint_margin=1e-5,
    )


def test_projection_repairs_state_dependent_adaptive_margin() -> None:
    policy = _policy()
    enabled = np.array([False, False, True, False])

    maximum = policy.maximum_enlargement[2]
    mu_s = policy.domain_margin_ratio

    # At s=0, y = h_c - mu_s rho_max.  Put y slightly below zero.
    conservative = np.array(
        [1.0, 1.0, mu_s * maximum - 0.01, 1.0]
    )
    state = np.zeros(4)

    projected, correction = policy.project_to_current_domain(
        state,
        conservative,
        enabled=enabled,
    )

    residual = (
        conservative[2]
        - mu_s * maximum
        + (1.0 + mu_s) * maximum * projected[2]
    )
    assert correction[2] > 0.0
    assert residual >= policy.minimum_constraint_margin - 1e-12
    assert projected[2] <= 1.0


def test_implicit_step_avoids_forward_euler_singularity_spike() -> None:
    policy = _policy()
    enabled = np.array([False, False, True, False])

    maximum = policy.maximum_enlargement[2]
    mu_s = policy.domain_margin_ratio

    # At s=0 choose h_c so that y equals the numerical denominator floor.
    conservative = np.array(
        [
            1.0,
            1.0,
            mu_s * maximum + policy.minimum_constraint_margin,
            1.0,
        ]
    )
    state = np.zeros(4)
    dt = 0.02

    evaluation = policy.evaluate(
        state,
        conservative_values=conservative,
        enabled=enabled,
    )
    explicit_state = state + dt * evaluation.selected_rate

    implicit_state, effective_rate = policy.advance(
        state,
        conservative_values=conservative,
        enabled=enabled,
        sample_time=dt,
    )

    assert explicit_state[2] > 1.0
    assert np.all(np.isfinite(implicit_state))
    assert np.all(np.isfinite(effective_rate))
    assert 0.0 < implicit_state[2] <= 1.0


def test_implicit_step_matches_implicit_recovery_when_barrier_is_off() -> None:
    policy = _policy()
    enabled = np.array([False, False, True, False])
    state = np.array([0.0, 0.0, 0.5, 0.0])

    # Large conservative value -> y is above activation_off_margin, hence
    # sigma = 0 and only -k_s s remains.
    conservative = np.array([1.0, 1.0, 1.0, 1.0])
    dt = 0.02

    next_state, _ = policy.advance(
        state,
        conservative_values=conservative,
        enabled=enabled,
        sample_time=dt,
    )

    expected = state[2] / (1.0 + dt * policy.recovery_gain[2])
    assert np.isclose(next_state[2], expected, rtol=0.0, atol=1e-10)


def test_upper_projection_blocks_outward_rate_at_s_one() -> None:
    policy = _policy()
    enabled = np.array([False, False, True, False])

    maximum = policy.maximum_enlargement[2]
    state = np.array([0.0, 0.0, 1.0, 0.0])

    # At s=1, y equals the physical margin.  Make it small enough that the
    # unprojected barrier action dominates recovery.
    conservative = np.array(
        [1.0, 1.0, -maximum + 0.01, 1.0]
    )

    evaluation = policy.evaluate(
        state,
        conservative_values=conservative,
        enabled=enabled,
    )
    next_state, effective_rate = policy.advance(
        state,
        conservative_values=conservative,
        enabled=enabled,
        sample_time=0.02,
    )

    assert evaluation.barrier_rate[2] > -evaluation.reference_rate[2]
    assert evaluation.selected_rate[2] == 0.0
    assert next_state[2] == 1.0
    assert effective_rate[2] == 0.0


def test_upper_projection_allows_recovery_from_s_one() -> None:
    policy = _policy()
    enabled = np.array([False, False, True, False])
    state = np.array([0.0, 0.0, 1.0, 0.0])

    # Large h_c switches the barrier term off, so the projected vector field
    # points inward and recovery from s=1 is allowed.
    conservative = np.array([1.0, 1.0, 1.0, 1.0])
    dt = 0.02

    evaluation = policy.evaluate(
        state,
        conservative_values=conservative,
        enabled=enabled,
    )
    next_state, _ = policy.advance(
        state,
        conservative_values=conservative,
        enabled=enabled,
        sample_time=dt,
    )

    expected = 1.0 / (1.0 + dt * policy.recovery_gain[2])
    assert evaluation.selected_rate[2] < 0.0
    assert np.isclose(next_state[2], expected, rtol=0.0, atol=1e-10)
