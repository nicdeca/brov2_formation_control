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


def test_projection_repairs_residual_not_only_raw_constraint() -> None:
    policy = _policy()
    enabled = np.array([False, False, True, False])

    margin = policy.domain_margin[2]
    maximum = policy.maximum_enlargement[2]

    # h_a is still positive, but y = h_a - h_margin is negative.  The old
    # projection did not repair this case.
    conservative = np.array([1.0, 1.0, margin - 0.01, 1.0])
    state = np.zeros(4)

    projected, correction = policy.project_to_current_domain(
        state,
        conservative,
        enabled=enabled,
    )

    residual = (
        conservative[2]
        + maximum * projected[2]
        - policy.domain_margin[2]
    )
    assert correction[2] > 0.0
    assert residual >= policy.minimum_constraint_margin - 1e-12


def test_implicit_step_avoids_forward_euler_singularity_spike() -> None:
    policy = _policy()
    enabled = np.array([False, False, True, False])

    margin = policy.domain_margin[2]
    conservative = np.array(
        [
            1.0,
            1.0,
            margin + policy.minimum_constraint_margin,
            1.0,
        ]
    )
    state = np.zeros(4)
    dt = 0.02

    # At this sample the explicit law evaluates essentially at the numerical
    # denominator floor and would generate the pathological O(10^2) one-step
    # state increment observed in SITL.
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

    assert explicit_state[2] > 100.0
    assert np.all(np.isfinite(implicit_state))
    assert np.all(np.isfinite(effective_rate))
    assert 0.0 < implicit_state[2] < 1.0


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


def test_implicit_step_does_not_clip_state_at_one() -> None:
    policy = _policy()
    enabled = np.array([False, False, True, False])
    state = np.array([0.0, 0.0, 1.2, 0.0])
    conservative = np.array([1.0, 1.0, 1.0, 1.0])

    next_state, _ = policy.advance(
        state,
        conservative_values=conservative,
        enabled=enabled,
        sample_time=0.02,
    )

    assert 1.0 < next_state[2] < state[2]
