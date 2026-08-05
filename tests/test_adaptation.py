import numpy as np
import pytest

from formation_control.constraints import DistanceDomain, FieldOfViewDomain
from formation_control.control import (
    AdaptiveDomainDynamics,
    AdaptiveEnlargementLaw,
)
from formation_control.simulation import RK4Integrator


def make_domains():
    return (
        DistanceDomain(
            d_min=0.5,
            d_max=5.0,
            d_min_conservative=0.8,
            d_max_conservative=4.0,
        ),
        FieldOfViewDomain(
            alpha_h_conservative=0.8,
            alpha_v_conservative=0.7,
        ),
    )


def test_effective_distance_limits_recover_conservative_and_physical_domains():
    distance, _ = make_domains()

    assert distance.effective_minimum_distance(0.0) == pytest.approx(0.8)
    assert distance.effective_maximum_distance(0.0) == pytest.approx(4.0)
    assert distance.effective_minimum_distance(distance.collision_enlargement_max) == pytest.approx(
        0.5
    )
    assert distance.effective_maximum_distance(distance.range_enlargement_max) == pytest.approx(5.0)


def test_effective_fov_limits_recover_conservative_and_physical_domains():
    _, fov = make_domains()

    assert fov.effective_horizontal_limit(0.0) == pytest.approx(0.8)
    assert fov.effective_vertical_limit(0.0) == pytest.approx(0.7)
    assert fov.effective_horizontal_limit(fov.horizontal_enlargement_max) == pytest.approx(1.0)
    assert fov.effective_vertical_limit(fov.vertical_enlargement_max) == pytest.approx(1.0)


def test_scalar_adaptation_expands_and_recovers():
    law = AdaptiveEnlargementLaw(
        maximum=1.0,
        expansion_gain=2.0,
        recovery_gain=0.5,
    )

    assert law.rate(0.2, activation=1.0) == pytest.approx(1.9)
    assert law.rate(0.2, activation=0.0) == pytest.approx(-0.1)


def test_scalar_adaptation_projection_prevents_outward_motion_at_boundaries():
    law = AdaptiveEnlargementLaw(
        maximum=1.0,
        expansion_gain=2.0,
        recovery_gain=0.5,
    )

    assert law.rate(0.0, activation=0.0) == pytest.approx(0.0)
    assert law.rate(1.0, activation=1.0) == pytest.approx(0.0)
    assert law.project(-0.2) == pytest.approx(0.0)
    assert law.project(1.3) == pytest.approx(1.0)


def test_domain_adaptation_is_driven_by_slack_threshold():
    distance, fov = make_domains()
    adaptation = AdaptiveDomainDynamics.from_domains(
        distance,
        fov,
        expansion_gain=2.0,
        recovery_gain=0.5,
        slack_threshold=0.2,
    )
    state = np.array([0.1, 0.2, 0.05, 0.1])

    below = adaptation.dynamics(state, np.array([0.1]))
    above = adaptation.dynamics(state, np.array([0.7]))

    np.testing.assert_allclose(below, -0.5 * state)
    assert np.all(above > below)


def test_adaptation_integrates_with_generic_rk4_and_respects_bounds():
    distance, fov = make_domains()
    adaptation = AdaptiveDomainDynamics.from_domains(
        distance,
        fov,
        expansion_gain=10.0,
        recovery_gain=0.0,
        slack_threshold=0.0,
    )
    integrator = RK4Integrator()
    state = adaptation.initialize()

    for _ in range(10):
        state = integrator.step(
            adaptation,
            state,
            np.array([10.0]),
            dt=0.1,
        )

    maximum = adaptation.maximum_state.as_array()
    assert np.all(state >= 0.0)
    assert np.all(state <= maximum + 1e-12)
    np.testing.assert_allclose(state, maximum, atol=1e-12)
