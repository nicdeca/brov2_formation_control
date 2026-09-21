import numpy as np
import pytest

from formation_control.constraints import (
    DistanceDomain,
    MaximumDistanceConstraint,
    MinimumDistanceConstraint,
)


def test_direct_distance_is_the_default():
    p = np.array([3.0, 4.0, 0.0])
    minimum = MinimumDistanceConstraint(2.0)
    maximum = MaximumDistanceConstraint(7.0)

    minimum_eval = minimum.evaluate(p)
    maximum_eval = maximum.evaluate(p)

    assert minimum_eval.value == pytest.approx(3.0)
    assert maximum_eval.value == pytest.approx(2.0)
    np.testing.assert_allclose(minimum_eval.gradient, np.array([0.6, 0.8, 0.0]))
    np.testing.assert_allclose(maximum_eval.gradient, -np.array([0.6, 0.8, 0.0]))


def test_squared_distance_legacy_mode_is_retained():
    p = np.array([3.0, 4.0, 0.0])
    minimum = MinimumDistanceConstraint(2.0, squared=True)
    maximum = MaximumDistanceConstraint(7.0, squared=True)

    minimum_eval = minimum.evaluate(p)
    maximum_eval = maximum.evaluate(p)

    assert minimum_eval.value == pytest.approx(21.0)
    assert maximum_eval.value == pytest.approx(24.0)
    np.testing.assert_allclose(minimum_eval.gradient, 2.0 * p)
    np.testing.assert_allclose(maximum_eval.gradient, -2.0 * p)


def test_direct_distance_domain_enlargement_matches_paper():
    domain = DistanceDomain(
        d_min=0.5,
        d_max=3.6,
        d_min_conservative=0.8,
        d_max_conservative=3.0,
    )
    assert domain.collision_enlargement_max == pytest.approx(0.3)
    assert domain.range_enlargement_max == pytest.approx(0.6)
    assert domain.effective_minimum_distance(0.3) == pytest.approx(0.5)
    assert domain.effective_maximum_distance(0.6) == pytest.approx(3.6)


def test_squared_distance_domain_enlargement_matches_legacy_code():
    domain = DistanceDomain(
        d_min=0.5,
        d_max=3.6,
        d_min_conservative=0.8,
        d_max_conservative=3.0,
        squared=True,
    )
    assert domain.collision_enlargement_max == pytest.approx(0.8**2 - 0.5**2)
    assert domain.range_enlargement_max == pytest.approx(3.6**2 - 3.0**2)
    assert domain.effective_minimum_distance(
        domain.collision_enlargement_max
    ) == pytest.approx(0.5)
    assert domain.effective_maximum_distance(
        domain.range_enlargement_max
    ) == pytest.approx(3.6)
