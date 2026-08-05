import numpy as np
import pytest

from formation_control.constraints import (
    DistanceDomain,
    FieldOfViewDomain,
    HorizontalFieldOfViewConstraint,
    MaximumDistanceConstraint,
    MinimumDistanceConstraint,
    NormalizedImagePoint,
    PositiveDepthConstraint,
    VerticalFieldOfViewConstraint,
)


def finite_difference_gradient(function, value, epsilon=1e-7):
    value = np.asarray(value, dtype=float)
    gradient = np.zeros_like(value)
    for index in range(value.size):
        perturbation = np.zeros_like(value)
        perturbation[index] = epsilon
        gradient[index] = (function(value + perturbation) - function(value - perturbation)) / (
            2.0 * epsilon
        )
    return gradient


def test_distance_domain_enlargement_matches_physical_limits():
    domain = DistanceDomain(
        d_min=0.5,
        d_max=5.0,
        d_min_conservative=0.8,
        d_max_conservative=4.0,
    )

    assert domain.collision_enlargement_max == pytest.approx(0.8**2 - 0.5**2)
    assert domain.range_enlargement_max == pytest.approx(5.0**2 - 4.0**2)


def test_distance_domain_requires_strictly_nested_intervals():
    with pytest.raises(ValueError):
        DistanceDomain(
            d_min=0.5,
            d_max=5.0,
            d_min_conservative=0.5,
            d_max_conservative=4.0,
        )


def test_fov_domain_enlargement_terminates_at_physical_fov():
    domain = FieldOfViewDomain(
        alpha_h_conservative=0.8,
        alpha_v_conservative=0.7,
    )

    assert domain.horizontal_enlargement_max == pytest.approx(1.0 - 0.8**2)
    assert domain.vertical_enlargement_max == pytest.approx(1.0 - 0.7**2)


@pytest.mark.parametrize(
    ("constraint", "relative_position"),
    [
        (MinimumDistanceConstraint(0.5), np.array([1.2, -0.4, 0.3])),
        (MaximumDistanceConstraint(3.0), np.array([1.2, -0.4, 0.3])),
    ],
)
def test_distance_constraint_gradient_matches_finite_difference(constraint, relative_position):
    analytic = constraint.evaluate(relative_position).gradient
    numeric = finite_difference_gradient(constraint.value, relative_position)
    np.testing.assert_allclose(analytic, numeric, atol=1e-7)


def test_distance_constraints_use_strict_admissibility():
    relative_position = np.array([1.0, 0.0, 0.0])

    minimum = MinimumDistanceConstraint(1.0)
    maximum = MaximumDistanceConstraint(1.0)

    assert minimum.value(relative_position) == pytest.approx(0.0)
    assert maximum.value(relative_position) == pytest.approx(0.0)
    assert not minimum.is_satisfied(relative_position)
    assert not maximum.is_satisfied(relative_position)


def test_normalized_fov_constraints():
    point = NormalizedImagePoint(alpha_h=0.4, alpha_v=-0.3)
    horizontal = HorizontalFieldOfViewConstraint(alpha_limit=0.8)
    vertical = VerticalFieldOfViewConstraint(alpha_limit=0.7)

    assert horizontal.value(point) == pytest.approx(0.8**2 - 0.4**2)
    assert vertical.value(point) == pytest.approx(0.7**2 - 0.3**2)
    np.testing.assert_allclose(
        horizontal.evaluate(point).gradient,
        np.array([-0.8, 0.0]),
    )
    np.testing.assert_allclose(
        vertical.evaluate(point).gradient,
        np.array([0.0, 0.6]),
    )


def test_positive_depth_constraint():
    constraint = PositiveDepthConstraint()
    point = np.array([2.0, -0.1, 0.5])

    assert constraint.is_satisfied(point)
    np.testing.assert_allclose(
        constraint.evaluate(point).gradient,
        np.array([1.0, 0.0, 0.0]),
    )
