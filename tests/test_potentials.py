import numpy as np
import pytest

from formation_control.constraints import (
    HorizontalFieldOfViewConstraint,
    MinimumDistanceConstraint,
    NormalizedImagePoint,
)
from formation_control.potentials import (
    ConstraintBarrierPotential,
    ImageCenteringPotential,
    RecenteredLogBarrier,
    RelativePositionPotential,
    SumPotential,
    WeightedPotential,
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


def test_recentered_log_barrier_is_zero_at_reference():
    barrier = RecenteredLogBarrier(reference_value=2.0)

    assert barrier.value(2.0) == pytest.approx(0.0)
    assert barrier.derivative(2.0) == pytest.approx(0.0)


@pytest.mark.parametrize("constraint_value", [0.1, 0.5, 1.0, 3.0, 5.0])
def test_recentered_log_barrier_is_nonnegative(constraint_value):
    barrier = RecenteredLogBarrier(reference_value=2.0)

    assert barrier.value(constraint_value) >= -1e-14


def test_recentered_log_barrier_grows_toward_boundary():
    barrier = RecenteredLogBarrier(reference_value=1.0)

    assert barrier.value(1e-8) > barrier.value(1e-4)
    assert barrier.value(1e-4) > barrier.value(1e-2)


def test_recentered_log_barrier_rejects_nonpositive_constraint_values():
    barrier = RecenteredLogBarrier(reference_value=1.0)

    with pytest.raises(ValueError):
        barrier.value(0.0)

    with pytest.raises(ValueError):
        barrier.derivative(-1.0)


def test_distance_barrier_depends_only_on_distance():
    constraint = MinimumDistanceConstraint(minimum_distance=0.5)
    reference = np.array([1.5, 0.0, 0.0])
    potential = ConstraintBarrierPotential.from_reference(
        constraint,
        reference,
    )

    rotated_reference = np.array([0.0, 1.5, 0.0])

    assert potential.evaluate(reference).value == pytest.approx(0.0)
    assert potential.evaluate(rotated_reference).value == pytest.approx(0.0)


def test_distance_barrier_gradient_matches_finite_difference():
    constraint = MinimumDistanceConstraint(minimum_distance=0.5)
    reference = np.array([1.5, 0.0, 0.0])
    potential = ConstraintBarrierPotential.from_reference(
        constraint,
        reference,
        weight=2.3,
    )
    relative_position = np.array([1.2, 0.4, -0.2])

    analytic = potential.evaluate(relative_position).gradient
    numeric = finite_difference_gradient(
        lambda value: potential.evaluate(value).value,
        relative_position,
    )

    np.testing.assert_allclose(analytic, numeric, atol=1e-7)


def test_constraint_barrier_has_zero_gradient_at_reference():
    constraint = MinimumDistanceConstraint(minimum_distance=0.5)
    reference = np.array([1.5, -0.2, 0.4])
    potential = ConstraintBarrierPotential.from_reference(
        constraint,
        reference,
    )

    evaluation = potential.evaluate(reference)

    assert evaluation.value == pytest.approx(0.0)
    np.testing.assert_allclose(evaluation.gradient, np.zeros(3), atol=1e-14)


def test_relative_position_potential():
    desired = np.array([1.0, -2.0, 0.5])
    potential = RelativePositionPotential.isotropic(desired, gain=3.0)

    evaluation_desired = potential.evaluate(desired)
    assert evaluation_desired.value == pytest.approx(0.0)
    np.testing.assert_allclose(evaluation_desired.gradient, np.zeros(3))

    relative_position = np.array([2.0, -1.0, 0.0])
    error = relative_position - desired
    evaluation = potential.evaluate(relative_position)

    assert evaluation.value == pytest.approx(1.5 * float(error @ error))
    np.testing.assert_allclose(evaluation.gradient, 3.0 * error)


def test_image_centering_potential():
    potential = ImageCenteringPotential(
        alpha_h_desired=0.1,
        alpha_v_desired=-0.2,
        horizontal_gain=2.0,
        vertical_gain=3.0,
    )
    point = NormalizedImagePoint(alpha_h=0.4, alpha_v=0.2)

    evaluation = potential.evaluate(point)

    assert evaluation.value == pytest.approx(0.5 * (2.0 * 0.3**2 + 3.0 * 0.4**2))
    np.testing.assert_allclose(
        evaluation.gradient,
        np.array([0.6, 1.2]),
    )


def test_fov_barrier_recentered_at_image_center_has_zero_gradient():
    constraint = HorizontalFieldOfViewConstraint(alpha_limit=0.8)
    reference = NormalizedImagePoint(alpha_h=0.0, alpha_v=0.0)
    potential = ConstraintBarrierPotential.from_reference(
        constraint,
        reference,
    )

    evaluation = potential.evaluate(reference)

    assert evaluation.value == pytest.approx(0.0)
    np.testing.assert_allclose(evaluation.gradient, np.zeros(2))


def test_weighted_and_sum_potentials():
    desired = np.zeros(3)
    first = RelativePositionPotential.isotropic(desired, gain=1.0)
    second = WeightedPotential(
        RelativePositionPotential.isotropic(desired, gain=2.0),
        weight=0.5,
    )
    total = SumPotential([first, second])
    point = np.array([1.0, -2.0, 0.5])

    evaluation = total.evaluate(point)

    # Both terms are equivalent to gain I after weighting.
    assert evaluation.value == pytest.approx(float(point @ point))
    np.testing.assert_allclose(evaluation.gradient, 2.0 * point)
