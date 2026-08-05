import numpy as np
import pytest

from formation_control.constraints import (
    DistanceDomain,
    MinimumDistanceConstraint,
)
from formation_control.potentials import (
    AdaptiveConstraintBarrierPotential,
    ConstraintBarrierPotential,
)


def test_adaptive_barrier_remains_recentered_for_every_enlargement():
    constraint = MinimumDistanceConstraint(0.8)
    reference = np.array([1.5, 0.0, 0.0])
    potential = AdaptiveConstraintBarrierPotential(
        constraint=constraint,
        reference_state=reference,
        weight=2.0,
    )

    for enlargement in (0.0, 0.1, 0.3):
        evaluation = potential.evaluate(reference, enlargement)
        assert evaluation.value == pytest.approx(0.0, abs=1e-14)
        np.testing.assert_allclose(
            evaluation.gradient,
            np.zeros(3),
            atol=1e-14,
        )
        assert evaluation.enlargement_derivative == pytest.approx(0.0)


def test_adaptive_barrier_matches_physical_barrier_at_maximum_enlargement():
    domain = DistanceDomain(
        d_min=0.5,
        d_max=5.0,
        d_min_conservative=0.8,
        d_max_conservative=4.0,
    )
    reference = np.array([1.5, 0.0, 0.0])
    state = np.array([1.2, 0.3, -0.1])

    adaptive = AdaptiveConstraintBarrierPotential(
        constraint=MinimumDistanceConstraint(domain.d_min_conservative),
        reference_state=reference,
        weight=1.7,
    )
    physical = ConstraintBarrierPotential.from_reference(
        MinimumDistanceConstraint(domain.d_min),
        reference,
        weight=1.7,
    )

    adaptive_evaluation = adaptive.evaluate(
        state,
        domain.collision_enlargement_max,
    )
    physical_evaluation = physical.evaluate(state)

    assert adaptive_evaluation.value == pytest.approx(physical_evaluation.value)
    np.testing.assert_allclose(
        adaptive_evaluation.gradient,
        physical_evaluation.gradient,
        atol=1e-12,
    )


def test_adaptive_barrier_enlargement_derivative_matches_finite_difference():
    potential = AdaptiveConstraintBarrierPotential(
        constraint=MinimumDistanceConstraint(0.8),
        reference_state=np.array([1.5, 0.0, 0.0]),
        weight=1.4,
    )
    state = np.array([1.0, 0.2, -0.1])
    enlargement = 0.15
    epsilon = 1e-7

    analytic = potential.evaluate(
        state,
        enlargement,
    ).enlargement_derivative
    numeric = (
        potential.evaluate(state, enlargement + epsilon).value
        - potential.evaluate(state, enlargement - epsilon).value
    ) / (2.0 * epsilon)

    assert analytic == pytest.approx(numeric, abs=1e-8)
    assert analytic <= 0.0


def test_bound_adaptive_barrier_works_as_standard_potential():
    potential = AdaptiveConstraintBarrierPotential(
        constraint=MinimumDistanceConstraint(0.8),
        reference_state=np.array([1.5, 0.0, 0.0]),
    )
    bound = potential.bind(0.2)
    state = np.array([1.1, 0.2, 0.0])

    bound_evaluation = bound.evaluate(state)
    adaptive_evaluation = potential.evaluate(state, 0.2)

    assert bound_evaluation.value == pytest.approx(adaptive_evaluation.value)
    np.testing.assert_allclose(
        bound_evaluation.gradient,
        adaptive_evaluation.gradient,
    )
    assert bound.enlargement_derivative(state) == pytest.approx(
        adaptive_evaluation.enlargement_derivative
    )
