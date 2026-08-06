import numpy as np

from formation_control.constraints import MinimumDistanceConstraint
from formation_control.potentials import AdaptiveConstraintBarrierPotential, EdgePotential


def test_edge_potential_exposes_adaptive_enlargement_derivative():
    reference = np.array([2.0, 0.0, 0.0])
    template = AdaptiveConstraintBarrierPotential(
        constraint=MinimumDistanceConstraint(0.8),
        reference_state=reference,
        weight=0.3,
    )
    bound = template.bind(0.1)
    potential = EdgePotential(collision_barrier=bound)

    observer = np.zeros(3)
    target = np.array([1.6, 0.0, 0.0])
    rotation = np.eye(3)
    evaluation = potential.evaluate(observer, rotation, target)

    expected = bound.enlargement_derivative(target - observer)
    assert evaluation.enlargement_derivatives["collision_barrier"] == expected
