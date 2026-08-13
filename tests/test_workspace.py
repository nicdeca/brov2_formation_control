import numpy as np

from formation_control.workspace import (
    AxisAlignedWorkspaceDomain,
    WorkspaceBarrierPotential,
    WorkspaceRelaxationPolicy,
)


def make_domain() -> AxisAlignedWorkspaceDomain:
    return AxisAlignedWorkspaceDomain(
        physical_lower=np.array([-3.125, -1.225, -96.58]),
        physical_upper=np.array([0.825, 5.575, -94.20]),
        conservative_lower=np.array([-2.975, -1.075, -96.38]),
        conservative_upper=np.array([0.675, 5.425, -94.75]),
    )


def test_full_relaxation_recovers_physical_box():
    domain = make_domain()
    lower, upper = domain.adaptive_bounds(np.ones(6))
    np.testing.assert_allclose(lower, domain.physical_lower)
    np.testing.assert_allclose(upper, domain.physical_upper)


def test_recentered_workspace_barrier_is_zero_at_reference():
    domain = make_domain()
    potential = WorkspaceBarrierPotential(domain, weight=0.1)
    reference = np.array([-1.9, 1.15, -95.2])
    evaluation = potential.evaluate(reference, reference, np.zeros(6))
    assert abs(evaluation.value) < 1e-12
    np.testing.assert_allclose(evaluation.position_gradient, 0.0, atol=1e-12)


def test_workspace_gradient_matches_finite_difference():
    domain = make_domain()
    potential = WorkspaceBarrierPotential(domain, weight=0.1)
    position = np.array([-2.4, 2.0, -95.4])
    reference = np.array([-1.9, 1.15, -95.2])
    evaluation = potential.evaluate(position, reference, np.zeros(6))

    epsilon = 1e-6
    numerical = np.zeros(3)
    for axis in range(3):
        plus = position.copy()
        minus = position.copy()
        plus[axis] += epsilon
        minus[axis] -= epsilon
        numerical[axis] = (
            potential.evaluate(plus, reference, np.zeros(6)).value
            - potential.evaluate(minus, reference, np.zeros(6)).value
        ) / (2.0 * epsilon)

    np.testing.assert_allclose(
        evaluation.position_gradient,
        numerical,
        rtol=2e-5,
        atol=2e-6,
    )


def test_relaxation_projects_state_when_outside_conservative_box():
    domain = make_domain()
    policy = WorkspaceRelaxationPolicy(domain.maximum_enlargement)
    position = np.array([-3.02, 1.0, -95.0])
    projected, changed = policy.project_to_current_domain(
        np.zeros(6),
        domain.conservative_values(position),
    )
    assert changed
    assert 0.0 < projected[0] < 1.0
    assert np.all(projected[1:] == 0.0)
