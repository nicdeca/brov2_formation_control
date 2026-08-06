import numpy as np
import pytest

from formation_control.control import CLFQP, LinearClassK, PolyhedralControlSet
from formation_control.control.backstepping_clf import BacksteppingCLFEvaluation


def make_clf(value, drift, gradient):
    gradient = np.asarray(gradient, dtype=float)
    return BacksteppingCLFEvaluation(
        value=value,
        configuration_value=value,
        velocity_error_value=0.0,
        velocity_error=np.zeros(1),
        drift=drift,
        control_gradient=gradient,
    )


def test_reference_is_used_when_clf_is_inactive():
    qp = CLFQP.isotropic(
        control_dim=1,
        control_weight=2.0,
        slack_penalty=100.0,
        alpha=LinearClassK(gain=1.0),
        control_set=PolyhedralControlSet.box(-1.0, 1.0, dimension=1),
    )
    clf = make_clf(0.2, -2.0, np.array([1.0]))

    result = qp.solve(clf, control_reference=np.array([0.35]))

    assert result.control[0] == pytest.approx(0.35, abs=1e-12)
    assert result.slack == pytest.approx(0.0, abs=1e-12)


def test_reference_active_solution_matches_scalar_kkt_formula():
    penalty = 10.0
    reference = 0.4
    qp = CLFQP.isotropic(
        control_dim=1,
        control_weight=1.0,
        slack_penalty=penalty,
        alpha=LinearClassK(gain=1.0),
    )
    clf = make_clf(1.0, 0.5, np.array([1.0]))

    result = qp.solve(clf, control_reference=np.array([reference]))

    constant = 1.5
    multiplier = (constant + reference) / (1.0 + 1.0 / penalty)
    expected_control = reference - multiplier
    expected_slack = multiplier / penalty

    assert result.control[0] == pytest.approx(expected_control, abs=1e-10)
    assert result.slack == pytest.approx(expected_slack, abs=1e-10)


def test_dynamic_bounds_are_intersected_with_static_box():
    qp = CLFQP.isotropic(
        control_dim=1,
        control_weight=1.0,
        slack_penalty=100.0,
        alpha=LinearClassK(gain=1.0),
        control_set=PolyhedralControlSet.box(-2.0, 2.0, dimension=1),
    )
    clf = make_clf(0.1, -2.0, np.array([0.0]))

    result = qp.solve(
        clf,
        control_reference=np.array([1.5]),
        upper_override=np.array([0.3]),
    )

    assert result.control[0] == pytest.approx(0.3, abs=1e-12)


def test_linear_slack_penalty_creates_zero_slack_region():
    p1 = 2.0
    p2 = 10.0
    qp = CLFQP.isotropic(
        control_dim=1,
        control_weight=1.0,
        slack_penalty=p2,
        slack_linear_penalty=p1,
        alpha=LinearClassK(gain=1.0),
    )
    clf = make_clf(0.5, 0.5, np.array([1.0]))

    result = qp.solve(clf)

    # Hard zero-slack optimum is u = -1.  Its CLF multiplier is lambda = 1,
    # which is below p1, so the exact linear penalty keeps delta exactly zero.
    assert result.control[0] == pytest.approx(-1.0, abs=1e-10)
    assert result.slack == pytest.approx(0.0, abs=1e-12)


def test_linear_slack_penalty_direct_solver_matches_kkt_formula():
    p1 = 1.0
    p2 = 10.0
    qp = CLFQP.isotropic(
        control_dim=1,
        control_weight=1.0,
        slack_penalty=p2,
        slack_linear_penalty=p1,
        alpha=LinearClassK(gain=1.0),
    )
    clf = make_clf(2.0, 1.0, np.array([1.0]))

    result = qp.solve(clf)

    # Active unconstrained KKT equations:
    # u = -lambda, delta = (lambda-p1)/p2,
    # 3 - lambda - (lambda-p1)/p2 = 0.
    multiplier = (3.0 + p1 / p2) / (1.0 + 1.0 / p2)
    expected_slack = (multiplier - p1) / p2

    assert result.control[0] == pytest.approx(-multiplier, abs=1e-10)
    assert result.slack == pytest.approx(expected_slack, abs=1e-10)
