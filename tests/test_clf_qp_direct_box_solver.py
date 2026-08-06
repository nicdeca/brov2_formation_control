import numpy as np
import pytest

from formation_control.control import (
    CLFQP,
    LinearClassK,
    PolyhedralControlSet,
)
from formation_control.control.backstepping_clf import (
    BacksteppingCLFEvaluation,
)


def make_clf(
    *,
    value: float,
    drift: float,
    gradient: np.ndarray,
) -> BacksteppingCLFEvaluation:
    gradient = np.asarray(gradient, dtype=float)
    return BacksteppingCLFEvaluation(
        value=value,
        configuration_value=value,
        velocity_error_value=0.0,
        velocity_error=np.zeros(gradient.size),
        drift=drift,
        control_gradient=gradient,
    )


def test_diagonal_box_qp_uses_direct_solver():
    qp = CLFQP.isotropic(
        control_dim=2,
        control_weight=1.0,
        slack_penalty=100.0,
        alpha=LinearClassK(gain=1.0),
        control_set=PolyhedralControlSet.box(
            -1.0,
            1.0,
            dimension=2,
        ),
    )

    assert qp.uses_direct_box_solver


def test_general_polyhedral_qp_keeps_external_solver_path():
    qp = CLFQP.isotropic(
        control_dim=2,
        control_weight=1.0,
        slack_penalty=100.0,
        alpha=LinearClassK(gain=1.0),
        control_set=PolyhedralControlSet(
            dimension=2,
            inequality_matrix=np.array([[1.0, 1.0]]),
            inequality_bound=np.array([1.0]),
        ),
    )

    assert not qp.uses_direct_box_solver


def test_direct_solver_matches_unconstrained_kkt_solution():
    penalty = 9.0
    qp = CLFQP.isotropic(
        control_dim=1,
        control_weight=1.0,
        slack_penalty=penalty,
        alpha=LinearClassK(gain=1.0),
    )
    clf = make_clf(
        value=1.0,
        drift=1.0,
        gradient=np.array([1.0]),
    )

    result = qp.solve(clf)

    assert result.control[0] == pytest.approx(
        -2.0 * penalty / (1.0 + penalty),
        abs=1e-10,
    )
    assert result.slack == pytest.approx(
        2.0 / (1.0 + penalty),
        abs=1e-10,
    )


def test_direct_solver_handles_active_thruster_bound():
    qp = CLFQP.isotropic(
        control_dim=1,
        control_weight=1.0,
        slack_penalty=1e4,
        alpha=LinearClassK(gain=1.0),
        control_set=PolyhedralControlSet.box(
            -0.5,
            0.5,
            dimension=1,
        ),
    )
    clf = make_clf(
        value=1.0,
        drift=2.0,
        gradient=np.array([1.0]),
    )

    result = qp.solve(clf)

    assert result.control[0] == pytest.approx(-0.5, abs=1e-10)
    assert result.slack == pytest.approx(2.5, abs=1e-8)
    assert result.clf_residual <= 1e-8


def test_direct_solver_remains_finite_for_large_barrier_coefficients():
    lower = np.array([-18.0] * 8)
    upper = np.array([18.0] * 8)
    qp = CLFQP(
        control_weight=(1.0 / 18.0**2) * np.eye(8),
        slack_penalty=5e3,
        alpha=LinearClassK(gain=0.8),
        control_set=PolyhedralControlSet(
            dimension=8,
            lower=lower,
            upper=upper,
        ),
    )
    clf = make_clf(
        value=2.0e8,
        drift=3.0e10,
        gradient=np.array([1.0e9, -8.0e8, 6.0e8, -5.0e8, 3.0e8, -2.0e8, 1.0e8, -5.0e7]),
    )

    result = qp.solve(clf)

    assert np.all(np.isfinite(result.control))
    assert np.isfinite(result.slack)
    assert np.all(result.control >= lower)
    assert np.all(result.control <= upper)
    assert result.clf_residual <= 1e-3


def test_zero_control_gradient_is_solved_exactly_by_slack():
    qp = CLFQP.isotropic(
        control_dim=2,
        control_weight=1.0,
        slack_penalty=100.0,
        alpha=LinearClassK(gain=1.0),
    )
    clf = make_clf(
        value=1.0,
        drift=0.5,
        gradient=np.zeros(2),
    )

    result = qp.solve(clf)

    np.testing.assert_allclose(result.control, np.zeros(2), atol=0.0)
    assert result.slack == pytest.approx(1.5, abs=1e-12)
