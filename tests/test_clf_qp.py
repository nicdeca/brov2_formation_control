import numpy as np
import pytest

pytest.importorskip("qpsolvers")
pytest.importorskip("osqp")

from formation_control.control import (  # noqa: E402
    CLFQP,
    BacksteppingCLFEvaluation,
    LinearClassK,
    PolyhedralControlSet,
)


def make_clf_evaluation(
    *,
    value=1.0,
    drift=1.0,
    control_gradient=None,
):
    if control_gradient is None:
        control_gradient = np.array([1.0])

    return BacksteppingCLFEvaluation(
        value=value,
        configuration_value=value,
        velocity_error_value=0.0,
        velocity_error=np.zeros(control_gradient.size),
        drift=drift,
        control_gradient=np.asarray(control_gradient, dtype=float),
    )


def test_clf_qp_problem_assembly():
    clf = make_clf_evaluation(
        value=2.0,
        drift=0.5,
        control_gradient=np.array([1.0, -2.0]),
    )
    control_set = PolyhedralControlSet(
        dimension=2,
        lower=np.array([-1.0, -2.0]),
        upper=np.array([3.0, 4.0]),
        inequality_matrix=np.array([[1.0, 1.0]]),
        inequality_bound=np.array([2.0]),
    )
    qp = CLFQP(
        control_weight=np.diag([2.0, 3.0]),
        slack_penalty=100.0,
        alpha=LinearClassK(gain=1.5),
        control_set=control_set,
    )

    problem = qp.build_problem(clf)

    np.testing.assert_allclose(
        problem.quadratic_cost,
        np.diag([2.0, 3.0, 100.0]),
    )
    np.testing.assert_allclose(problem.linear_cost, np.zeros(3))
    np.testing.assert_allclose(
        problem.inequality_matrix,
        np.array(
            [
                [1.0, -2.0, -1.0],
                [1.0, 1.0, 0.0],
            ]
        ),
    )
    np.testing.assert_allclose(
        problem.inequality_bound,
        np.array([-3.5, 2.0]),
    )
    np.testing.assert_allclose(
        problem.lower_bounds,
        np.array([-1.0, -2.0, 0.0]),
    )
    np.testing.assert_allclose(
        problem.upper_bounds,
        np.array([3.0, 4.0, np.inf]),
    )


def test_unconstrained_qp_matches_analytic_solution():
    # min 1/2 u^2 + 1/2 p delta^2
    # s.t. u - delta <= -2.
    # KKT gives u = -2p/(1+p), delta = 2/(1+p).
    penalty = 9.0
    clf = make_clf_evaluation(
        value=1.0,
        drift=1.0,
        control_gradient=np.array([1.0]),
    )
    qp = CLFQP.isotropic(
        control_dim=1,
        control_weight=1.0,
        slack_penalty=penalty,
        alpha=LinearClassK(gain=1.0),
    )

    result = qp.solve(clf)

    assert result.control[0] == pytest.approx(
        -2.0 * penalty / (1.0 + penalty),
        abs=2e-4,
    )
    assert result.slack == pytest.approx(
        2.0 / (1.0 + penalty),
        abs=2e-4,
    )
    assert result.clf_residual <= 2e-4


def test_actuator_limit_is_respected_and_slack_absorbs_infeasibility():
    clf = make_clf_evaluation(
        value=1.0,
        drift=2.0,
        control_gradient=np.array([1.0]),
    )
    control_set = PolyhedralControlSet.box(
        -0.5,
        0.5,
        dimension=1,
    )
    qp = CLFQP.isotropic(
        control_dim=1,
        control_weight=1.0,
        slack_penalty=1e4,
        alpha=LinearClassK(gain=1.0),
        control_set=control_set,
    )

    result = qp.solve(clf)

    assert control_set.contains(result.control, tolerance=1e-5)
    assert result.control[0] == pytest.approx(-0.5, abs=2e-4)
    assert result.slack == pytest.approx(2.5, abs=3e-4)
    assert result.clf_residual <= 3e-4


def test_zero_control_gradient_uses_only_slack():
    clf = make_clf_evaluation(
        value=1.0,
        drift=0.5,
        control_gradient=np.array([0.0, 0.0]),
    )
    qp = CLFQP.isotropic(
        control_dim=2,
        control_weight=1.0,
        slack_penalty=100.0,
        alpha=LinearClassK(gain=1.0),
    )

    result = qp.solve(clf)

    np.testing.assert_allclose(result.control, np.zeros(2), atol=1e-7)
    assert result.slack == pytest.approx(1.5, abs=2e-4)


def test_polyhedral_control_constraint_is_respected():
    clf = make_clf_evaluation(
        value=0.5,
        drift=0.5,
        control_gradient=np.array([-1.0, -1.0]),
    )
    control_set = PolyhedralControlSet(
        dimension=2,
        inequality_matrix=np.array([[1.0, 1.0]]),
        inequality_bound=np.array([0.4]),
    )
    qp = CLFQP.isotropic(
        control_dim=2,
        control_weight=1.0,
        slack_penalty=1e4,
        alpha=LinearClassK(gain=1.0),
        control_set=control_set,
    )

    result = qp.solve(clf)

    assert control_set.contains(result.control, tolerance=1e-5)
    assert result.clf_residual <= 3e-4
