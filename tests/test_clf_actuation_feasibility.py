import numpy as np
import pytest

from formation_control.control import (
    CLFQP,
    LinearClassK,
    PolyhedralControlSet,
    box_clf_actuation_feasibility,
)
from formation_control.control.backstepping_clf import BacksteppingCLFEvaluation


def make_clf(*, drift: float, gradient: np.ndarray, value: float = 1.0):
    return BacksteppingCLFEvaluation(
        value=value,
        configuration_value=value,
        velocity_error_value=0.0,
        velocity_error=np.zeros(gradient.size),
        drift=drift,
        control_gradient=gradient,
    )


def test_box_feasibility_uses_bound_selected_by_gradient_sign():
    clf = make_clf(
        drift=0.2,
        gradient=np.array([2.0, -3.0, 0.0]),
    )
    control_set = PolyhedralControlSet.box(
        lower=np.array([-1.0, -2.0, -4.0]),
        upper=np.array([3.0, 5.0, 6.0]),
    )

    result = box_clf_actuation_feasibility(
        clf,
        alpha=LinearClassK(gain=0.5),
        control_set=control_set,
    )

    np.testing.assert_allclose(
        result.minimizing_control,
        np.array([-1.0, 5.0, 1.0]),
    )


def test_required_slack_is_zero_when_zero_slack_clf_is_feasible():
    clf = make_clf(
        drift=0.0,
        gradient=np.array([1.0]),
        value=1.0,
    )
    control_set = PolyhedralControlSet.box(
        lower=np.array([-2.0]),
        upper=np.array([2.0]),
    )

    result = box_clf_actuation_feasibility(
        clf,
        alpha=LinearClassK(gain=1.0),
        control_set=control_set,
    )

    # Minimum derivative = -2 <= -1.
    assert result.required_slack == pytest.approx(0.0)
    assert result.margin == pytest.approx(1.0)
    assert result.zero_slack_feasible


def test_required_slack_equals_actuation_deficit():
    clf = make_clf(
        drift=1.0,
        gradient=np.array([1.0]),
        value=2.0,
    )
    control_set = PolyhedralControlSet.box(
        lower=np.array([-0.5]),
        upper=np.array([0.5]),
    )

    result = box_clf_actuation_feasibility(
        clf,
        alpha=LinearClassK(gain=1.0),
        control_set=control_set,
    )

    # Minimum modeled derivative = 1 - 0.5 = 0.5, while the
    # zero-slack target is -2.  At least 2.5 units of relaxation are required.
    assert result.minimum_modeled_derivative == pytest.approx(0.5)
    assert result.required_slack == pytest.approx(2.5)
    assert result.margin == pytest.approx(-2.5)
    assert not result.zero_slack_feasible


def test_soft_qp_reports_required_slack_separately_from_optimal_slack():
    clf = make_clf(
        drift=0.0,
        gradient=np.array([1.0]),
        value=1.0,
    )
    qp = CLFQP(
        control_weight=np.eye(1),
        slack_penalty=1.0,
        alpha=LinearClassK(gain=1.0),
        control_set=PolyhedralControlSet.box(
            lower=np.array([-2.0]),
            upper=np.array([2.0]),
        ),
    )

    result = qp.solve(clf)

    # Zero-slack satisfaction is physically feasible (u <= -1), but with
    # equal control/slack weights the soft QP trades effort for positive slack.
    assert result.required_slack == pytest.approx(0.0)
    assert result.slack > 0.0


def test_polyhedral_control_set_skips_closed_form_feasibility():
    clf = make_clf(
        drift=0.0,
        gradient=np.array([1.0]),
        value=1.0,
    )
    qp = CLFQP(
        control_weight=np.eye(1),
        slack_penalty=10.0,
        alpha=LinearClassK(gain=1.0),
        control_set=PolyhedralControlSet.polyhedron(
            inequality_matrix=np.array([[1.0], [-1.0]]),
            inequality_bound=np.array([2.0, 2.0]),
        ),
    )

    assert qp.actuation_feasibility(clf) is None
