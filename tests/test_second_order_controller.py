import numpy as np
import pytest

pytest.importorskip("qpsolvers")
pytest.importorskip("osqp")

from formation_control.control import (  # noqa: E402
    CLFQP,
    BacksteppingCLF,
    FirstOrderCommandFilter,
    LinearClassK,
    PolyhedralControlSet,
    SecondOrderCLFQPController,
)


def build_controller():
    return SecondOrderCLFQPController(
        virtual_gain=2.0 * np.eye(3),
        command_filter=FirstOrderCommandFilter(
            signal_dim=3,
            bandwidth=4.0,
        ),
        clf=BacksteppingCLF(np.eye(3)),
        qp=CLFQP.isotropic(
            control_dim=3,
            control_weight=1.0,
            slack_penalty=1e4,
            alpha=LinearClassK(gain=2.0),
            control_set=PolyhedralControlSet.box(
                -10.0,
                10.0,
                dimension=3,
            ),
        ),
    )


def test_filter_initialization_uses_virtual_command():
    controller = build_controller()
    gradient = np.array([1.0, -0.5, 0.2])

    desired = controller.desired_velocity(gradient)
    state = controller.initialize_filter(gradient)

    np.testing.assert_allclose(desired, -2.0 * gradient)
    np.testing.assert_allclose(state, desired)


def test_evaluation_does_not_advance_filter_state():
    controller = build_controller()
    gradient = np.array([0.5, -0.2, 0.1])
    filter_state = controller.initialize_filter(gradient)
    filter_state_before = filter_state.copy()

    evaluation = controller.evaluate(
        configuration_value=0.5,
        configuration_gradient=gradient,
        generalized_velocity=np.zeros(3),
        filter_state=filter_state,
        dynamics_bias=np.zeros(3),
    )

    np.testing.assert_allclose(filter_state, filter_state_before)
    np.testing.assert_allclose(
        evaluation.filter.output,
        filter_state_before,
    )


def test_feedforward_is_added_before_filtering():
    controller = build_controller()
    gradient = np.array([0.5, -0.2, 0.1])
    feedforward = np.array([0.3, 0.4, -0.1])

    desired = controller.desired_velocity(
        gradient,
        feedforward_velocity=feedforward,
    )

    np.testing.assert_allclose(
        desired,
        feedforward - 2.0 * gradient,
    )
