"""Tests for exact feedforward handling in the second-order controller."""

import numpy as np

from formation_control.control import (
    CLFQP,
    BacksteppingCLF,
    FirstOrderCommandFilter,
    LinearClassK,
    PolyhedralControlSet,
    SecondOrderCLFQPController,
)


def make_controller() -> SecondOrderCLFQPController:
    return SecondOrderCLFQPController(
        virtual_gain=np.eye(3),
        command_filter=FirstOrderCommandFilter(
            signal_dim=3,
            bandwidth=4.0,
        ),
        clf=BacksteppingCLF(np.eye(3)),
        qp=CLFQP.isotropic(
            control_dim=3,
            control_weight=1.0,
            slack_penalty=1e4,
            alpha=LinearClassK(gain=1.0),
            control_set=PolyhedralControlSet.box(
                -100.0,
                100.0,
                dimension=3,
            ),
        ),
    )


def test_split_feedforward_uses_exact_velocity_and_derivative_at_zero_error():
    controller = make_controller()
    gradient = np.zeros(3)
    filter_state = controller.initialize_feedback_filter(gradient)

    feedforward = np.array([0.4, -0.2, 0.1])
    feedforward_derivative = np.array([0.05, 0.02, -0.03])
    evaluation = controller.evaluate_with_feedforward_derivative(
        configuration_value=0.0,
        configuration_gradient=gradient,
        generalized_velocity=feedforward,
        filter_state=filter_state,
        dynamics_bias=np.zeros(3),
        feedforward_velocity=feedforward,
        feedforward_velocity_derivative=feedforward_derivative,
    )

    np.testing.assert_allclose(
        evaluation.filter.output,
        feedforward,
    )
    np.testing.assert_allclose(
        evaluation.filter.output_derivative,
        feedforward_derivative,
    )
    np.testing.assert_allclose(
        evaluation.filter_command,
        np.zeros(3),
    )


def test_standard_evaluation_filter_command_remains_full_desired_velocity():
    controller = make_controller()
    gradient = np.array([0.2, -0.1, 0.05])
    filter_state = controller.initialize_filter(gradient)

    evaluation = controller.evaluate(
        configuration_value=0.1,
        configuration_gradient=gradient,
        generalized_velocity=np.zeros(3),
        filter_state=filter_state,
        dynamics_bias=np.zeros(3),
    )

    np.testing.assert_allclose(
        evaluation.filter_command,
        evaluation.desired_velocity,
    )
