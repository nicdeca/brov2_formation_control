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


def _make_norm_saturated_controller(
    gain: np.ndarray,
    *,
    limits: np.ndarray | None = None,
) -> SecondOrderCLFQPController:
    if limits is None:
        limits = np.array([1.0, 2.0])

    return SecondOrderCLFQPController(
        virtual_gain=gain,
        command_filter=FirstOrderCommandFilter(
            signal_dim=6,
            bandwidth=4.0,
        ),
        clf=BacksteppingCLF(np.eye(6)),
        qp=CLFQP.isotropic(
            control_dim=6,
            control_weight=1.0,
            slack_penalty=1e4,
            alpha=LinearClassK(gain=2.0),
            control_set=PolyhedralControlSet.box(
                -10.0,
                10.0,
                dimension=6,
            ),
        ),
        virtual_velocity_norm_limits=limits,
        virtual_velocity_group_sizes=(3, 3),
    )


def test_smooth_norm_saturation_bounds_each_twist_block():
    controller = _make_norm_saturated_controller(np.diag([2.0, 3.0, 4.0, 2.0, 3.0, 4.0]))
    gradient = np.array([10.0, -10.0, 10.0, 8.0, -9.0, 7.0])

    desired = controller.desired_velocity(gradient)

    assert np.linalg.norm(desired[:3]) <= 1.0
    assert np.linalg.norm(desired[3:]) <= 2.0


def test_smooth_norm_saturation_preserves_block_directions():
    controller = _make_norm_saturated_controller(np.diag([2.0, 3.0, 4.0, 2.0, 3.0, 4.0]))
    gradient = np.array([2.0, -3.0, 1.0, 4.0, 1.0, -2.0])

    raw_correction = -controller.unlimited_desired_velocity(gradient)
    saturated_correction = -controller.desired_velocity(gradient)

    for block in (slice(0, 3), slice(3, 6)):
        raw = raw_correction[block]
        saturated = saturated_correction[block]
        cosine = float(raw @ saturated / (np.linalg.norm(raw) * np.linalg.norm(saturated)))
        assert cosine == pytest.approx(1.0)


def test_smooth_norm_saturation_is_locally_linear():
    gain = np.diag([2.0, 3.0, 4.0, 2.0, 3.0, 4.0])
    controller = _make_norm_saturated_controller(gain)
    gradient = np.array([1e-7, -2e-7, 1e-7, -1e-7, 2e-7, -1e-7])

    np.testing.assert_allclose(
        controller.desired_velocity(gradient),
        -gain @ gradient,
        rtol=1e-10,
        atol=1e-15,
    )


def test_norm_saturation_applies_only_to_feedback_correction():
    controller = _make_norm_saturated_controller(2.0 * np.eye(6))
    feedforward = np.array([0.4, -0.3, 0.2, 0.1, -0.2, 0.3])
    gradient = np.array([10.0, -10.0, 10.0, 8.0, -9.0, 7.0])

    desired = controller.desired_velocity(
        gradient,
        feedforward_velocity=feedforward,
    )
    correction = feedforward - desired

    assert np.linalg.norm(correction[:3]) <= 1.0
    assert np.linalg.norm(correction[3:]) <= 2.0


def test_norm_saturation_allows_full_spd_gain_within_each_block():
    translational_gain = np.array(
        [
            [2.0, 0.3, 0.1],
            [0.3, 2.5, 0.2],
            [0.1, 0.2, 3.0],
        ]
    )
    rotational_gain = np.array(
        [
            [1.5, 0.2, 0.1],
            [0.2, 2.0, 0.3],
            [0.1, 0.3, 2.5],
        ]
    )
    gain = np.block(
        [
            [translational_gain, np.zeros((3, 3))],
            [np.zeros((3, 3)), rotational_gain],
        ]
    )

    controller = _make_norm_saturated_controller(gain)
    gradient = np.array([1.0, -2.0, 0.5, -1.0, 0.5, 2.0])
    correction = -controller.desired_velocity(gradient)

    assert float(gradient @ correction) > 0.0


def test_norm_saturation_rejects_gain_coupling_between_blocks():
    gain = 2.0 * np.eye(6)
    gain[0, 3] = 0.1
    gain[3, 0] = 0.1

    with pytest.raises(ValueError, match="block diagonal"):
        _make_norm_saturated_controller(gain)


def test_second_order_qp_uses_velocity_error_dissipation_inequality():
    velocity_error_gain = np.diag([3.0, 4.0, 5.0])
    controller = SecondOrderCLFQPController(
        virtual_gain=2.0 * np.eye(3),
        command_filter=FirstOrderCommandFilter(signal_dim=3, bandwidth=4.0),
        clf=BacksteppingCLF(np.diag([2.0, 3.0, 4.0])),
        qp=CLFQP.isotropic(
            control_dim=3,
            control_weight=1.0,
            slack_penalty=1e4,
            alpha=LinearClassK(gain=99.0),
            control_set=PolyhedralControlSet.box(-50.0, 50.0, dimension=3),
        ),
        velocity_error_gain=velocity_error_gain,
    )

    gradient = np.array([0.2, -0.1, 0.3])
    filter_state = np.array([0.05, -0.02, 0.04])
    velocity = np.array([0.15, -0.12, 0.09])
    bias = np.array([0.4, -0.2, 0.1])

    evaluation = controller.evaluate(
        configuration_value=0.7,
        configuration_gradient=gradient,
        generalized_velocity=velocity,
        filter_state=filter_state,
        dynamics_bias=bias,
    )

    error = evaluation.clf.velocity_error
    expected_a = float(
        error
        @ (
            gradient
            - bias
            - controller.clf.inertia @ evaluation.filter.output_derivative
        )
    )
    expected_dissipation = float(error @ velocity_error_gain @ error)

    assert evaluation.constraint_drift == pytest.approx(expected_a)
    assert evaluation.dissipation_rate == pytest.approx(expected_dissipation)

    lhs = (
        evaluation.constraint_drift
        + evaluation.clf.control_gradient @ evaluation.control
    )
    rhs = -evaluation.dissipation_rate + evaluation.slack
    assert lhs <= rhs + 1e-7


def test_trim_reference_is_active_only_near_zero_control_lie_derivative():
    controller = SecondOrderCLFQPController(
        virtual_gain=np.eye(1),
        command_filter=FirstOrderCommandFilter(signal_dim=1, bandwidth=4.0),
        clf=BacksteppingCLF(np.eye(1)),
        qp=CLFQP.isotropic(
            control_dim=1,
            control_weight=1.0,
            slack_penalty=1e4,
            alpha=LinearClassK(gain=1.0),
            control_set=PolyhedralControlSet.box(-10.0, 10.0, dimension=1),
        ),
        velocity_error_gain=np.eye(1),
        trim_control_allocator=lambda wrench: np.asarray(wrench, dtype=float),
        trim_activation_on=0.1,
        trim_activation_off=0.5,
    )

    at_trim = controller.evaluate(
        configuration_value=0.0,
        configuration_gradient=np.zeros(1),
        generalized_velocity=np.zeros(1),
        filter_state=np.zeros(1),
        dynamics_bias=np.array([2.0]),
        trim_wrench=np.array([2.0]),
    )
    assert at_trim.trim_metric == pytest.approx(0.0)
    assert at_trim.trim_activation == pytest.approx(1.0)
    np.testing.assert_allclose(at_trim.control_reference, np.array([2.0]))
    np.testing.assert_allclose(at_trim.control, np.array([2.0]))

    away = controller.evaluate(
        configuration_value=0.0,
        configuration_gradient=np.zeros(1),
        generalized_velocity=np.array([1.0]),
        filter_state=np.zeros(1),
        dynamics_bias=np.array([2.0]),
        trim_wrench=np.array([2.0]),
    )
    assert away.trim_metric >= 0.5
    assert away.trim_activation == pytest.approx(0.0)
    np.testing.assert_allclose(away.control_reference, np.zeros(1))
