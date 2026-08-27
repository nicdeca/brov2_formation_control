import numpy as np

from formation_control.actuation import BlueROV2HeavyThrusterAllocation
from formation_control.control import build_bluerov2_controller_design
from formation_control.models import BlueROV2Model


def test_physical_bluerov2_clf_qp_has_only_eight_thruster_inputs():
    model = BlueROV2Model()
    allocation = BlueROV2HeavyThrusterAllocation.default_45deg()

    design = build_bluerov2_controller_design(
        model,
        allocation,
        control_space="thruster",
    )

    assert design.decision_dimension == 8
    input_matrix = design.agent_controller.dynamics_controller.clf.input_matrix
    assert input_matrix.shape == (6, 8)
    np.testing.assert_allclose(input_matrix, allocation.matrix)
    assert design.agent_controller.dynamics_controller.qp.uses_direct_box_solver


def test_physical_controller_retains_linear_slack_penalty():
    model = BlueROV2Model()
    allocation = BlueROV2HeavyThrusterAllocation.default_45deg()

    design = build_bluerov2_controller_design(
        model,
        allocation,
        control_space="thruster",
        slack_linear_penalty=123.0,
    )

    assert design.agent_controller.dynamics_controller.qp.slack_linear_penalty == 123.0


def test_bluerov2_design_uses_smooth_virtual_twist_norm_limits_by_default():
    model = BlueROV2Model()
    allocation = BlueROV2HeavyThrusterAllocation.default_45deg()

    design = build_bluerov2_controller_design(
        model,
        allocation,
        control_space="thruster",
    )

    np.testing.assert_allclose(
        design.agent_controller.dynamics_controller.virtual_velocity_norm_limits,
        np.array([1.5, 2.0]),
    )


def test_bluerov2_design_uses_velocity_error_gain_for_new_qp_inequality():
    model = BlueROV2Model()
    allocation = BlueROV2HeavyThrusterAllocation.default_45deg()

    design = build_bluerov2_controller_design(
        model,
        allocation,
        control_space="thruster",
        alpha_gain=0.8,
    )

    np.testing.assert_allclose(
        design.agent_controller.dynamics_controller.velocity_error_gain,
        0.8 * model.mass_matrix,
    )


def test_bluerov2_design_configures_smooth_trim_activation():
    model = BlueROV2Model()
    allocation = BlueROV2HeavyThrusterAllocation.default_45deg()

    design = build_bluerov2_controller_design(
        model,
        allocation,
        control_space="thruster",
        trim_activation_on=0.4,
        trim_activation_off=1.6,
    )
    controller = design.agent_controller.dynamics_controller

    assert controller.trim_control_allocator is not None
    assert controller.trim_activation_on == 0.4
    assert controller.trim_activation_off == 1.6


def test_thruster_trim_allocator_reproduces_restoring_wrench():
    model = BlueROV2Model()
    allocation = BlueROV2HeavyThrusterAllocation.default_45deg()
    design = build_bluerov2_controller_design(
        model,
        allocation,
        control_space="thruster",
    )
    controller = design.agent_controller.dynamics_controller
    assert controller.trim_control_allocator is not None

    quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    restoring = model.restoring_wrench(quaternion)
    trim = controller.trim_control_allocator(restoring)

    assert trim is not None
    assert allocation.contains(trim, tolerance=1e-8)
    np.testing.assert_allclose(
        allocation.matrix @ trim,
        restoring,
        atol=1e-5,
    )
