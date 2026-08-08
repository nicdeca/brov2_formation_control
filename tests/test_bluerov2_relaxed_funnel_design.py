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
