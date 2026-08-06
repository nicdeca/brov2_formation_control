import numpy as np
import pytest

from formation_control.actuation import BlueROV2HeavyThrusterAllocation
from formation_control.control import build_bluerov2_controller_design
from formation_control.models import BlueROV2Model
from formation_control.potentials import EdgePotential, RelativePositionPotential


def test_physical_clf_configuration_value_contains_only_current_potential():
    model = BlueROV2Model()
    allocation = BlueROV2HeavyThrusterAllocation.default_45deg()
    design = build_bluerov2_controller_design(
        model,
        allocation,
        control_space="thruster",
    )
    controller = design.agent_controller

    follower_state = np.zeros(model.state_dim)
    follower_state[3] = 1.0
    parent_position = np.array([2.0, 0.0, 0.0])

    edge_potential = EdgePotential(
        formation=RelativePositionPotential.isotropic(
            np.array([1.5, 0.0, 0.0]),
            gain=1.0,
        )
    )
    filter_state = controller.initialize_filter(
        follower_state=follower_state,
        parent_position=parent_position,
        edge_potential=edge_potential,
    )

    potential, _ = controller.evaluate_edge_potential(
        follower_state=follower_state,
        parent_position=parent_position,
        edge_potential=edge_potential,
    )

    evaluation = controller.evaluate(
        follower_state=follower_state,
        parent_position=parent_position,
        edge_potential=edge_potential,
        filter_state=filter_state,
    )

    assert evaluation.controller.clf.configuration_value == pytest.approx(potential.value)
    assert evaluation.optimized_input.shape == (8,)
