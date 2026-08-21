"""Tests for the BlueROV2 leader CLF-QP adapter."""

import numpy as np
import pytest

from formation_control.actuation import BlueROV2HeavyThrusterAllocation
from formation_control.control.bluerov2_design import (
    build_bluerov2_controller_design,
)
from formation_control.control.bluerov2_leader import (
    BlueROV2LeaderController,
    LeaderTrajectorySample,
)
from formation_control.geometry import quaternion_from_roll_pitch_yaw
from formation_control.models import BlueROV2Model


def make_state() -> np.ndarray:
    state = np.zeros(13)
    state[:3] = np.array([1.0, -0.5, -1.2])
    state[3:7] = quaternion_from_roll_pitch_yaw(
        roll=0.0,
        pitch=0.0,
        yaw=0.0,
    )
    return state


def test_leader_zero_tracking_error_has_zero_configuration_gradient():
    model = BlueROV2Model()
    allocation = BlueROV2HeavyThrusterAllocation.default_45deg()
    design = build_bluerov2_controller_design(
        model,
        allocation,
        control_space="thruster",
    )
    state = make_state()
    leader = BlueROV2LeaderController.from_initial_state(
        design.agent_controller.dynamics_controller,
        model,
        state,
    )
    reference = LeaderTrajectorySample(
        position=state[:3],
        velocity=np.array([0.3, 0.0, 0.0]),
        acceleration=np.array([0.1, 0.0, 0.0]),
    )

    value, gradient, feedforward, feedforward_dot = leader.tracking_terms(state, reference)

    assert value == 0.0
    np.testing.assert_allclose(gradient, np.zeros(6))
    np.testing.assert_allclose(
        feedforward,
        np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(
        feedforward_dot,
        np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )


def test_leader_uses_same_dynamic_controller_instance_as_followers():
    model = BlueROV2Model()
    allocation = BlueROV2HeavyThrusterAllocation.default_45deg()
    design = build_bluerov2_controller_design(
        model,
        allocation,
        control_space="thruster",
    )
    state = make_state()
    leader = BlueROV2LeaderController.from_initial_state(
        design.agent_controller.dynamics_controller,
        model,
        state,
    )

    assert leader.dynamics_controller is design.agent_controller.dynamics_controller


def test_leader_stationary_equilibrium_uses_restoring_wrench_trim():
    model = BlueROV2Model()
    allocation = BlueROV2HeavyThrusterAllocation.default_45deg()
    design = build_bluerov2_controller_design(
        model,
        allocation,
        control_space="thruster",
    )
    state = make_state()
    leader = BlueROV2LeaderController.from_initial_state(
        design.agent_controller.dynamics_controller,
        model,
        state,
    )
    reference = LeaderTrajectorySample(
        position=state[:3],
        velocity=np.zeros(3),
        acceleration=np.zeros(3),
    )
    filter_state = leader.initialize_filter(
        state=state,
        reference=reference,
    )

    evaluation = leader.evaluate(
        state=state,
        reference=reference,
        filter_state=filter_state,
    )

    assert evaluation.controller.trim_activation == pytest.approx(1.0)
    np.testing.assert_allclose(
        evaluation.wrench_body,
        model.restoring_wrench(state[3:7]),
        atol=1e-8,
    )
