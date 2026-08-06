import numpy as np

from formation_control.geometry import (
    quaternion_from_roll_pitch_yaw,
    rotation_matrix_from_quaternion,
)
from formation_control.models import BlueROV2Model
from formation_control.simulation.stress_cases import (
    outward_unit_vector,
    set_inertial_linear_velocity,
    set_outward_linear_velocity,
)


def test_outward_unit_vector_points_from_target_to_observer():
    observer = np.array([2.0, 0.0, -1.0])
    target = np.array([0.0, 0.0, -1.0])

    direction = outward_unit_vector(observer, target)

    np.testing.assert_allclose(direction, np.array([1.0, 0.0, 0.0]))


def test_set_inertial_linear_velocity_accounts_for_attitude():
    model = BlueROV2Model()
    state = np.zeros(model.state_dim)
    state[3:7] = quaternion_from_roll_pitch_yaw(
        roll=0.0,
        pitch=0.0,
        yaw=np.pi / 2.0,
    )

    desired_inertial = np.array([1.2, -0.3, 0.15])
    updated = set_inertial_linear_velocity(
        model,
        state,
        desired_inertial,
    )

    rotation = rotation_matrix_from_quaternion(updated[3:7])
    realized_inertial = rotation @ updated[7:10]

    np.testing.assert_allclose(
        realized_inertial,
        desired_inertial,
        atol=1e-12,
    )


def test_set_outward_velocity_moves_away_from_target():
    model = BlueROV2Model()
    state = np.zeros(model.state_dim)
    state[:3] = np.array([-2.0, 0.5, -1.0])
    state[3:7] = quaternion_from_roll_pitch_yaw(0.0, 0.0, 0.4)
    target = np.array([0.0, 0.0, -1.0])

    updated = set_outward_linear_velocity(
        model,
        state,
        target,
        speed=0.8,
    )

    rotation = rotation_matrix_from_quaternion(updated[3:7])
    velocity_inertial = rotation @ updated[7:10]
    outward = outward_unit_vector(updated[:3], target)

    np.testing.assert_allclose(
        velocity_inertial,
        0.8 * outward,
        atol=1e-12,
    )
