import numpy as np

from formation_control.geometry import (
    quaternion_from_roll_pitch_yaw,
    quaternion_to_roll_pitch_yaw,
    rotation_matrix_from_quaternion,
)


def test_roll_pitch_yaw_quaternion_round_trip():
    expected = np.array([0.2, -0.3, 0.7])
    quaternion = quaternion_from_roll_pitch_yaw(*expected)
    actual = np.array(quaternion_to_roll_pitch_yaw(quaternion))

    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_body_x_axis_matches_expected_yaw_pitch_direction():
    yaw = 0.6
    pitch = -0.25
    quaternion = quaternion_from_roll_pitch_yaw(
        0.0,
        pitch,
        yaw,
    )
    rotation = rotation_matrix_from_quaternion(quaternion)

    expected = np.array(
        [
            np.cos(yaw) * np.cos(pitch),
            np.sin(yaw) * np.cos(pitch),
            -np.sin(pitch),
        ]
    )

    np.testing.assert_allclose(rotation[:, 0], expected, atol=1e-12)
