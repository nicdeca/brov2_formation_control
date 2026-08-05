import numpy as np
import pytest

from formation_control.simulation import FormationTrajectory


def test_quaternion_history_is_normalized_on_storage():
    trajectory = FormationTrajectory(
        times=np.array([0.0, 0.1]),
        positions=np.zeros((2, 2, 3)),
        quaternions=np.array(
            [
                [[2.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0]],
            ]
        ),
    )

    norms = np.linalg.norm(trajectory.quaternions, axis=2)
    np.testing.assert_allclose(norms, np.ones((2, 2)))


def test_quaternion_history_shape_is_checked():
    with pytest.raises(ValueError):
        FormationTrajectory(
            times=np.array([0.0, 0.1]),
            positions=np.zeros((2, 2, 3)),
            quaternions=np.zeros((2, 2, 3)),
        )
