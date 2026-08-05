import numpy as np
import pytest

from formation_control.simulation import FormationTrajectory


def test_formation_trajectory_properties():
    times = np.array([0.0, 0.1, 0.2])
    positions = np.zeros((3, 4, 3))
    velocities = np.ones((3, 4, 3))
    controls = np.zeros((2, 4, 3))

    trajectory = FormationTrajectory(
        times=times,
        positions=positions,
        velocities=velocities,
        controls=controls,
    )

    assert trajectory.n_samples == 3
    assert trajectory.n_agents == 4
    assert trajectory.dimension == 3
    assert trajectory.duration == pytest.approx(0.2)
    assert trajectory.path(2).shape == (3, 3)


def test_formation_trajectory_rejects_misaligned_controls():
    with pytest.raises(ValueError):
        FormationTrajectory(
            times=np.array([0.0, 0.1, 0.2]),
            positions=np.zeros((3, 2, 3)),
            controls=np.zeros((3, 2, 3)),
        )
