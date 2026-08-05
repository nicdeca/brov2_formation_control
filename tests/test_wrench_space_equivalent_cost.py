import numpy as np

from formation_control.actuation import BlueROV2HeavyThrusterAllocation
from formation_control.control import (
    equivalent_wrench_weight,
    minimum_effort_allocation_matrix,
)


def test_equivalent_wrench_weight_matches_minimum_thruster_effort():
    allocation = BlueROV2HeavyThrusterAllocation.default_45deg()
    matrix = allocation.matrix
    thruster_weight = 0.7 * np.eye(8)

    wrench_weight = equivalent_wrench_weight(
        matrix,
        thruster_weight,
    )
    allocation_map = minimum_effort_allocation_matrix(
        matrix,
        thruster_weight,
    )

    wrench = np.array([0.6, -0.4, 0.2, 0.03, -0.02, 0.04])
    forces = allocation_map @ wrench

    np.testing.assert_allclose(
        matrix @ forces,
        wrench,
        atol=1e-11,
    )

    thruster_cost = 0.5 * forces @ thruster_weight @ forces
    wrench_cost = 0.5 * wrench @ wrench_weight @ wrench

    np.testing.assert_allclose(
        thruster_cost,
        wrench_cost,
        rtol=1e-11,
        atol=1e-12,
    )


def test_equivalent_wrench_weight_is_positive_definite():
    allocation = BlueROV2HeavyThrusterAllocation.default_45deg()
    weight = equivalent_wrench_weight(
        allocation.matrix,
        np.eye(8),
    )

    np.testing.assert_allclose(weight, weight.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(weight) > 0.0)
