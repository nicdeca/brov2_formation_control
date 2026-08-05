import numpy as np
import pytest

from formation_control.actuation import (
    BlueROV2HeavyThrusterAllocation,
    BlueROV2HeavyThrusterConfiguration,
    T200ForceLimits,
)


def test_t200_nominal_16v_limits_match_published_values():
    limits = T200ForceLimits.from_voltage(16)

    assert limits.forward == pytest.approx(5.25 * 9.80665)
    assert limits.reverse == pytest.approx(4.10 * 9.80665)
    assert limits.lower == pytest.approx(-limits.reverse)
    assert limits.upper == pytest.approx(limits.forward)


def test_t200_derating_scales_both_directions():
    full = T200ForceLimits.from_voltage(16)
    derated = T200ForceLimits.from_voltage(16, derating=0.8)

    assert derated.forward == pytest.approx(0.8 * full.forward)
    assert derated.reverse == pytest.approx(0.8 * full.reverse)


def test_default_heavy_allocation_has_rank_six():
    allocation = BlueROV2HeavyThrusterAllocation.default_45deg()

    assert allocation.matrix.shape == (6, 8)
    assert allocation.rank == 6


def test_each_allocation_column_is_force_and_moment_from_geometry():
    configuration = BlueROV2HeavyThrusterConfiguration.default_45deg()
    allocation = BlueROV2HeavyThrusterAllocation(configuration)

    for index in range(8):
        direction = configuration.directions_body[index]
        position = configuration.positions_body[index]
        expected = np.concatenate((direction, np.cross(position, direction)))
        np.testing.assert_allclose(
            allocation.matrix[:, index],
            expected,
        )


def test_horizontal_thruster_height_generates_roll_and_pitch_moments():
    allocation = BlueROV2HeavyThrusterAllocation.default_45deg()

    # Unlike the old simplified mixer, the physical r x e term retains the
    # small roll/pitch coupling from the horizontal thrusters' z offset.
    assert abs(allocation.matrix[3, 0]) > 0.0
    assert abs(allocation.matrix[4, 0]) > 0.0


@pytest.mark.parametrize("axis", range(6))
def test_small_canonical_wrenches_are_reconstructible(axis):
    allocation = BlueROV2HeavyThrusterAllocation.default_45deg()
    wrench = np.zeros(6)
    wrench[axis] = 1.0

    result = allocation.bounded_least_squares(wrench)

    np.testing.assert_allclose(
        result.achieved_wrench,
        wrench,
        atol=1e-8,
    )
    assert allocation.contains(result.forces)


def test_utilization_respects_asymmetric_limits():
    allocation = BlueROV2HeavyThrusterAllocation.default_45deg()
    forces = np.zeros(8)
    forces[0] = allocation.upper_bounds[0]
    forces[1] = allocation.lower_bounds[1]

    utilization = allocation.utilization(forces)

    assert utilization[0] == pytest.approx(1.0)
    assert utilization[1] == pytest.approx(1.0)


def test_corner_wrench_count():
    allocation = BlueROV2HeavyThrusterAllocation.default_45deg()

    corners = allocation.corner_wrenches()

    assert corners.shape == (256, 6)
