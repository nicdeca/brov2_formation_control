import numpy as np
import pytest

from formation_control.actuation import (
    BlueROV2HeavyThrusterAllocation,
    wrench_polytope_from_allocation,
)


@pytest.fixture
def polytope():
    return wrench_polytope_from_allocation(
        BlueROV2HeavyThrusterAllocation.default_45deg(voltage=16)
    )


def test_all_corner_wrenches_are_inside(polytope):
    assert all(polytope.contains(w, tolerance=5e-6) for w in polytope.vertices)


def test_support_matches_corner_maximum(polytope):
    direction = np.array([0.4, -0.2, 0.1, 0.3, -0.5, 0.6])
    assert polytope.support(direction) == pytest.approx(np.max(polytope.vertices @ direction))


def test_directional_interval_matches_vertex_extrema(polytope):
    direction = np.array([1.0, 0.2, -0.1, 0.0, 0.0, 0.0])
    unit = direction / np.linalg.norm(direction)
    minimum, maximum = polytope.directional_interval(direction)
    projections = polytope.vertices @ unit
    assert minimum == pytest.approx(np.min(projections))
    assert maximum == pytest.approx(np.max(projections))


def test_chebyshev_center_is_inside(polytope):
    assert polytope.contains(polytope.chebyshev_ball.center)
    assert polytope.chebyshev_ball.radius > 0.0


def test_canonical_axis_intervals_shape(polytope):
    intervals = polytope.canonical_axis_intervals()
    assert intervals.shape == (6, 2)
    assert np.all(intervals[:, 0] < intervals[:, 1])


def test_projection_is_two_dimensional(polytope):
    projection = polytope.projection(0, 1)
    assert projection.vertices.shape[1] == 2
    assert projection.vertices.shape[0] >= 4
