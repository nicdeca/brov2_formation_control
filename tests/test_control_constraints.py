import numpy as np
import pytest

from formation_control.control import PolyhedralControlSet


def test_box_control_set_with_scalar_bounds():
    control_set = PolyhedralControlSet.box(
        -2.0,
        3.0,
        dimension=3,
    )

    np.testing.assert_allclose(control_set.lower, -2.0 * np.ones(3))
    np.testing.assert_allclose(control_set.upper, 3.0 * np.ones(3))
    assert control_set.contains(np.array([0.0, -2.0, 3.0]))
    assert not control_set.contains(np.array([0.0, -2.1, 3.0]))


def test_box_control_set_infers_dimension_from_vectors():
    control_set = PolyhedralControlSet.box(
        np.array([-1.0, -2.0]),
        np.array([1.0, 2.0]),
    )

    assert control_set.dimension == 2


def test_polyhedral_control_set():
    matrix = np.array(
        [
            [1.0, 1.0],
            [-1.0, 0.0],
        ]
    )
    bound = np.array([1.0, 0.5])
    control_set = PolyhedralControlSet.polyhedron(matrix, bound)

    assert control_set.contains(np.array([0.2, 0.3]))
    assert not control_set.contains(np.array([0.8, 0.5]))
    assert not control_set.contains(np.array([-0.6, 0.0]))


def test_control_set_rejects_inconsistent_bounds():
    with pytest.raises(ValueError):
        PolyhedralControlSet.box(
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
        )
