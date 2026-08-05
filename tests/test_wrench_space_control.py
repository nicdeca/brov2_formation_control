import numpy as np
import pytest

pytest.importorskip("qpsolvers")
pytest.importorskip("osqp")

from formation_control.actuation import (  # noqa: E402
    BlueROV2HeavyThrusterAllocation,
    wrench_polytope_from_allocation,
)
from formation_control.control import (  # noqa: E402
    FirstOrderCommandFilter,
    LinearClassK,
    build_wrench_space_controller,
    wrench_polytope_control_set,
)
from formation_control.models import BlueROV2Model  # noqa: E402


def make_polytope():
    allocation = BlueROV2HeavyThrusterAllocation.default_45deg()
    return allocation, wrench_polytope_from_allocation(allocation)


def test_wrench_control_set_accepts_all_polytope_corner_wrenches():
    _, polytope = make_polytope()
    control_set = wrench_polytope_control_set(polytope)

    for wrench in polytope.vertices:
        assert control_set.contains(wrench, tolerance=5e-6)


def test_wrench_control_set_rejects_outside_point():
    _, polytope = make_polytope()
    control_set = wrench_polytope_control_set(polytope)

    direction = np.array([1.0, 0.2, -0.1, 0.05, 0.0, 0.1])
    direction /= np.linalg.norm(direction)
    outside = 1.05 * polytope.support(direction) * direction

    # The radial point constructed this way can occasionally still be inside
    # an anisotropic polytope, so push it farther if needed.
    if control_set.contains(outside):
        outside = 2.0 * polytope.support(direction) * direction

    assert not control_set.contains(outside)


def test_wrench_space_controller_has_six_dimensional_input():
    _, polytope = make_polytope()
    model = BlueROV2Model()

    design = build_wrench_space_controller(
        inertia=model.mass_matrix,
        virtual_gain=np.eye(6),
        command_filter=FirstOrderCommandFilter(
            signal_dim=6,
            bandwidth=4.0,
        ),
        polytope=polytope,
        control_weight=np.eye(6),
        slack_penalty=1000.0,
        alpha=LinearClassK(gain=1.0),
    )

    assert design.controller.control_dim == 6
    np.testing.assert_allclose(
        design.controller.clf.input_matrix,
        np.eye(6),
    )


def test_thruster_corner_images_and_h_representation_agree():
    _, polytope = make_polytope()
    halfspaces = polytope.halfspaces

    residual = halfspaces.matrix @ polytope.vertices.T - halfspaces.vector[:, None]

    assert np.max(residual) <= 5e-6
