import numpy as np
import pytest

pytest.importorskip("qpsolvers")
pytest.importorskip("osqp")

from formation_control.actuation import (  # noqa: E402
    BlueROV2HeavyThrusterAllocation,
)
from formation_control.control import (  # noqa: E402
    build_bluerov2_controller_design,
)
from formation_control.models import BlueROV2Model  # noqa: E402


@pytest.mark.parametrize(
    ("control_space", "decision_dimension"),
    [
        ("thruster", 8),
        ("wrench", 6),
    ],
)
def test_bluerov2_control_space_decision_dimension(
    control_space,
    decision_dimension,
):
    model = BlueROV2Model()
    allocation = BlueROV2HeavyThrusterAllocation.default_45deg()

    design = build_bluerov2_controller_design(
        model,
        allocation,
        control_space=control_space,
    )

    assert design.decision_dimension == decision_dimension


def test_thruster_and_wrench_designs_share_the_same_physical_polytope():
    model = BlueROV2Model()
    allocation = BlueROV2HeavyThrusterAllocation.default_45deg()

    thruster = build_bluerov2_controller_design(
        model,
        allocation,
        control_space="thruster",
    )
    wrench = build_bluerov2_controller_design(
        model,
        allocation,
        control_space="wrench",
    )

    np.testing.assert_allclose(
        thruster.wrench_polytope.vertices,
        wrench.wrench_polytope.vertices,
    )
    np.testing.assert_allclose(
        thruster.thruster_weight,
        wrench.thruster_weight,
    )
    np.testing.assert_allclose(
        thruster.wrench_weight,
        wrench.wrench_weight,
    )


def test_invalid_control_space_is_rejected():
    model = BlueROV2Model()
    allocation = BlueROV2HeavyThrusterAllocation.default_45deg()

    with pytest.raises(ValueError):
        build_bluerov2_controller_design(
            model,
            allocation,
            control_space="invalid",
        )
