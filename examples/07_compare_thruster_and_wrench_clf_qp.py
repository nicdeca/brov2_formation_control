"""Compare thruster-space and exact wrench-space CLF-QP formulations."""

from __future__ import annotations

import argparse

import numpy as np

from formation_control.actuation import (
    BlueROV2HeavyThrusterAllocation,
    wrench_polytope_from_allocation,
)
from formation_control.control import (
    CLFQP,
    BacksteppingCLF,
    FirstOrderCommandFilter,
    LinearClassK,
    PolyhedralControlSet,
    SecondOrderCLFQPController,
    build_wrench_space_controller,
)
from formation_control.models import BlueROV2Model


def build_thruster_space_controller(
    model: BlueROV2Model,
    allocation: BlueROV2HeavyThrusterAllocation,
) -> SecondOrderCLFQPController:
    force_scale = max(
        allocation.configuration.force_limits.forward,
        allocation.configuration.force_limits.reverse,
    )

    return SecondOrderCLFQPController(
        virtual_gain=np.diag([0.55, 0.55, 0.55, 0.8, 0.8, 0.8]),
        command_filter=FirstOrderCommandFilter(
            signal_dim=6,
            bandwidth=np.array([3.0, 3.0, 3.0, 4.0, 4.0, 4.0]),
        ),
        clf=BacksteppingCLF(
            model.mass_matrix,
            input_matrix=allocation.matrix,
        ),
        qp=CLFQP(
            control_weight=(1.0 / force_scale**2) * np.eye(8),
            slack_penalty=5e3,
            alpha=LinearClassK(gain=0.8),
            control_set=PolyhedralControlSet.box(
                allocation.lower_bounds,
                allocation.upper_bounds,
            ),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thruster-voltage", type=int, choices=(12, 16, 20), default=16)
    parser.add_argument("--thrust-derating", type=float, default=1.0)
    args = parser.parse_args()

    model = BlueROV2Model()
    allocation = BlueROV2HeavyThrusterAllocation.default_45deg(
        voltage=args.thruster_voltage,
        derating=args.thrust_derating,
    )
    polytope = wrench_polytope_from_allocation(allocation)

    thruster_controller = build_thruster_space_controller(model, allocation)
    wrench_design = build_wrench_space_controller(
        inertia=model.mass_matrix,
        virtual_gain=np.diag([0.55, 0.55, 0.55, 0.8, 0.8, 0.8]),
        command_filter=FirstOrderCommandFilter(
            signal_dim=6,
            bandwidth=np.array([3.0, 3.0, 3.0, 4.0, 4.0, 4.0]),
        ),
        polytope=polytope,
        control_weight=np.eye(6),
        slack_penalty=5e3,
        alpha=LinearClassK(gain=0.8),
    )

    print("Thruster-space QP:")
    print(f"  decision dimension: {thruster_controller.control_dim}")
    print("  constraints: physical asymmetric T200 force bounds")
    print()
    print("Wrench-space QP:")
    print(f"  decision dimension: {wrench_design.controller.control_dim}")
    print(f"  constraints: {polytope.halfspaces.n_facets} exact wrench-polytope inequalities")
    print()
    print(
        "Both formulations expose exactly the same physical wrench-feasible set; "
        "their optimal commands need not coincide because the quadratic objectives "
        "are expressed in different coordinates."
    )


if __name__ == "__main__":
    main()
