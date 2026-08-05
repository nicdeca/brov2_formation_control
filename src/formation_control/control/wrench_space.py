"""CLF-QP helpers for exact wrench-polytope constraints.

The thruster-space and wrench-space formulations share the same physical
feasibility set

    W = {B f : f_min <= f <= f_max}.

The thruster-space formulation optimizes directly over ``f``.  The wrench-space
formulation instead optimizes over ``tau`` subject to the exact H-representation

    A_W tau <= b_W.

The latter is useful when a downstream autopilot accepts a body wrench and
performs its own allocation.

For a quadratic thruster effort ``1/2 f.T H_f f``, the minimum unconstrained
effort required to realize a wrench ``tau`` is

    1/2 tau.T H_tau tau,

where

    H_tau = (B H_f^{-1} B.T)^{-1}.

Using ``H_tau`` in the wrench-space QP therefore makes the two objectives
locally equivalent whenever the minimum-effort allocation is not limited by
individual thruster bounds.  Close to saturation, exact equivalence is lost
because the bounded allocation cost becomes piecewise quadratic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from formation_control.actuation import AchievableWrenchPolytope
from formation_control.models.base import FloatArray

from .backstepping_clf import BacksteppingCLF
from .class_k import ClassKFunction
from .clf_qp import CLFQP
from .control_constraints import PolyhedralControlSet
from .filters import CommandFilter
from .second_order import SecondOrderCLFQPController


def _positive_definite_matrix(
    value: FloatArray,
    dimension: int,
    *,
    name: str,
) -> FloatArray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (dimension, dimension):
        raise ValueError(f"{name} must have shape ({dimension}, {dimension}), got {matrix.shape}.")
    if not np.allclose(matrix, matrix.T):
        raise ValueError(f"{name} must be symmetric.")
    if np.any(np.linalg.eigvalsh(matrix) <= 0.0):
        raise ValueError(f"{name} must be positive definite.")
    return matrix


def wrench_polytope_control_set(
    polytope: AchievableWrenchPolytope,
) -> PolyhedralControlSet:
    """Return the exact 6-D polyhedral wrench constraint."""
    halfspaces = polytope.halfspaces
    return PolyhedralControlSet(
        dimension=6,
        inequality_matrix=halfspaces.matrix,
        inequality_bound=halfspaces.vector,
    )


def equivalent_wrench_weight(
    allocation_matrix: FloatArray,
    thruster_weight: FloatArray,
) -> FloatArray:
    """Return the interior-equivalent quadratic wrench-effort matrix.

    If

        min_f  1/2 f.T H_f f
        s.t.   B f = tau

    is unconstrained by individual thruster bounds, its optimum value is

        1/2 tau.T H_tau tau,

    with ``H_tau = (B H_f^{-1} B.T)^{-1}``.
    """
    matrix = np.asarray(allocation_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != 6:
        raise ValueError("allocation_matrix must have shape (6, n_thrusters).")
    if np.linalg.matrix_rank(matrix) < 6:
        raise ValueError("allocation_matrix must have full wrench rank.")

    weight = _positive_definite_matrix(
        thruster_weight,
        matrix.shape[1],
        name="thruster_weight",
    )

    # Avoid forming H_f^{-1} explicitly.
    weighted_transpose = np.linalg.solve(weight, matrix.T)
    wrench_gramian = matrix @ weighted_transpose
    wrench_weight = np.linalg.inv(wrench_gramian)

    # Remove tiny numerical asymmetry before downstream SPD validation.
    return 0.5 * (wrench_weight + wrench_weight.T)


def minimum_effort_allocation_matrix(
    allocation_matrix: FloatArray,
    thruster_weight: FloatArray,
) -> FloatArray:
    """Return the unconstrained minimum-effort map ``f = K tau``."""
    matrix = np.asarray(allocation_matrix, dtype=float)
    wrench_weight = equivalent_wrench_weight(
        matrix,
        thruster_weight,
    )
    weight = _positive_definite_matrix(
        thruster_weight,
        matrix.shape[1],
        name="thruster_weight",
    )

    return np.linalg.solve(weight, matrix.T) @ wrench_weight


@dataclass(frozen=True)
class WrenchSpaceControllerDesign:
    """Convenience data for a wrench-space second-order controller."""

    controller: SecondOrderCLFQPController
    polytope: AchievableWrenchPolytope


def build_wrench_space_controller(
    *,
    inertia: FloatArray,
    virtual_gain: FloatArray,
    command_filter: CommandFilter,
    polytope: AchievableWrenchPolytope,
    control_weight: FloatArray,
    slack_penalty: float,
    alpha: ClassKFunction,
    solver: str = "osqp",
) -> WrenchSpaceControllerDesign:
    """Build a controller whose QP decision variable is body wrench."""
    controller = SecondOrderCLFQPController(
        virtual_gain=np.asarray(virtual_gain, dtype=float),
        command_filter=command_filter,
        clf=BacksteppingCLF(
            inertia=np.asarray(inertia, dtype=float),
            input_matrix=np.eye(6),
        ),
        qp=CLFQP(
            control_weight=np.asarray(control_weight, dtype=float),
            slack_penalty=slack_penalty,
            alpha=alpha,
            control_set=wrench_polytope_control_set(polytope),
            solver=solver,
        ),
    )
    return WrenchSpaceControllerDesign(
        controller=controller,
        polytope=polytope,
    )
