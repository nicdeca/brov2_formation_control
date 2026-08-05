"""Generic soft CLF quadratic program.

For a control-affine CLF derivative

    W_dot = a + b.T u,

the optimization problem is

    minimize    1/2 u.T H u + 1/2 p_delta delta^2

    subject to  a + b.T u <= -alpha(W) + delta,
                delta >= 0,
                u in U.

The control set ``U`` may contain box constraints, linear inequalities, or
both.  The QP is deliberately independent of robot dynamics, camera geometry,
potentials, and the particular origin of the CLF coefficients.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
from qpsolvers import solve_qp
from scipy.sparse import csc_matrix

from formation_control.models.base import FloatArray

from .backstepping_clf import BacksteppingCLFEvaluation
from .class_k import ClassKFunction
from .clf_feasibility import (
    CLFActuationFeasibility,
    box_clf_actuation_feasibility,
)
from .control_constraints import PolyhedralControlSet


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


@dataclass(frozen=True)
class CLFQPProblem:
    """Standard-form QP matrices for decision ``z = [u, delta]``."""

    quadratic_cost: FloatArray
    linear_cost: FloatArray
    inequality_matrix: FloatArray
    inequality_bound: FloatArray
    lower_bounds: FloatArray
    upper_bounds: FloatArray

    @property
    def decision_dim(self) -> int:
        return self.linear_cost.size


@dataclass(frozen=True)
class CLFQPResult:
    """Solution and diagnostics of one CLF-QP evaluation."""

    control: FloatArray
    slack: float
    objective: float
    modeled_clf_derivative: float
    desired_clf_upper_bound: float
    clf_residual: float
    actuation_feasibility: CLFActuationFeasibility | None = None

    def __post_init__(self) -> None:
        control = np.asarray(self.control, dtype=float)
        if control.ndim != 1:
            raise ValueError("control must be one-dimensional.")
        if self.slack < -1e-9:
            raise ValueError("slack must be nonnegative.")
        object.__setattr__(self, "control", control)

    @property
    def required_slack(self) -> float | None:
        """Return the minimum slack imposed by actuator feasibility, if known."""
        if self.actuation_feasibility is None:
            return None
        return self.actuation_feasibility.required_slack

    @property
    def actuation_margin(self) -> float | None:
        """Return the zero-slack CLF feasibility margin, if known."""
        if self.actuation_feasibility is None:
            return None
        return self.actuation_feasibility.margin


class CLFQPSolverError(RuntimeError):
    """Raised when the numerical QP solver does not return a solution."""


def _default_solver_options(solver: str) -> dict[str, object]:
    """Return numerically robust defaults for supported solvers.

    OSQP's generic defaults are intentionally fairly loose.  CLF-QPs can also
    become moderately ill-conditioned when the slack penalty is much larger
    than the control penalty, so use tighter tolerances, polishing, and a
    larger iteration budget by default.
    """
    if solver == "osqp":
        return {
            "eps_abs": 1e-7,
            "eps_rel": 1e-7,
            "max_iter": 100_000,
            "polishing": True,
            "raise_error": False,
        }

    return {}


@dataclass(frozen=True)
class CLFQP:
    """Soft CLF-QP with direct control minimization."""

    control_weight: FloatArray
    slack_penalty: float
    alpha: ClassKFunction
    control_set: PolyhedralControlSet
    solver: str = "osqp"
    solver_options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        control_weight = _positive_definite_matrix(
            self.control_weight,
            self.control_set.dimension,
            name="control_weight",
        )
        if not np.isfinite(self.slack_penalty) or self.slack_penalty <= 0.0:
            raise ValueError("slack_penalty must be finite and strictly positive.")
        if not self.solver:
            raise ValueError("solver must be a nonempty string.")

        solver_options = _default_solver_options(self.solver)
        solver_options.update(dict(self.solver_options))

        object.__setattr__(self, "control_weight", control_weight)
        object.__setattr__(
            self,
            "solver_options",
            MappingProxyType(solver_options),
        )

    @classmethod
    def isotropic(
        cls,
        *,
        control_dim: int,
        control_weight: float,
        slack_penalty: float,
        alpha: ClassKFunction,
        control_set: PolyhedralControlSet | None = None,
        solver: str = "osqp",
        solver_options: Mapping[str, object] | None = None,
    ) -> CLFQP:
        """Construct a QP with cost matrix ``control_weight * I``."""
        if control_dim <= 0:
            raise ValueError("control_dim must be positive.")
        if not np.isfinite(control_weight) or control_weight <= 0.0:
            raise ValueError("control_weight must be finite and strictly positive.")

        if control_set is None:
            control_set = PolyhedralControlSet.unconstrained(control_dim)
        elif control_set.dimension != control_dim:
            raise ValueError("control_set dimension does not match control_dim.")

        return cls(
            control_weight=control_weight * np.eye(control_dim),
            slack_penalty=slack_penalty,
            alpha=alpha,
            control_set=control_set,
            solver=solver,
            solver_options={} if solver_options is None else solver_options,
        )

    @property
    def control_dim(self) -> int:
        return self.control_set.dimension

    def build_problem(
        self,
        clf: BacksteppingCLFEvaluation,
    ) -> CLFQPProblem:
        """Assemble the numerical QP matrices."""
        if clf.control_gradient.shape != (self.control_dim,):
            raise ValueError("CLF control-gradient dimension does not match the QP.")

        decay = float(self.alpha(clf.value))
        if not np.isfinite(decay) or decay < 0.0:
            raise ValueError("alpha must return a finite nonnegative value for W >= 0.")

        decision_dim = self.control_dim + 1
        slack_index = self.control_dim

        quadratic_cost = np.zeros((decision_dim, decision_dim))
        quadratic_cost[: self.control_dim, : self.control_dim] = self.control_weight
        quadratic_cost[slack_index, slack_index] = self.slack_penalty
        linear_cost = np.zeros(decision_dim)

        # CLF condition:
        # b.T u - delta <= -alpha(W) - a.
        clf_row = np.concatenate((clf.control_gradient, np.array([-1.0])))
        inequality_rows = [clf_row]
        inequality_bounds = [-decay - clf.drift]

        if self.control_set.inequality_matrix is not None:
            assert self.control_set.inequality_bound is not None
            control_rows = np.column_stack(
                (
                    self.control_set.inequality_matrix,
                    np.zeros(self.control_set.inequality_matrix.shape[0]),
                )
            )
            inequality_rows.extend(control_rows)
            inequality_bounds.extend(self.control_set.inequality_bound)

        inequality_matrix = np.vstack(inequality_rows)
        inequality_bound = np.asarray(inequality_bounds, dtype=float)

        lower_bounds = np.full(decision_dim, -np.inf)
        upper_bounds = np.full(decision_dim, np.inf)

        if self.control_set.lower is not None:
            lower_bounds[: self.control_dim] = self.control_set.lower
        if self.control_set.upper is not None:
            upper_bounds[: self.control_dim] = self.control_set.upper

        lower_bounds[slack_index] = 0.0

        return CLFQPProblem(
            quadratic_cost=quadratic_cost,
            linear_cost=linear_cost,
            inequality_matrix=inequality_matrix,
            inequality_bound=inequality_bound,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )

    def actuation_feasibility(
        self,
        clf: BacksteppingCLFEvaluation,
    ) -> CLFActuationFeasibility | None:
        """Return exact zero-slack feasibility for a finite box control set.

        General polyhedral sets are left as ``None`` deliberately: the online
        thruster-space controller has a box input set and admits a closed-form
        calculation, whereas solving an additional LP for a high-facet wrench
        polytope would defeat the purpose of this inexpensive diagnostic.
        """
        control_set = self.control_set
        if control_set.inequality_matrix is not None:
            return None
        if control_set.lower is None or control_set.upper is None:
            return None
        if not (np.all(np.isfinite(control_set.lower)) and np.all(np.isfinite(control_set.upper))):
            return None

        return box_clf_actuation_feasibility(
            clf,
            alpha=self.alpha,
            control_set=control_set,
        )

    def solve(
        self,
        clf: BacksteppingCLFEvaluation,
    ) -> CLFQPResult:
        """Solve the CLF-QP and return control, slack, and diagnostics."""
        problem = self.build_problem(clf)
        actuation_feasibility = self.actuation_feasibility(clf)

        quadratic_cost = problem.quadratic_cost
        inequality_matrix = problem.inequality_matrix

        if self.solver == "osqp":
            quadratic_cost = csc_matrix(quadratic_cost)
            inequality_matrix = csc_matrix(inequality_matrix)

        solution = solve_qp(
            P=quadratic_cost,
            q=problem.linear_cost,
            G=inequality_matrix,
            h=problem.inequality_bound,
            lb=problem.lower_bounds,
            ub=problem.upper_bounds,
            solver=self.solver,
            **dict(self.solver_options),
        )

        if solution is None:
            raise CLFQPSolverError(f"QP solver {self.solver!r} did not return a solution.")

        solution = np.asarray(solution, dtype=float)
        if solution.shape != (self.control_dim + 1,):
            raise CLFQPSolverError("QP solver returned a solution with an unexpected dimension.")
        if not np.all(np.isfinite(solution)):
            raise CLFQPSolverError("QP solver returned a non-finite solution.")

        control = solution[: self.control_dim]
        slack = max(float(solution[-1]), 0.0)

        modeled_derivative = clf.derivative(control)
        desired_upper_bound = -float(self.alpha(clf.value)) + slack
        residual = modeled_derivative - desired_upper_bound

        objective = float(
            0.5 * control @ self.control_weight @ control + 0.5 * self.slack_penalty * slack**2
        )

        return CLFQPResult(
            control=control,
            slack=slack,
            objective=objective,
            modeled_clf_derivative=modeled_derivative,
            desired_clf_upper_bound=desired_upper_bound,
            clf_residual=residual,
            actuation_feasibility=actuation_feasibility,
        )
