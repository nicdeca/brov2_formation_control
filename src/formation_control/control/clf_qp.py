"""Generic soft CLF quadratic program.

For an affine scalar dissipation condition

    a_c + b.T u <= -d + delta,

the optimization problem is

    minimize    1/2 (u-u_ref).T H (u-u_ref)
                + p_1 delta + 1/2 p_2 delta^2

    subject to  a_c + b.T u <= -d + delta,
                delta >= 0,
                u in U.

The default generic behavior remains the conventional CLF condition
``a + b.T u <= -alpha(W) + delta``.  The command-filtered second-order
controller supplies the paper-specific pair ``(a_c, d)`` explicitly, with

    a_c = e_nu.T (zeta - h - M nu_c_dot),
    d   = e_nu.T K_e e_nu.

This keeps the QP layer model-independent while implementing the updated
backstepping inequality exactly.

``slack_linear_penalty`` is ``p_1`` and ``slack_penalty`` is the quadratic
coefficient ``p_2``.

For the online BlueROV2 controller, ``U`` is a thruster-force box and ``H`` is
diagonal.  That special case admits a deterministic scalar-KKT solver.
General polyhedral cases continue to use the configured external QP solver.
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


def _is_diagonal(matrix: FloatArray) -> bool:
    diagonal = np.diag(np.diag(matrix))
    return bool(
        np.allclose(
            matrix,
            diagonal,
            rtol=1e-12,
            atol=1e-14,
        )
    )


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
    """Raised when a numerical QP solver does not return a valid solution."""


def _default_solver_options(solver: str) -> dict[str, object]:
    """Return numerically robust defaults for supported external solvers."""
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
    slack_linear_penalty: float = 0.0
    solver: str = "osqp"
    solver_options: Mapping[str, object] = field(default_factory=dict)
    feasibility_control_dim: int | None = None

    def __post_init__(self) -> None:
        control_weight = _positive_definite_matrix(
            self.control_weight,
            self.control_set.dimension,
            name="control_weight",
        )
        if not np.isfinite(self.slack_penalty) or self.slack_penalty <= 0.0:
            raise ValueError("slack_penalty must be finite and strictly positive.")
        if not np.isfinite(self.slack_linear_penalty) or self.slack_linear_penalty < 0.0:
            raise ValueError("slack_linear_penalty must be finite and nonnegative.")
        if not self.solver:
            raise ValueError("solver must be a nonempty string.")
        if self.feasibility_control_dim is not None and not (
            1 <= self.feasibility_control_dim <= self.control_set.dimension
        ):
            raise ValueError("feasibility_control_dim must lie in [1, control_dim].")

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
        slack_linear_penalty: float = 0.0,
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
            slack_linear_penalty=slack_linear_penalty,
            solver=solver,
            solver_options={} if solver_options is None else solver_options,
        )

    @property
    def control_dim(self) -> int:
        return self.control_set.dimension

    @property
    def uses_direct_box_solver(self) -> bool:
        """Whether this QP has the structure required by the direct solver."""
        return self.control_set.inequality_matrix is None and _is_diagonal(self.control_weight)

    def _constraint_terms(
        self,
        clf: BacksteppingCLFEvaluation,
        *,
        constraint_drift: float | None,
        dissipation_rate: float | None,
    ) -> tuple[float, float]:
        """Return ``(a_c, d)`` for ``a_c + b.T u <= -d + delta``.

        If no explicit terms are supplied, retain the generic legacy CLF
        condition with ``a_c = clf.drift`` and ``d = alpha(W)``.
        """
        if (constraint_drift is None) != (dissipation_rate is None):
            raise ValueError(
                "constraint_drift and dissipation_rate must be supplied together."
            )

        if constraint_drift is None:
            drift = float(clf.drift)
        else:
            drift = float(constraint_drift)
            if not np.isfinite(drift):
                raise ValueError("constraint_drift must be finite.")

        if dissipation_rate is None:
            decay = float(self.alpha(clf.value))
            if not np.isfinite(decay) or decay < 0.0:
                raise ValueError(
                    "alpha must return a finite nonnegative value for W >= 0."
                )
        else:
            decay = float(dissipation_rate)
            if not np.isfinite(decay) or decay < 0.0:
                raise ValueError("dissipation_rate must be finite and nonnegative.")

        return drift, decay

    def build_problem(
        self,
        clf: BacksteppingCLFEvaluation,
        *,
        control_reference: FloatArray | None = None,
        constraint_drift: float | None = None,
        dissipation_rate: float | None = None,
        lower_override: FloatArray | None = None,
        upper_override: FloatArray | None = None,
    ) -> CLFQPProblem:
        """Assemble the numerical QP matrices.

        ``constraint_drift`` and ``dissipation_rate`` define

            constraint_drift + b.T u <= -dissipation_rate + delta.

        Omitting both recovers the generic ``-alpha(W)`` CLF formulation.
        """
        if clf.control_gradient.shape != (self.control_dim,):
            raise ValueError("CLF control-gradient dimension does not match the QP.")

        drift, decay = self._constraint_terms(
            clf,
            constraint_drift=constraint_drift,
            dissipation_rate=dissipation_rate,
        )

        decision_dim = self.control_dim + 1
        slack_index = self.control_dim

        quadratic_cost = np.zeros((decision_dim, decision_dim))
        quadratic_cost[: self.control_dim, : self.control_dim] = self.control_weight
        quadratic_cost[slack_index, slack_index] = self.slack_penalty

        reference = self._control_reference(control_reference)
        linear_cost = np.zeros(decision_dim)
        linear_cost[: self.control_dim] = -(self.control_weight @ reference)
        linear_cost[slack_index] = self.slack_linear_penalty

        # b.T u - delta <= -d - a_c.
        clf_row = np.concatenate((clf.control_gradient, np.array([-1.0])))
        inequality_rows = [clf_row]
        inequality_bounds = [-decay - drift]

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

        lower_control, upper_control = self._control_bounds(
            lower_override=lower_override,
            upper_override=upper_override,
        )
        lower_bounds = np.full(decision_dim, -np.inf)
        upper_bounds = np.full(decision_dim, np.inf)
        lower_bounds[: self.control_dim] = lower_control
        upper_bounds[: self.control_dim] = upper_control
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
        *,
        constraint_drift: float | None = None,
        dissipation_rate: float | None = None,
        lower_override: FloatArray | None = None,
        upper_override: FloatArray | None = None,
    ) -> CLFActuationFeasibility | None:
        """Return exact zero-slack feasibility for a finite box control set."""
        if self.control_set.inequality_matrix is not None:
            return None

        lower, upper = self._control_bounds(
            lower_override=lower_override,
            upper_override=upper_override,
        )
        if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
            return None

        feasibility_dim = (
            self.control_dim
            if self.feasibility_control_dim is None
            else self.feasibility_control_dim
        )

        # Preserve the existing helper for callers using the conventional
        # ``-alpha(W)`` formulation.
        if constraint_drift is None and dissipation_rate is None:
            feasibility_clf = clf
            if feasibility_dim != self.control_dim:
                feasibility_clf = BacksteppingCLFEvaluation(
                    value=clf.value,
                    configuration_value=clf.configuration_value,
                    velocity_error_value=clf.velocity_error_value,
                    velocity_error=clf.velocity_error,
                    drift=clf.drift,
                    control_gradient=clf.control_gradient[:feasibility_dim],
                )

            return box_clf_actuation_feasibility(
                feasibility_clf,
                alpha=self.alpha,
                control_set=PolyhedralControlSet(
                    dimension=feasibility_dim,
                    lower=lower[:feasibility_dim],
                    upper=upper[:feasibility_dim],
                ),
            )

        drift, decay = self._constraint_terms(
            clf,
            constraint_drift=constraint_drift,
            dissipation_rate=dissipation_rate,
        )
        gradient = np.asarray(clf.control_gradient[:feasibility_dim], dtype=float)
        lower_f = lower[:feasibility_dim]
        upper_f = upper[:feasibility_dim]

        midpoint = 0.5 * (lower_f + upper_f)
        minimizing_control = midpoint.copy()
        minimizing_control[gradient > 0.0] = lower_f[gradient > 0.0]
        minimizing_control[gradient < 0.0] = upper_f[gradient < 0.0]

        minimum_constraint_lhs = drift + float(gradient @ minimizing_control)
        constraint_margin = -decay - minimum_constraint_lhs
        required_slack = max(-constraint_margin, 0.0)

        # Keep the diagnostic fields in full-CLF coordinates.  The offset
        # ``clf.drift - drift`` cancels exactly from the feasibility margin.
        full_gradient = np.asarray(clf.control_gradient, dtype=float)
        full_control = np.zeros(self.control_dim)
        full_control[:feasibility_dim] = minimizing_control
        minimum_modeled_derivative = float(
            clf.drift + full_gradient @ full_control
        )
        zero_slack_upper_bound = float(clf.drift - drift - decay)

        return CLFActuationFeasibility(
            minimizing_control=minimizing_control,
            minimum_modeled_derivative=minimum_modeled_derivative,
            zero_slack_upper_bound=zero_slack_upper_bound,
            margin=float(constraint_margin),
            required_slack=float(required_slack),
        )

    def _control_reference(
        self,
        control_reference: FloatArray | None,
    ) -> FloatArray:
        if control_reference is None:
            return np.zeros(self.control_dim)

        reference = np.asarray(control_reference, dtype=float)
        if reference.shape != (self.control_dim,):
            raise ValueError(
                f"control_reference must have shape ({self.control_dim},), got {reference.shape}."
            )
        if not np.all(np.isfinite(reference)):
            raise ValueError("control_reference must contain only finite values.")
        return reference

    def _control_bounds(
        self,
        *,
        lower_override: FloatArray | None = None,
        upper_override: FloatArray | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        lower = np.full(self.control_dim, -np.inf)
        upper = np.full(self.control_dim, np.inf)

        if self.control_set.lower is not None:
            lower = np.asarray(self.control_set.lower, dtype=float).copy()
        if self.control_set.upper is not None:
            upper = np.asarray(self.control_set.upper, dtype=float).copy()

        if lower_override is not None:
            override = np.asarray(lower_override, dtype=float)
            if override.shape != (self.control_dim,):
                raise ValueError(
                    f"lower_override must have shape ({self.control_dim},), got {override.shape}."
                )
            if np.any(np.isnan(override)):
                raise ValueError("lower_override must not contain NaN.")
            lower = np.maximum(lower, override)

        if upper_override is not None:
            override = np.asarray(upper_override, dtype=float)
            if override.shape != (self.control_dim,):
                raise ValueError(
                    f"upper_override must have shape ({self.control_dim},), got {override.shape}."
                )
            if np.any(np.isnan(override)):
                raise ValueError("upper_override must not contain NaN.")
            upper = np.minimum(upper, override)

        if np.any(lower > upper):
            indices = np.flatnonzero(lower > upper)
            raise CLFQPSolverError(
                f"dynamic control bounds are inconsistent at indices {indices.tolist()}."
            )

        return lower, upper

    def _direct_box_diagonal_solution(
        self,
        clf: BacksteppingCLFEvaluation,
        *,
        control_reference: FloatArray | None = None,
        constraint_drift: float | None = None,
        dissipation_rate: float | None = None,
        lower_override: FloatArray | None = None,
        upper_override: FloatArray | None = None,
    ) -> tuple[FloatArray, float]:
        """Solve a diagonal-cost box CLF-QP from its scalar KKT multiplier.

        With diagonal ``H`` and no general linear input constraints, stationarity
        gives

            u_k(lambda) = clip(u_ref,k - lambda b_k / H_kk, l_k, u_k),
            delta(lambda)
              = max(0, (lambda - p_1) / p_2).

        If the CLF constraint is active, the unique multiplier is the root of

            phi(lambda)
              = a_c + d
                + b.T u(lambda)
                - max(0, (lambda - p_1) / p_2).

        ``phi`` is continuous and monotonically decreasing, so a scalar
        bisection is sufficient.  This avoids the severe scaling difficulties
        that generic ADMM-based QP solvers can encounter when logarithmic
        barrier gradients become large near a sensing-domain boundary.
        """
        if not self.uses_direct_box_solver:
            raise RuntimeError(
                "direct box solver requires diagonal control cost and "
                "no general input inequalities."
            )

        gradient = np.asarray(clf.control_gradient, dtype=float)
        if gradient.shape != (self.control_dim,):
            raise ValueError("CLF control-gradient dimension does not match the QP.")
        if not np.all(np.isfinite(gradient)):
            raise CLFQPSolverError("CLF control gradient is non-finite before solving.")

        drift, decay = self._constraint_terms(
            clf,
            constraint_drift=constraint_drift,
            dissipation_rate=dissipation_rate,
        )
        constant = drift + decay
        if not np.isfinite(constant):
            raise CLFQPSolverError(
                "CLF constraint drift plus dissipation term is non-finite before solving."
            )

        diagonal = np.diag(self.control_weight)
        reference = self._control_reference(control_reference)
        lower, upper = self._control_bounds(
            lower_override=lower_override,
            upper_override=upper_override,
        )

        def control_at(multiplier: float) -> FloatArray:
            with np.errstate(over="ignore", invalid="ignore"):
                unconstrained = reference - multiplier * gradient / diagonal
            return np.clip(unconstrained, lower, upper)

        def slack_at(multiplier: float) -> float:
            return max(
                0.0,
                (multiplier - self.slack_linear_penalty) / self.slack_penalty,
            )

        def phi(multiplier: float) -> float:
            control = control_at(multiplier)
            value = constant + float(gradient @ control) - slack_at(multiplier)
            if np.isnan(value):
                raise CLFQPSolverError("direct CLF-QP multiplier equation became NaN.")
            return value

        control_zero = control_at(0.0)
        phi_zero = constant + float(gradient @ control_zero)

        # If the CLF inequality is already satisfied at the unconstrained
        # minimum-effort input, lambda = delta = 0 is optimal.
        if phi_zero <= 0.0:
            return control_zero, 0.0

        # Because b.T u(lambda) is non-increasing in lambda and
        #
        #   delta(lambda) = max(0, (lambda - p_1) / p_2),
        #
        # choosing lambda = p_1 + p_2 * phi(0) guarantees
        #
        #   phi(lambda) <= phi(0) - phi(0) = 0.
        #
        # Thus this is a deterministic root bracket even with the exact
        # linear slack penalty.
        multiplier_upper = self.slack_linear_penalty + self.slack_penalty * phi_zero
        multiplier_upper *= 1.0 + 1e-12
        multiplier_upper = max(multiplier_upper, np.finfo(float).tiny)

        if not np.isfinite(multiplier_upper):
            raise CLFQPSolverError("direct CLF-QP multiplier bracket overflowed.")

        phi_upper = phi(multiplier_upper)

        # Extremely ill-scaled floating-point cases may make the theoretical
        # bracket land infinitesimally above zero.  Expand deterministically.
        expansion_count = 0
        while phi_upper > 0.0:
            multiplier_upper *= 2.0
            expansion_count += 1
            if not np.isfinite(multiplier_upper) or expansion_count > 64:
                raise CLFQPSolverError("could not bracket the direct CLF-QP multiplier.")
            phi_upper = phi(multiplier_upper)

        multiplier_lower = 0.0

        # Eighty bisection iterations are inexpensive for an eight-thruster
        # controller and drive the scalar multiplier to near machine precision.
        for _ in range(80):
            multiplier_mid = 0.5 * (multiplier_lower + multiplier_upper)
            if phi(multiplier_mid) > 0.0:
                multiplier_lower = multiplier_mid
            else:
                multiplier_upper = multiplier_mid

        # Use the feasible side of the bracket.
        multiplier = multiplier_upper
        control = control_at(multiplier)
        slack = slack_at(multiplier)

        return control, float(slack)

    def _external_solution(
        self,
        problem: CLFQPProblem,
    ) -> tuple[FloatArray, float]:
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

        return (
            solution[: self.control_dim],
            max(float(solution[-1]), 0.0),
        )

    def solve(
        self,
        clf: BacksteppingCLFEvaluation,
        *,
        control_reference: FloatArray | None = None,
        constraint_drift: float | None = None,
        dissipation_rate: float | None = None,
        lower_override: FloatArray | None = None,
        upper_override: FloatArray | None = None,
    ) -> CLFQPResult:
        """Solve the soft affine-dissipation QP.

        ``control_reference`` changes the control objective to

            1/2 (u - u_ref).T H (u - u_ref).

        The optional pair ``(constraint_drift, dissipation_rate)`` replaces the
        generic CLF constraint with

            constraint_drift + b.T u
                <= -dissipation_rate + delta.

        Optional bound overrides are intersected with the static control box.
        """
        reference = self._control_reference(control_reference)
        drift, decay = self._constraint_terms(
            clf,
            constraint_drift=constraint_drift,
            dissipation_rate=dissipation_rate,
        )
        actuation_feasibility = self.actuation_feasibility(
            clf,
            constraint_drift=constraint_drift,
            dissipation_rate=dissipation_rate,
            lower_override=lower_override,
            upper_override=upper_override,
        )

        if self.uses_direct_box_solver:
            control, slack = self._direct_box_diagonal_solution(
                clf,
                control_reference=reference,
                constraint_drift=constraint_drift,
                dissipation_rate=dissipation_rate,
                lower_override=lower_override,
                upper_override=upper_override,
            )
        else:
            problem = self.build_problem(
                clf,
                control_reference=reference,
                constraint_drift=constraint_drift,
                dissipation_rate=dissipation_rate,
                lower_override=lower_override,
                upper_override=upper_override,
            )
            control, slack = self._external_solution(problem)

        # Report the equivalent bound in full modeled-Wdot coordinates:
        #
        #   Wdot = clf.drift + b.T u
        #        <= (clf.drift - a_c) - d + delta.
        #
        # Its residual is exactly the optimized scalar inequality residual.
        modeled_derivative = clf.derivative(control)
        desired_upper_bound = float(clf.drift - drift - decay + slack)
        residual = modeled_derivative - desired_upper_bound

        residual_scale = max(
            1.0,
            abs(modeled_derivative),
            abs(desired_upper_bound),
        )

        if self.uses_direct_box_solver:
            residual_tolerance = 1e-8 * residual_scale
        elif self.solver == "osqp":
            eps_abs = float(self.solver_options.get("eps_abs", 1e-7))
            eps_rel = float(self.solver_options.get("eps_rel", 1e-7))
            residual_tolerance = 10.0 * (eps_abs + eps_rel * residual_scale)
        else:
            residual_tolerance = 1e-7 * residual_scale

        if residual > residual_tolerance:
            raise CLFQPSolverError(
                "CLF-QP solution violates its affine dissipation inequality "
                "beyond the numerical tolerance associated with the selected "
                f"solver (residual={residual:.3e}, "
                f"tolerance={residual_tolerance:.3e})."
            )

        control_error = control - reference
        objective = float(
            0.5 * control_error @ self.control_weight @ control_error
            + self.slack_linear_penalty * slack
            + 0.5 * self.slack_penalty * slack**2
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

