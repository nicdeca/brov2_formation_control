"""Actuation-feasibility diagnostics for affine CLF constraints.

For the zero-slack CLF condition

    a + b.T u <= -alpha(W),

the minimum relaxation that is *physically required* by the actuator set is

    delta_req
      = max(0, a + alpha(W) + min_{u in U} b.T u).

This differs from the optimal slack returned by a soft CLF-QP.  With a finite
slack penalty the optimizer may deliberately use positive slack to save control
effort even when a zero-slack input exists.  ``delta_req`` is therefore the
cleaner signal when domain adaptation is intended to react specifically to
actuation infeasibility.

For a box-constrained input set, the affine minimization is exact and requires
only one sign test per input coordinate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from formation_control.models.base import FloatArray

from .backstepping_clf import BacksteppingCLFEvaluation
from .class_k import ClassKFunction
from .control_constraints import PolyhedralControlSet


@dataclass(frozen=True)
class CLFActuationFeasibility:
    """Minimum achievable CLF derivative under the actuator bounds."""

    minimizing_control: FloatArray
    minimum_modeled_derivative: float
    zero_slack_upper_bound: float
    margin: float
    required_slack: float

    def __post_init__(self) -> None:
        control = np.asarray(self.minimizing_control, dtype=float)
        if control.ndim != 1 or not np.all(np.isfinite(control)):
            raise ValueError("minimizing_control must be a finite one-dimensional array.")
        scalars = (
            self.minimum_modeled_derivative,
            self.zero_slack_upper_bound,
            self.margin,
            self.required_slack,
        )
        if not all(np.isfinite(value) for value in scalars):
            raise ValueError("feasibility diagnostics must be finite.")
        if self.required_slack < -1e-12:
            raise ValueError("required_slack must be nonnegative.")

        object.__setattr__(self, "minimizing_control", control.copy())

    @property
    def zero_slack_feasible(self) -> bool:
        """Whether the zero-relaxation CLF inequality can be satisfied."""
        return self.required_slack <= 1e-12


def box_clf_actuation_feasibility(
    clf: BacksteppingCLFEvaluation,
    *,
    alpha: ClassKFunction,
    control_set: PolyhedralControlSet,
) -> CLFActuationFeasibility:
    """Evaluate exact CLF feasibility for a finite box input set.

    Linear inequalities beyond the box are intentionally rejected.  The online
    BlueROV2 controller uses only the eight physical thruster bounds, for which
    this closed-form computation is exact and essentially free.
    """
    if control_set.inequality_matrix is not None:
        raise ValueError("box_clf_actuation_feasibility requires a box-only control set.")
    if control_set.lower is None or control_set.upper is None:
        raise ValueError("finite lower and upper control bounds are required.")

    lower = np.asarray(control_set.lower, dtype=float)
    upper = np.asarray(control_set.upper, dtype=float)
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("control bounds must be finite.")

    gradient = np.asarray(clf.control_gradient, dtype=float)
    if gradient.shape != (control_set.dimension,):
        raise ValueError("CLF control-gradient dimension does not match the control set.")

    midpoint = 0.5 * (lower + upper)
    minimizing_control = midpoint.copy()
    minimizing_control[gradient > 0.0] = lower[gradient > 0.0]
    minimizing_control[gradient < 0.0] = upper[gradient < 0.0]

    minimum_derivative = clf.derivative(minimizing_control)
    zero_slack_upper_bound = -float(alpha(clf.value))
    margin = zero_slack_upper_bound - minimum_derivative
    required_slack = max(-margin, 0.0)

    return CLFActuationFeasibility(
        minimizing_control=minimizing_control,
        minimum_modeled_derivative=float(minimum_derivative),
        zero_slack_upper_bound=float(zero_slack_upper_bound),
        margin=float(margin),
        required_slack=float(required_slack),
    )
