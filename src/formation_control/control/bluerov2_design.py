"""Canonical BlueROV2 Heavy CLF-QP controller designs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from formation_control.actuation import (
    AchievableWrenchPolytope,
    BlueROV2HeavyThrusterAllocation,
    wrench_polytope_from_allocation,
)
from formation_control.models import BlueROV2Model
from formation_control.models.base import FloatArray

from .backstepping_clf import BacksteppingCLF
from .bluerov2 import BlueROV2AgentController, BlueROV2AgentEvaluation
from .class_k import LinearClassK
from .clf_qp import CLFQP
from .control_constraints import PolyhedralControlSet
from .filters import FirstOrderCommandFilter
from .second_order import SecondOrderCLFQPController
from .wrench_space import (
    build_wrench_space_controller,
    equivalent_wrench_weight,
    minimum_effort_allocation_matrix,
)

type BlueROV2ControlSpace = Literal["thruster", "wrench"]


def _default_virtual_gain() -> FloatArray:
    return np.diag([0.55, 0.55, 0.55, 0.8, 0.8, 0.8])


def _default_filter_bandwidth() -> FloatArray:
    return np.array([3.0, 3.0, 3.0, 4.0, 4.0, 4.0])


def _default_virtual_velocity_norm_limits() -> FloatArray:
    # Independent norm limits for translational and rotational corrections.
    return np.array([1.5, 2.0])


@dataclass(frozen=True)
class BlueROV2ControllerDesign:
    """Controller plus common actuation data for one control-coordinate choice."""

    control_space: BlueROV2ControlSpace
    agent_controller: BlueROV2AgentController
    allocation: BlueROV2HeavyThrusterAllocation
    wrench_polytope: AchievableWrenchPolytope
    thruster_weight: FloatArray
    wrench_weight: FloatArray
    minimum_effort_map: FloatArray

    def __post_init__(self) -> None:
        if self.control_space not in ("thruster", "wrench"):
            raise ValueError("control_space must be 'thruster' or 'wrench'.")

    @property
    def decision_dimension(self) -> int:
        return self.agent_controller.dynamics_controller.control_dim

    def representative_thruster_forces(
        self,
        evaluation: BlueROV2AgentEvaluation,
    ) -> FloatArray:
        """Return one physical thruster allocation for diagnostics.

        In thruster space this is exactly the QP decision.  In wrench space the
        QP specifies only the body wrench.  We first use the unconstrained
        minimum-effort allocation; if it violates a thruster bound, a bounded
        exact allocation is recovered for diagnostics.  A real autopilot may
        of course choose a different null-space allocation.
        """
        if self.control_space == "thruster":
            decision = np.asarray(evaluation.optimized_input, dtype=float)
            if decision.ndim != 1 or decision.size < 8:
                raise ValueError("thruster-space decision must contain at least eight inputs.")
            return decision[:8].copy()

        wrench = np.asarray(evaluation.wrench_body, dtype=float)
        forces = self.minimum_effort_map @ wrench

        if self.allocation.contains(forces, tolerance=1e-8):
            return forces

        bounded = self.allocation.bounded_least_squares(wrench)
        if bounded.residual_norm > 1e-5:
            raise RuntimeError(
                "wrench-space QP returned a wrench that could not be "
                "reallocated within the physical thruster limits."
            )
        return bounded.forces


def build_bluerov2_controller_design(
    model: BlueROV2Model,
    allocation: BlueROV2HeavyThrusterAllocation,
    *,
    control_space: BlueROV2ControlSpace = "thruster",
    virtual_gain: FloatArray | None = None,
    virtual_velocity_norm_limits: FloatArray | None = None,
    filter_bandwidth: float | FloatArray | None = None,
    slack_penalty: float = 5e3,
    slack_linear_penalty: float = 100.0,
    alpha_gain: float = 0.8,
    velocity_error_gain: float | FloatArray | None = None,
    trim_activation_on: float = 0.5,
    trim_activation_off: float = 2.0,
    trim_allocation_tolerance: float = 1e-5,
) -> BlueROV2ControllerDesign:
    """Build the canonical Heavy controller in thruster or wrench coordinates.

    ``alpha_gain`` is retained as the scalar default for backward-compatible
    configuration.  In the updated backstepping QP it is interpreted as an
    inverse-time velocity-error decay gain and sets
    ``K_e = alpha_gain * M`` unless ``velocity_error_gain`` is supplied.

    The trim allocator uses only the hydrostatic restoring wrench.  In
    thruster coordinates it first tries the weighted minimum-effort exact
    allocation and falls back to bounded least squares only if required.
    """
    if control_space not in ("thruster", "wrench"):
        raise ValueError("control_space must be 'thruster' or 'wrench'.")

    if virtual_gain is None:
        virtual_gain = _default_virtual_gain()
    if virtual_velocity_norm_limits is None:
        virtual_velocity_norm_limits = _default_virtual_velocity_norm_limits()
    if filter_bandwidth is None:
        filter_bandwidth = _default_filter_bandwidth()

    if velocity_error_gain is None:
        # K_e is a damping-like generalized-wrench / generalized-velocity
        # gain in
        #
        #     e_nu.T K_e e_nu.
        #
        # Using alpha_gain * I (the first compatibility implementation) is
        # dimensionally inconsistent with the marine backstepping dynamics
        # and makes the translational error damping much too weak.  Scale the
        # inertia instead:
        #
        #     K_e = k_e M.
        #
        # Then alpha_gain has units 1/s and directly sets the decay rate of
        # the kinetic velocity-error energy.
        velocity_error_gain_matrix = float(alpha_gain) * model.mass_matrix
    else:
        gain_array = np.asarray(velocity_error_gain, dtype=float)
        if gain_array.ndim == 0:
            velocity_error_gain_matrix = float(gain_array) * np.eye(6)
        else:
            velocity_error_gain_matrix = gain_array

    if not np.isfinite(trim_allocation_tolerance) or trim_allocation_tolerance <= 0.0:
        raise ValueError("trim_allocation_tolerance must be finite and positive.")

    force_scale = max(
        allocation.configuration.force_limits.forward,
        allocation.configuration.force_limits.reverse,
    )
    thruster_weight = (1.0 / force_scale**2) * np.eye(8)

    wrench_polytope = wrench_polytope_from_allocation(allocation)
    wrench_weight = equivalent_wrench_weight(
        allocation.matrix,
        thruster_weight,
    )
    minimum_effort_map = minimum_effort_allocation_matrix(
        allocation.matrix,
        thruster_weight,
    )

    def thruster_trim_allocator(wrench: FloatArray) -> FloatArray | None:
        """Return an exact admissible trim allocation when one is available."""
        wrench = np.asarray(wrench, dtype=float)
        forces = minimum_effort_map @ wrench
        if allocation.contains(forces, tolerance=1e-8):
            return forces

        bounded = allocation.bounded_least_squares(wrench)
        if bounded.residual_norm <= trim_allocation_tolerance:
            return bounded.forces
        return None

    def wrench_trim_allocator(wrench: FloatArray) -> FloatArray:
        return np.asarray(wrench, dtype=float).copy()

    command_filter = FirstOrderCommandFilter(
        signal_dim=6,
        bandwidth=filter_bandwidth,
    )
    alpha = LinearClassK(gain=alpha_gain)

    if control_space == "thruster":
        dynamics_controller = SecondOrderCLFQPController(
            virtual_gain=np.asarray(virtual_gain, dtype=float),
            command_filter=command_filter,
            virtual_velocity_norm_limits=np.asarray(
                virtual_velocity_norm_limits,
                dtype=float,
            ),
            virtual_velocity_group_sizes=(3, 3),
            clf=BacksteppingCLF(
                model.mass_matrix,
                input_matrix=allocation.matrix,
            ),
            qp=CLFQP(
                control_weight=thruster_weight,
                slack_penalty=slack_penalty,
                slack_linear_penalty=slack_linear_penalty,
                alpha=alpha,
                control_set=PolyhedralControlSet.box(
                    allocation.lower_bounds,
                    allocation.upper_bounds,
                ),
            ),
            velocity_error_gain=velocity_error_gain_matrix,
            trim_control_allocator=thruster_trim_allocator,
            trim_activation_on=trim_activation_on,
            trim_activation_off=trim_activation_off,
        )
    else:
        base_controller = build_wrench_space_controller(
            inertia=model.mass_matrix,
            virtual_gain=np.asarray(virtual_gain, dtype=float),
            command_filter=command_filter,
            virtual_velocity_norm_limits=np.asarray(
                virtual_velocity_norm_limits,
                dtype=float,
            ),
            virtual_velocity_group_sizes=(3, 3),
            polytope=wrench_polytope,
            control_weight=wrench_weight,
            slack_penalty=slack_penalty,
            slack_linear_penalty=slack_linear_penalty,
            alpha=alpha,
        ).controller
        dynamics_controller = replace(
            base_controller,
            velocity_error_gain=velocity_error_gain_matrix,
            trim_control_allocator=wrench_trim_allocator,
            trim_activation_on=trim_activation_on,
            trim_activation_off=trim_activation_off,
        )

    return BlueROV2ControllerDesign(
        control_space=control_space,
        agent_controller=BlueROV2AgentController(
            dynamics_controller=dynamics_controller,
            model=model,
        ),
        allocation=allocation,
        wrench_polytope=wrench_polytope,
        thruster_weight=thruster_weight,
        wrench_weight=wrench_weight,
        minimum_effort_map=minimum_effort_map,
    )
