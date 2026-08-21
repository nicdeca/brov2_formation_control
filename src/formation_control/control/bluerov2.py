"""BlueROV2 adapter for the generic command-filtered CLF-QP controller.

The adapter connects pose-dependent edge potentials to the generic second-order
controller for the 6-DoF marine dynamics

    M nu_dot + h(eta, nu) = B u.

The BlueROV2 state convention is

    x = col(p, q, nu),

where ``q`` is scalar-first and ``nu = col(v, omega)`` is expressed in the
body frame.

For a configuration potential ``V`` with inertial position gradient ``g_p``
and body-frame orientation gradient ``g_R``, the generalized gradient entering
the backstepping design is

    zeta = col(R.T g_p, g_R),

because

    V_dot = zeta.T nu + chi.

The optional term ``chi`` contains the parent translational motion.  The
adapter does not advance the plant, command filter, or adaptive-domain states.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from formation_control.geometry import rotation_matrix_from_quaternion
from formation_control.models import BlueROV2Model
from formation_control.models.base import FloatArray
from formation_control.potentials import EdgePotential, EdgePotentialEvaluation

from .second_order import (
    SecondOrderCLFQPController,
    SecondOrderControllerEvaluation,
)
from .virtual_control import generalized_configuration_gradient


def _vector3(value: FloatArray, *, name: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _vector6(value: FloatArray, *, name: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.shape != (6,):
        raise ValueError(f"{name} must have shape (6,), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


@dataclass(frozen=True)
class BlueROV2AgentEvaluation:
    """Potential and dynamic-control results for one BlueROV2 follower."""

    potential: EdgePotentialEvaluation
    generalized_configuration_gradient: FloatArray
    controller: SecondOrderControllerEvaluation
    input_matrix: FloatArray

    def __post_init__(self) -> None:
        gradient = np.asarray(
            self.generalized_configuration_gradient,
            dtype=float,
        )
        input_matrix = np.asarray(self.input_matrix, dtype=float)

        if gradient.shape != (6,):
            raise ValueError("generalized_configuration_gradient must have shape (6,).")
        if input_matrix.ndim != 2 or input_matrix.shape[0] != 6:
            raise ValueError("input_matrix must have six generalized-wrench rows.")
        if input_matrix.shape[1] != self.controller.control.size:
            raise ValueError("input_matrix columns must match optimized-input dimension.")

        object.__setattr__(
            self,
            "generalized_configuration_gradient",
            gradient,
        )
        object.__setattr__(self, "input_matrix", input_matrix)

    @property
    def optimized_input(self) -> FloatArray:
        """Return the QP decision coordinates.

        With ``B = I`` these are body-wrench coordinates.  With a thruster
        allocation matrix ``B`` these can instead be thruster-force
        coordinates.
        """
        return self.controller.control

    @property
    def wrench_body(self) -> FloatArray:
        """Return the physical generalized wrench applied to the vehicle."""
        return self.input_matrix @ self.optimized_input

    @property
    def slack(self) -> float:
        """Return the optimal CLF-QP relaxation."""
        return self.controller.slack

    @property
    def required_slack(self) -> float | None:
        """Return the minimum relaxation required by actuator limits."""
        return self.controller.required_slack

    @property
    def actuation_margin(self) -> float | None:
        """Return the zero-slack CLF feasibility margin."""
        return self.controller.actuation_margin


@dataclass(frozen=True)
class BlueROV2AgentController:
    """Thin adapter from an edge potential to 6-DoF BlueROV2 dynamics.

    The dynamic controller may optimize either body wrench directly or another
    control vector mapped to wrench by ``BacksteppingCLF.input_matrix``.
    """

    dynamics_controller: SecondOrderCLFQPController
    model: BlueROV2Model

    def __post_init__(self) -> None:
        if self.dynamics_controller.velocity_dim != 6:
            raise ValueError("BlueROV2 dynamics require six generalized-velocity coordinates.")

        if not np.allclose(
            self.dynamics_controller.clf.inertia,
            self.model.mass_matrix,
        ):
            raise ValueError("the backstepping CLF inertia must match the BlueROV2 mass matrix.")

        input_matrix = self.dynamics_controller.clf.input_matrix
        assert input_matrix is not None
        if input_matrix.shape[0] != 6:
            raise ValueError("the CLF input matrix must map optimized inputs to a 6-D wrench.")

    def evaluate_edge_potential(
        self,
        *,
        follower_state: FloatArray,
        parent_position: FloatArray,
        edge_potential: EdgePotential,
    ) -> tuple[EdgePotentialEvaluation, FloatArray]:
        """Evaluate the edge potential and return ``(evaluation, zeta)``."""
        follower_position, quaternion, _ = self.model.split_state(follower_state)
        parent_position = _vector3(
            parent_position,
            name="parent_position",
        )
        rotation = rotation_matrix_from_quaternion(quaternion)

        potential = edge_potential.evaluate(
            observer_position=follower_position,
            observer_rotation=rotation,
            target_position=parent_position,
        )

        zeta = generalized_configuration_gradient(
            rotation,
            potential.observer_position_gradient,
            potential.observer_orientation_gradient_body,
        )

        return potential, zeta

    def initialize_filter(
        self,
        *,
        follower_state: FloatArray,
        parent_position: FloatArray,
        edge_potential: EdgePotential,
        feedforward_velocity_body: FloatArray | None = None,
        configuration_gradient_offset: FloatArray | None = None,
    ) -> FloatArray:
        """Initialize the command filter at the initial virtual command."""
        _, zeta = self.evaluate_edge_potential(
            follower_state=follower_state,
            parent_position=parent_position,
            edge_potential=edge_potential,
        )
        if configuration_gradient_offset is not None:
            zeta = zeta + _vector6(
                configuration_gradient_offset,
                name="configuration_gradient_offset",
            )

        return self.dynamics_controller.initialize_filter(
            zeta,
            feedforward_velocity=feedforward_velocity_body,
        )

    def evaluate(
        self,
        *,
        follower_state: FloatArray,
        parent_position: FloatArray,
        edge_potential: EdgePotential,
        filter_state: FloatArray,
        parent_linear_velocity_inertial: FloatArray | None = None,
        feedforward_velocity_body: FloatArray | None = None,
        configuration_value_offset: float = 0.0,
        configuration_gradient_offset: FloatArray | None = None,
        control_gradient_offset: FloatArray | None = None,
        control_reference: FloatArray | None = None,
        control_lower: FloatArray | None = None,
        control_upper: FloatArray | None = None,
    ) -> BlueROV2AgentEvaluation:
        """Evaluate one BlueROV2 follower control cycle.

        The exact local potential derivative is

            V_dot
            = zeta.T nu
            + grad_{p_j}(V).T p_j_dot.

        If ``parent_linear_velocity_inertial`` is supplied, the second term is
        included in the implemented CLF derivative.  If it is omitted, the QP
        uses zero for that term and the actual parent motion can be treated as
        an interconnection perturbation in the analysis.

        ``configuration_value_offset`` and ``configuration_gradient_offset``
        can add a nonnegative single-agent storage term (for example a
        workspace barrier) without modifying the edge-potential abstraction.
        ``control_gradient_offset`` remains available for auxiliary terms that
        depend directly on optimized inputs.  Unless an explicit
        ``control_reference`` is supplied, the restoring wrench is passed
        to the common controller as the candidate stationary trim wrench;
        its smooth activation is handled there from the local CLF input
        direction.
        """
        _, quaternion, generalized_velocity = self.model.split_state(follower_state)

        potential, zeta = self.evaluate_edge_potential(
            follower_state=follower_state,
            parent_position=parent_position,
            edge_potential=edge_potential,
        )
        if configuration_gradient_offset is not None:
            zeta = zeta + _vector6(
                configuration_gradient_offset,
                name="configuration_gradient_offset",
            )

        if parent_linear_velocity_inertial is None:
            configuration_rate_offset = 0.0
        else:
            parent_velocity = _vector3(
                parent_linear_velocity_inertial,
                name="parent_linear_velocity_inertial",
            )
            configuration_rate_offset = float(potential.target_position_gradient @ parent_velocity)

        if not np.isfinite(configuration_value_offset) or (configuration_value_offset < 0.0):
            raise ValueError("configuration_value_offset must be finite and nonnegative.")

        controller_evaluation = self.dynamics_controller.evaluate(
            configuration_value=(potential.value + configuration_value_offset),
            configuration_gradient=zeta,
            generalized_velocity=generalized_velocity,
            filter_state=filter_state,
            dynamics_bias=self.model.drift_wrench(follower_state),
            configuration_rate_offset=configuration_rate_offset,
            feedforward_velocity=feedforward_velocity_body,
            control_gradient_offset=control_gradient_offset,
            control_reference=control_reference,
            trim_wrench=(
                None
                if control_reference is not None
                else self.model.restoring_wrench(quaternion)
            ),
            control_lower=control_lower,
            control_upper=control_upper,
        )

        input_matrix = self.dynamics_controller.clf.input_matrix
        assert input_matrix is not None

        return BlueROV2AgentEvaluation(
            potential=potential,
            generalized_configuration_gradient=zeta,
            controller=controller_evaluation,
            input_matrix=input_matrix,
        )
