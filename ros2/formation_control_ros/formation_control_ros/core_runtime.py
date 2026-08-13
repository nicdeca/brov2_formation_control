"""Stateful ROS-side runtime around the ROS-independent controller core.

No controller equations are reimplemented here.  This module only:

* constructs the already validated core objects;
* owns command-filter and adaptive-domain states between ROS timer ticks;
* evaluates the core controller;
* advances those auxiliary states;
* packages diagnostics for ROS publication.

The same class can therefore also be unit-tested without spinning an rclpy
executor.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from formation_control.actuation import BlueROV2HeavyThrusterAllocation
from formation_control.constraints import (
    DistanceDomain,
    FieldOfViewDomain,
    HorizontalFieldOfViewConstraint,
    MaximumDistanceConstraint,
    MinimumDistanceConstraint,
    NormalizedImagePoint,
    VerticalFieldOfViewConstraint,
    evaluate_sensing_constraint_kinematics,
)
from formation_control.control import (
    BlueROV2ControlSpace,
    FunnelRelaxationPolicy,
    build_bluerov2_controller_design,
)
from formation_control.control.bluerov2_leader import (
    BlueROV2LeaderController,
    LeaderTrajectorySample,
)
from formation_control.experiment import (
    follower_snapshot_values,
    leader_snapshot_values,
)
from formation_control.geometry import (
    PinholeCamera,
    rotation_matrix_from_quaternion,
)
from formation_control.models import BlueROV2Model
from formation_control.potentials import (
    AdaptiveConstraintBarrierPotential,
    ConstraintBarrierPotential,
    EdgePotential,
    ImageCenteringPotential,
    RelativePositionPotential,
)
from formation_control.simulation import RK4Integrator
from formation_control.workspace import (
    AxisAlignedWorkspaceDomain,
    WorkspaceBarrierPotential,
    WorkspaceRelaxationPolicy,
)


@dataclass(frozen=True, slots=True)
class CoreControllerConfig:
    """Core tuning shared by leader and follower ROS nodes."""

    control_space: BlueROV2ControlSpace = "thruster"
    thruster_voltage: int = 16
    thrust_derating: float = 1.0
    virtual_linear_speed_limit: float = 1.5
    virtual_angular_speed_limit: float = 2.0
    slack_linear_penalty: float = 100.0
    slack_quadratic_penalty: float = 5e3
    alpha_gain: float = 0.8


@dataclass(frozen=True, slots=True)
class WorkspaceConfig:
    """Switchable pool-workspace barrier and relaxation configuration."""

    enabled: bool = False
    adaptive: bool = True
    physical_lower: tuple[float, float, float] = (-3.125, -1.225, -96.58)
    physical_upper: tuple[float, float, float] = (0.825, 5.575, -94.20)
    conservative_lower: tuple[float, float, float] = (-2.975, -1.075, -96.38)
    conservative_upper: tuple[float, float, float] = (0.675, 5.425, -94.75)
    barrier_weight: float = 0.10
    reference_margin: float = 0.05
    relaxation_recovery_gain: float = 0.8
    relaxation_domain_margin_ratio: float = 0.10
    minimum_constraint_margin: float = 1e-3


@dataclass(frozen=True, slots=True)
class FollowerTaskConfig:
    desired_relative_position: tuple[float, float, float] = (1.8, 0.0, 0.0)
    d_min: float = 0.5
    d_max: float = 3.6
    d_min_conservative: float = 0.8
    d_max_conservative: float = 3.0
    alpha_h_conservative: float = 0.72
    alpha_v_conservative: float = 0.72
    horizontal_half_angle_deg: float = 45.0
    vertical_half_angle_deg: float = 30.0
    formation_gain: float = 1.0
    image_horizontal_gain: float = 0.7
    image_vertical_gain: float = 0.7
    collision_barrier_weight: float = 0.18
    range_barrier_weight: float = 0.18
    horizontal_fov_barrier_weight: float = 0.25
    vertical_fov_barrier_weight: float = 0.25
    adaptive: bool = True
    relaxation_recovery_gain: float = 0.8
    relaxation_domain_margin_ratio: float = 0.1
    use_parent_velocity_in_clf: bool = False


@dataclass(frozen=True)
class ControllerDiagnostics:
    slack: float
    required_slack: float | None
    actuation_margin: float | None
    thruster_utilization: float

    # Existing four-channel sensing-domain diagnostics.
    relaxation_state: np.ndarray
    conservative_constraint_values: np.ndarray
    minimum_physical_margin: float

    # Separate six-channel workspace diagnostics.  Keeping these fields
    # independent preserves the meaning of the validated sensing topics.
    workspace_barrier_enabled: bool
    workspace_adaptive: bool
    workspace_barrier_value: float
    workspace_relaxation_state: np.ndarray
    workspace_conservative_constraint_values: np.ndarray
    workspace_physical_constraint_values: np.ndarray
    workspace_minimum_physical_margin: float


@dataclass(frozen=True)
class CoreStepResult:
    wrench_body: np.ndarray
    diagnostics: ControllerDiagnostics
    snapshot: Mapping[str, object] | None = None


@dataclass(frozen=True)
class _AdaptiveTemplates:
    collision: AdaptiveConstraintBarrierPotential[np.ndarray]
    sensing_range: AdaptiveConstraintBarrierPotential[np.ndarray]
    horizontal_fov: AdaptiveConstraintBarrierPotential[NormalizedImagePoint]
    vertical_fov: AdaptiveConstraintBarrierPotential[NormalizedImagePoint]


def _inertial_linear_velocity(
    state: np.ndarray,
) -> np.ndarray:
    quaternion = state[3:7]
    rotation = rotation_matrix_from_quaternion(quaternion)
    return rotation @ state[7:10]


def _workspace_generalized_gradient(
    state: np.ndarray,
    position_gradient: np.ndarray,
) -> np.ndarray:
    rotation = rotation_matrix_from_quaternion(state[3:7])
    return np.concatenate(
        (rotation.T @ np.asarray(position_gradient, dtype=float), np.zeros(3))
    )


class _WorkspaceRuntime:
    """Shared stateful workspace barrier used by leader and followers."""

    def __init__(self, config: WorkspaceConfig) -> None:
        self.config = config
        self.domain = AxisAlignedWorkspaceDomain(
            physical_lower=np.asarray(config.physical_lower, dtype=float),
            physical_upper=np.asarray(config.physical_upper, dtype=float),
            conservative_lower=np.asarray(config.conservative_lower, dtype=float),
            conservative_upper=np.asarray(config.conservative_upper, dtype=float),
        )
        self.potential = WorkspaceBarrierPotential(
            domain=self.domain,
            weight=config.barrier_weight,
            reference_margin=config.reference_margin,
        )
        self.relaxation = WorkspaceRelaxationPolicy(
            maximum_enlargement=self.domain.maximum_enlargement,
            recovery_gain=config.relaxation_recovery_gain,
            domain_margin_ratio=config.relaxation_domain_margin_ratio,
            minimum_constraint_margin=config.minimum_constraint_margin,
        )
        self.state = np.zeros(6, dtype=float)

    def set_state(self, state: np.ndarray) -> None:
        self.state = self.relaxation.validate_state(state).copy()

    def project_to_current_domain(self, position: np.ndarray) -> None:
        if not (self.config.enabled and self.config.adaptive):
            return
        projected, _ = self.relaxation.project_to_current_domain(
            self.state,
            self.domain.conservative_values(position),
        )
        self.state = projected

    def evaluate(self, state: np.ndarray, reference_position: np.ndarray):
        if not self.config.enabled:
            return None
        self.project_to_current_domain(state[:3])
        return self.potential.evaluate(
            state[:3],
            reference_position,
            self.state,
        )

    def selected_rate(self, state: np.ndarray, *, dt: float) -> np.ndarray:
        if not (self.config.enabled and self.config.adaptive):
            return np.zeros(6, dtype=float)
        evaluation = self.relaxation.evaluate(
            self.state,
            conservative_values=self.domain.conservative_values(state[:3]),
            conservative_rates=self.domain.conservative_rates(
                _inertial_linear_velocity(state)
            ),
            sample_time=dt,
        )
        return np.asarray(evaluation.selected_rate, dtype=float)

    def advance(self, rate: np.ndarray, *, dt: float) -> None:
        if not (self.config.enabled and self.config.adaptive):
            return
        self.state = np.clip(
            self.state + dt * np.asarray(rate, dtype=float),
            0.0,
            1.0,
        )

    def reference_bounds(self) -> tuple[np.ndarray, np.ndarray] | None:
        if not self.config.enabled:
            return None
        lower, upper = self.domain.adaptive_bounds(self.state)
        margin = self.config.reference_margin
        return lower + margin, upper - margin

    def diagnostics(
        self,
        evaluation,
        *,
        relaxation_state: np.ndarray | None = None,
    ) -> dict[str, object]:
        if evaluation is None:
            return {
                "workspace_barrier_enabled": False,
                "workspace_adaptive": False,
                "workspace_barrier_value": 0.0,
                "workspace_relaxation_state": np.zeros(6),
                "workspace_conservative_constraint_values": np.full(6, np.nan),
                "workspace_physical_constraint_values": np.full(6, np.nan),
                "workspace_minimum_physical_margin": np.nan,
            }
        state_for_diagnostics = (
            self.state
            if relaxation_state is None
            else self.relaxation.validate_state(relaxation_state)
        )
        return {
            "workspace_barrier_enabled": True,
            "workspace_adaptive": bool(self.config.adaptive),
            "workspace_barrier_value": float(evaluation.value),
            "workspace_relaxation_state": state_for_diagnostics.copy(),
            "workspace_conservative_constraint_values": (
                np.asarray(evaluation.conservative_values, dtype=float).copy()
            ),
            "workspace_physical_constraint_values": (
                np.asarray(evaluation.physical_values, dtype=float).copy()
            ),
            "workspace_minimum_physical_margin": float(
                evaluation.minimum_physical_margin
            ),
        }


class FollowerCoreRuntime:
    """Stateful one-parent follower controller."""

    def __init__(
        self,
        controller_config: CoreControllerConfig,
        task_config: FollowerTaskConfig,
        workspace_config: WorkspaceConfig = WorkspaceConfig(),
    ) -> None:
        self.controller_config = controller_config
        self.task_config = task_config
        self.workspace_config = workspace_config
        self._workspace = _WorkspaceRuntime(workspace_config)

        self.model = BlueROV2Model()
        self.allocation = BlueROV2HeavyThrusterAllocation.default_45deg(
            voltage=controller_config.thruster_voltage,
            derating=controller_config.thrust_derating,
        )
        self.design = build_bluerov2_controller_design(
            self.model,
            self.allocation,
            control_space=controller_config.control_space,
            virtual_velocity_norm_limits=np.array(
                [
                    controller_config.virtual_linear_speed_limit,
                    controller_config.virtual_angular_speed_limit,
                ],
                dtype=float,
            ),
            slack_linear_penalty=controller_config.slack_linear_penalty,
            slack_penalty=controller_config.slack_quadratic_penalty,
            alpha_gain=controller_config.alpha_gain,
        )
        self.controller = self.design.agent_controller

        self.distance_domain = DistanceDomain(
            d_min=task_config.d_min,
            d_max=task_config.d_max,
            d_min_conservative=task_config.d_min_conservative,
            d_max_conservative=task_config.d_max_conservative,
        )
        self.fov_domain = FieldOfViewDomain(
            alpha_h_conservative=task_config.alpha_h_conservative,
            alpha_v_conservative=task_config.alpha_v_conservative,
        )
        self.camera = PinholeCamera.from_degrees(
            horizontal_half_angle=task_config.horizontal_half_angle_deg,
            vertical_half_angle=task_config.vertical_half_angle_deg,
        )

        self._desired_relative_position = self._validate_desired_relative_position(
            task_config.desired_relative_position
        )
        desired_relative = self._desired_relative_position
        desired_image = NormalizedImagePoint(0.0, 0.0)
        self._templates = _AdaptiveTemplates(
            collision=AdaptiveConstraintBarrierPotential(
                constraint=MinimumDistanceConstraint(
                    self.distance_domain.d_min_conservative
                ),
                reference_state=desired_relative,
                weight=task_config.collision_barrier_weight,
            ),
            sensing_range=AdaptiveConstraintBarrierPotential(
                constraint=MaximumDistanceConstraint(
                    self.distance_domain.d_max_conservative
                ),
                reference_state=desired_relative,
                weight=task_config.range_barrier_weight,
            ),
            horizontal_fov=AdaptiveConstraintBarrierPotential(
                constraint=HorizontalFieldOfViewConstraint(
                    self.fov_domain.alpha_h_conservative
                ),
                reference_state=desired_image,
                weight=task_config.horizontal_fov_barrier_weight,
            ),
            vertical_fov=AdaptiveConstraintBarrierPotential(
                constraint=VerticalFieldOfViewConstraint(
                    self.fov_domain.alpha_v_conservative
                ),
                reference_state=desired_image,
                weight=task_config.vertical_fov_barrier_weight,
            ),
        )

        self.relaxation = FunnelRelaxationPolicy(
            maximum_enlargement=np.array(
                [
                    self.distance_domain.collision_enlargement_max,
                    self.distance_domain.range_enlargement_max,
                    self.fov_domain.horizontal_enlargement_max,
                    self.fov_domain.vertical_enlargement_max,
                ],
                dtype=float,
            ),
            recovery_gain=task_config.relaxation_recovery_gain,
            domain_margin_ratio=task_config.relaxation_domain_margin_ratio,
            minimum_constraint_margin=1e-5,
        )
        self._enabled_relaxation = np.ones(4, dtype=bool)
        self._relaxation_state = np.zeros(4)
        self._filter_state: np.ndarray | None = None
        self._integrator = RK4Integrator()

    @property
    def initialized(self) -> bool:
        return self._filter_state is not None

    @property
    def relaxation_state(self) -> np.ndarray:
        return self._relaxation_state.copy()

    @property
    def workspace_relaxation_state(self) -> np.ndarray:
        return self._workspace.state.copy()

    def set_workspace_relaxation_state(self, state: np.ndarray) -> None:
        """Transfer workspace relaxation across experiment-phase runtimes."""
        self._workspace.set_state(state)

    @property
    def desired_relative_position(self) -> np.ndarray:
        """Current parent-minus-follower formation reference in core NWU."""
        return self._desired_relative_position.copy()

    def _validate_desired_relative_position(
        self,
        desired_relative_position,
    ) -> np.ndarray:
        desired = np.asarray(desired_relative_position, dtype=float).reshape(-1)
        if desired.shape != (3,):
            raise ValueError(
                "desired_relative_position must contain exactly three values."
            )
        if not np.all(np.isfinite(desired)):
            raise ValueError(
                "desired_relative_position must contain only finite values."
            )

        distance = float(np.linalg.norm(desired))
        if not (
            self.distance_domain.d_min_conservative
            < distance
            < self.distance_domain.d_max_conservative
        ):
            raise ValueError(
                "desired_relative_position must lie strictly inside the "
                "conservative distance domain: "
                f"{self.distance_domain.d_min_conservative} < ||d|| < "
                f"{self.distance_domain.d_max_conservative}; got ||d||="
                f"{distance:.6g}."
            )
        return desired.copy()

    def set_desired_relative_position(
        self,
        desired_relative_position,
    ) -> None:
        """Update the formation reference without resetting controller state.

        Only the desired formation and the two distance-barrier reference
        templates depend on this vector. Command-filter and adaptive-domain
        states are intentionally preserved so an online formation change is
        handled as a reference switch rather than a controller reset.
        """
        desired = self._validate_desired_relative_position(
            desired_relative_position
        )
        cfg = self.task_config

        self._desired_relative_position = desired
        self._templates = _AdaptiveTemplates(
            collision=AdaptiveConstraintBarrierPotential(
                constraint=MinimumDistanceConstraint(
                    self.distance_domain.d_min_conservative
                ),
                reference_state=desired,
                weight=cfg.collision_barrier_weight,
            ),
            sensing_range=AdaptiveConstraintBarrierPotential(
                constraint=MaximumDistanceConstraint(
                    self.distance_domain.d_max_conservative
                ),
                reference_state=desired,
                weight=cfg.range_barrier_weight,
            ),
            horizontal_fov=self._templates.horizontal_fov,
            vertical_fov=self._templates.vertical_fov,
        )

    def _edge_potential(self) -> EdgePotential:
        cfg = self.task_config
        desired_relative = self._desired_relative_position
        desired_image = NormalizedImagePoint(0.0, 0.0)

        if cfg.adaptive:
            enlargement = self.relaxation.enlargement(self._relaxation_state)
            collision = self._templates.collision.bind(float(enlargement[0]))
            sensing_range = self._templates.sensing_range.bind(float(enlargement[1]))
            horizontal = self._templates.horizontal_fov.bind(float(enlargement[2]))
            vertical = self._templates.vertical_fov.bind(float(enlargement[3]))
        else:
            collision = ConstraintBarrierPotential.from_reference(
                MinimumDistanceConstraint(self.distance_domain.d_min_conservative),
                desired_relative,
                weight=cfg.collision_barrier_weight,
            )
            sensing_range = ConstraintBarrierPotential.from_reference(
                MaximumDistanceConstraint(self.distance_domain.d_max_conservative),
                desired_relative,
                weight=cfg.range_barrier_weight,
            )
            horizontal = ConstraintBarrierPotential.from_reference(
                HorizontalFieldOfViewConstraint(self.fov_domain.alpha_h_conservative),
                desired_image,
                weight=cfg.horizontal_fov_barrier_weight,
            )
            vertical = ConstraintBarrierPotential.from_reference(
                VerticalFieldOfViewConstraint(self.fov_domain.alpha_v_conservative),
                desired_image,
                weight=cfg.vertical_fov_barrier_weight,
            )

        return EdgePotential(
            formation=RelativePositionPotential.isotropic(
                desired_relative,
                gain=cfg.formation_gain,
            ),
            image_centering=ImageCenteringPotential(
                horizontal_gain=cfg.image_horizontal_gain,
                vertical_gain=cfg.image_vertical_gain,
            ),
            collision_barrier=collision,
            range_barrier=sensing_range,
            horizontal_fov_barrier=horizontal,
            vertical_fov_barrier=vertical,
            camera=self.camera,
        )

    def initialize(
        self,
        follower_state: np.ndarray,
        parent_state: np.ndarray,
    ) -> None:
        if self.task_config.adaptive:
            kinematics = evaluate_sensing_constraint_kinematics(
                self.model,
                self.camera,
                self.distance_domain,
                self.fov_domain,
                follower_state,
                parent_state,
            )
            projected, _ = self.relaxation.project_to_current_domain(
                self._relaxation_state,
                kinematics.values,
                enabled=self._enabled_relaxation,
            )
            self._relaxation_state = projected

        workspace_reference = (
            parent_state[:3] - self._desired_relative_position
        )
        workspace_evaluation = self._workspace.evaluate(
            follower_state,
            workspace_reference,
        )
        workspace_gradient = (
            None
            if workspace_evaluation is None
            else _workspace_generalized_gradient(
                follower_state,
                workspace_evaluation.position_gradient,
            )
        )

        self._filter_state = self.controller.initialize_filter(
            follower_state=follower_state,
            parent_position=parent_state[:3],
            edge_potential=self._edge_potential(),
            configuration_gradient_offset=workspace_gradient,
        )

    def step(
        self,
        follower_state: np.ndarray,
        parent_state: np.ndarray,
        *,
        dt: float,
    ) -> CoreStepResult:
        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        if not self.initialized:
            self.initialize(follower_state, parent_state)
        assert self._filter_state is not None

        # Match the timing convention used by the rich pure-Python example:
        # include sensing/adaptation and the physical controller evaluation,
        # but exclude diagnostic packing and state integration.
        start_time = perf_counter()

        kinematics = evaluate_sensing_constraint_kinematics(
            self.model,
            self.camera,
            self.distance_domain,
            self.fov_domain,
            follower_state,
            parent_state,
        )

        if self.task_config.adaptive:
            projected, _ = self.relaxation.project_to_current_domain(
                self._relaxation_state,
                kinematics.values,
                enabled=self._enabled_relaxation,
            )
            self._relaxation_state = projected

        # This is the normalized domain state actually used by this control
        # evaluation.  Keep it separate from the post-step state s_{k+1}.
        relaxation_state = self._relaxation_state.copy()

        parent_velocity_argument = None
        parent_velocity_for_snapshot = np.zeros(3, dtype=float)
        if self.task_config.use_parent_velocity_in_clf:
            parent_velocity_for_snapshot = _inertial_linear_velocity(
                parent_state
            )
            parent_velocity_argument = parent_velocity_for_snapshot

        workspace_reference = (
            parent_state[:3] - self._desired_relative_position
        )
        workspace_evaluation = self._workspace.evaluate(
            follower_state,
            workspace_reference,
        )
        workspace_gradient = (
            None
            if workspace_evaluation is None
            else _workspace_generalized_gradient(
                follower_state,
                workspace_evaluation.position_gradient,
            )
        )

        evaluation = self.controller.evaluate(
            follower_state=follower_state,
            parent_position=parent_state[:3],
            parent_linear_velocity_inertial=parent_velocity_argument,
            edge_potential=self._edge_potential(),
            filter_state=self._filter_state,
            configuration_value_offset=(
                0.0
                if workspace_evaluation is None
                else float(workspace_evaluation.value)
            ),
            configuration_gradient_offset=workspace_gradient,
        )

        workspace_relaxation_state = self._workspace.state.copy()
        workspace_relaxation_rate = self._workspace.selected_rate(
            follower_state,
            dt=dt,
        )

        relaxation_rate = np.zeros(4, dtype=float)
        if self.task_config.adaptive:
            relaxation_evaluation = self.relaxation.evaluate(
                relaxation_state,
                conservative_values=kinematics.values,
                conservative_rates=kinematics.rates,
                enabled=self._enabled_relaxation,
                sample_time=dt,
            )
            relaxation_rate = np.asarray(
                relaxation_evaluation.selected_rate,
                dtype=float,
            ).copy()

        controller_time_s = perf_counter() - start_time

        # Build the complete experiment record from this SAME evaluation.
        # No CLF-QP or controller mathematics is rerun here.  The helper is
        # deliberately restricted to the validated 8-thruster control space.
        snapshot = None
        if self.controller_config.control_space == "thruster":
            snapshot = follower_snapshot_values(
                model=self.model,
                allocation=self.allocation,
                camera=self.camera,
                follower_state=follower_state,
                parent_position=parent_state[:3],
                parent_velocity_inertial=parent_velocity_for_snapshot,
                desired_relative_position=self._desired_relative_position,
                evaluation=evaluation,
                conservative_values=kinematics.values,
                relaxation_state=relaxation_state,
                relaxation_rate=relaxation_rate,
                controller_time_s=controller_time_s,
                fallback=False,
                distance_domain=self.distance_domain,
                fov_domain=self.fov_domain,
                constraints_enabled=self._enabled_relaxation,
                adaptive_enabled=self.task_config.adaptive,
                domain_margin_ratio=(
                    self.task_config.relaxation_domain_margin_ratio
                ),
            )

        # Advance auxiliary states only after the atomic sample has been
        # assembled, so snapshot[s] and snapshot[s_dot] correspond to x_k and
        # the control input generated at the same timer tick.
        if self.task_config.adaptive:
            self._relaxation_state = np.clip(
                relaxation_state + dt * relaxation_rate,
                0.0,
                1.0,
            )

        self._workspace.state = workspace_relaxation_state
        self._workspace.advance(workspace_relaxation_rate, dt=dt)

        self._filter_state = self._integrator.step(
            self.controller.dynamics_controller.command_filter,
            self._filter_state,
            evaluation.controller.filter_command,
            dt,
        )

        if snapshot is not None:
            utilization = float(snapshot["thruster_utilization"])
            physical_margin = float(snapshot["minimum_physical_margin"])
        else:
            thruster_forces = self.design.representative_thruster_forces(
                evaluation
            )
            utilization = float(
                np.max(self.allocation.utilization(thruster_forces))
            )
            physical_margin = np.nan

        return CoreStepResult(
            wrench_body=np.asarray(evaluation.wrench_body, dtype=float),
            diagnostics=ControllerDiagnostics(
                slack=float(evaluation.slack),
                required_slack=evaluation.required_slack,
                actuation_margin=evaluation.actuation_margin,
                thruster_utilization=utilization,
                relaxation_state=relaxation_state,
                conservative_constraint_values=np.asarray(
                    kinematics.values,
                    dtype=float,
                ).copy(),
                minimum_physical_margin=physical_margin,
                **self._workspace.diagnostics(
                    workspace_evaluation,
                    relaxation_state=workspace_relaxation_state,
                ),
            ),
            snapshot=snapshot,
        )



class LeaderCoreRuntime:
    """Stateful leader using the same second-order CLF-QP controller."""

    def __init__(
        self,
        controller_config: CoreControllerConfig,
        *,
        position_gain: float = 1.0,
        attitude_gain: float = 1.0,
        workspace_config: WorkspaceConfig = WorkspaceConfig(),
    ) -> None:
        self.workspace_config = workspace_config
        self._workspace = _WorkspaceRuntime(workspace_config)
        self.model = BlueROV2Model()
        self.allocation = BlueROV2HeavyThrusterAllocation.default_45deg(
            voltage=controller_config.thruster_voltage,
            derating=controller_config.thrust_derating,
        )
        self.design = build_bluerov2_controller_design(
            self.model,
            self.allocation,
            control_space=controller_config.control_space,
            virtual_velocity_norm_limits=np.array(
                [
                    controller_config.virtual_linear_speed_limit,
                    controller_config.virtual_angular_speed_limit,
                ]
            ),
            slack_linear_penalty=controller_config.slack_linear_penalty,
            slack_penalty=controller_config.slack_quadratic_penalty,
            alpha_gain=controller_config.alpha_gain,
        )
        self._position_gain = position_gain
        self._attitude_gain = attitude_gain
        self._leader: BlueROV2LeaderController | None = None
        self._filter_state: np.ndarray | None = None
        self._integrator = RK4Integrator()

    @property
    def initialized(self) -> bool:
        return self._leader is not None

    @property
    def workspace_relaxation_state(self) -> np.ndarray:
        return self._workspace.state.copy()

    def set_workspace_relaxation_state(self, state: np.ndarray) -> None:
        self._workspace.set_state(state)

    def workspace_reference_bounds(
        self,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Bounds used to keep integrated manual references inside the pool."""
        return self._workspace.reference_bounds()

    def initialize(
        self,
        state: np.ndarray,
        reference: LeaderTrajectorySample,
    ) -> None:
        controller = BlueROV2LeaderController.from_initial_state(
            self.design.agent_controller.dynamics_controller,
            self.model,
            state,
            position_gain=self._position_gain * np.eye(3),
            attitude_gain=self._attitude_gain,
        )
        self._leader = controller
        workspace_evaluation = self._workspace.evaluate(
            state,
            reference.position,
        )
        workspace_gradient = (
            None
            if workspace_evaluation is None
            else _workspace_generalized_gradient(
                state,
                workspace_evaluation.position_gradient,
            )
        )
        self._filter_state = controller.initialize_filter(
            state=state,
            reference=reference,
            configuration_gradient_offset=workspace_gradient,
        )

    def step(
        self,
        state: np.ndarray,
        reference: LeaderTrajectorySample,
        *,
        dt: float,
    ) -> CoreStepResult:
        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        if not self.initialized:
            self.initialize(state, reference)
        assert self._leader is not None
        assert self._filter_state is not None

        start_time = perf_counter()
        workspace_evaluation = self._workspace.evaluate(
            state,
            reference.position,
        )
        workspace_gradient = (
            None
            if workspace_evaluation is None
            else _workspace_generalized_gradient(
                state,
                workspace_evaluation.position_gradient,
            )
        )
        evaluation = self._leader.evaluate(
            state=state,
            reference=reference,
            filter_state=self._filter_state,
            configuration_value_offset=(
                0.0
                if workspace_evaluation is None
                else float(workspace_evaluation.value)
            ),
            configuration_gradient_offset=workspace_gradient,
        )
        workspace_relaxation_state = self._workspace.state.copy()
        workspace_relaxation_rate = self._workspace.selected_rate(
            state,
            dt=dt,
        )
        controller_time_s = perf_counter() - start_time

        thruster_forces = self.design.representative_thruster_forces(evaluation)
        utilization = float(np.max(self.allocation.utilization(thruster_forces)))

        snapshot = leader_snapshot_values(
            thruster_forces=thruster_forces,
            body_wrench=np.asarray(evaluation.wrench_body, dtype=float),
            controller_time_s=controller_time_s,
            fallback=False,
            allocation=self.allocation,
            reference_position=np.asarray(reference.position, dtype=float),
            reference_velocity=np.asarray(reference.velocity, dtype=float),
            reference_acceleration=np.asarray(reference.acceleration, dtype=float),
            extra_values={
                "slack": float(evaluation.slack),
                "required_slack": (
                    float(evaluation.required_slack)
                    if evaluation.required_slack is not None
                    else np.nan
                ),
                "actuation_margin": (
                    float(evaluation.actuation_margin)
                    if evaluation.actuation_margin is not None
                    else np.nan
                ),
            },
        )

        self._workspace.state = workspace_relaxation_state
        self._workspace.advance(workspace_relaxation_rate, dt=dt)

        self._filter_state = self._integrator.step(
            self._leader.dynamics_controller.command_filter,
            self._filter_state,
            evaluation.controller.filter_command,
            dt,
        )

        return CoreStepResult(
            wrench_body=np.asarray(evaluation.wrench_body, dtype=float),
            diagnostics=ControllerDiagnostics(
                slack=float(evaluation.slack),
                required_slack=evaluation.required_slack,
                actuation_margin=evaluation.actuation_margin,
                thruster_utilization=utilization,
                relaxation_state=np.zeros(4),
                conservative_constraint_values=np.full(4, np.nan),
                minimum_physical_margin=np.nan,
                **self._workspace.diagnostics(
                    workspace_evaluation,
                    relaxation_state=workspace_relaxation_state,
                ),
            ),
            snapshot=snapshot,
        )

