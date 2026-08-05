"""Six-degree-of-freedom BlueROV2 Heavy dynamics with body-wrench input."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from formation_control.geometry.rotations import (
    normalize_quaternion,
    quaternion_derivative_body_rates,
    quaternion_to_roll_pitch_yaw,
    rotation_matrix_from_quaternion,
    skew,
)

from .base import ContinuousTimeModel, FloatArray


@dataclass(frozen=True, slots=True)
class BlueROV2Parameters:
    """Physical and hydrodynamic parameters of the BlueROV2 Heavy model.

    The defaults reproduce the parameters used by the previous codebase, but
    are stored with positive added-mass and damping magnitudes for clarity.
    """

    water_density: float = 1000.0
    gravity: float = 9.82
    mass: float = 13.5
    volume: float = 0.0134

    inertia_x: float = 0.26
    inertia_y: float = 0.23
    inertia_z: float = 0.37

    cg_x: float = 0.0
    cg_y: float = 0.0
    cg_z: float = 0.0
    cb_x: float = 0.0
    cb_y: float = 0.0
    cb_z: float = -0.01

    added_mass_u: float = 6.36
    added_mass_v: float = 7.12
    added_mass_w: float = 18.68
    added_mass_p: float = 0.189
    added_mass_q: float = 0.135
    added_mass_r: float = 0.222

    linear_damping_u: float = 13.7
    linear_damping_v: float = 0.0
    linear_damping_w: float = 33.0
    linear_damping_p: float = 0.0
    linear_damping_q: float = 0.8
    linear_damping_r: float = 0.0

    quadratic_damping_u: float = 141.0
    quadratic_damping_v: float = 217.0
    quadratic_damping_w: float = 190.0
    quadratic_damping_p: float = 1.19
    quadratic_damping_q: float = 0.47
    quadratic_damping_r: float = 1.5

    def __post_init__(self) -> None:
        positive = {
            "water_density": self.water_density,
            "gravity": self.gravity,
            "mass": self.mass,
            "volume": self.volume,
            "inertia_x": self.inertia_x,
            "inertia_y": self.inertia_y,
            "inertia_z": self.inertia_z,
        }
        for name, value in positive.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive.")

        nonnegative = (
            self.added_mass_diagonal,
            self.linear_damping_diagonal,
            self.quadratic_damping_diagonal,
        )
        if any(np.any(values < 0.0) for values in nonnegative):
            raise ValueError("added-mass and damping magnitudes must be nonnegative.")

    @property
    def center_of_gravity(self) -> FloatArray:
        return np.array([self.cg_x, self.cg_y, self.cg_z], dtype=float)

    @property
    def center_of_buoyancy(self) -> FloatArray:
        return np.array([self.cb_x, self.cb_y, self.cb_z], dtype=float)

    @property
    def rigid_body_inertia_diagonal(self) -> FloatArray:
        return np.array([self.inertia_x, self.inertia_y, self.inertia_z], dtype=float)

    @property
    def added_mass_diagonal(self) -> FloatArray:
        return np.array(
            [
                self.added_mass_u,
                self.added_mass_v,
                self.added_mass_w,
                self.added_mass_p,
                self.added_mass_q,
                self.added_mass_r,
            ],
            dtype=float,
        )

    @property
    def linear_damping_diagonal(self) -> FloatArray:
        return np.array(
            [
                self.linear_damping_u,
                self.linear_damping_v,
                self.linear_damping_w,
                self.linear_damping_p,
                self.linear_damping_q,
                self.linear_damping_r,
            ],
            dtype=float,
        )

    @property
    def quadratic_damping_diagonal(self) -> FloatArray:
        return np.array(
            [
                self.quadratic_damping_u,
                self.quadratic_damping_v,
                self.quadratic_damping_w,
                self.quadratic_damping_p,
                self.quadratic_damping_q,
                self.quadratic_damping_r,
            ],
            dtype=float,
        )


class BlueROV2Model(ContinuousTimeModel):
    """BlueROV2 Heavy model with state ``x = col(p, q, nu)``.

    State convention
    ----------------
    ``p``
        Position in the inertial frame, shape ``(3,)``.
    ``q``
        Scalar-first unit quaternion ``[q_w, q_x, q_y, q_z]`` mapping body
        coordinates to inertial coordinates, shape ``(4,)``.
    ``nu``
        Body-fixed generalized velocity ``col(v, omega)``, shape ``(6,)``.

    Input convention
    ----------------
    The input is the body-frame generalized wrench
    ``tau = col(force, torque)``.  Thruster allocation is intentionally kept
    outside the dynamics model.
    """

    POSITION_SLICE = slice(0, 3)
    QUATERNION_SLICE = slice(3, 7)
    VELOCITY_SLICE = slice(7, 13)

    def __init__(
        self,
        parameters: BlueROV2Parameters | None = None,
        *,
        current_velocity_inertial: FloatArray | None = None,
    ) -> None:
        self.parameters = parameters or BlueROV2Parameters()
        if current_velocity_inertial is None:
            current_velocity_inertial = np.zeros(3)
        self._current_velocity_inertial = np.asarray(
            current_velocity_inertial, dtype=float
        ).reshape(3)

        self._rigid_body_mass = self._build_rigid_body_mass_matrix()
        self._added_mass = np.diag(self.parameters.added_mass_diagonal)
        self._mass = self._rigid_body_mass + self._added_mass

        if not np.allclose(self._mass, self._mass.T):
            raise ValueError("generalized inertia matrix must be symmetric.")
        if np.min(np.linalg.eigvalsh(self._mass)) <= 0.0:
            raise ValueError("generalized inertia matrix must be positive definite.")

    @property
    def state_dim(self) -> int:
        return 13

    @property
    def input_dim(self) -> int:
        return 6

    @property
    def current_velocity_inertial(self) -> FloatArray:
        return self._current_velocity_inertial.copy()

    @property
    def rigid_body_mass_matrix(self) -> FloatArray:
        return self._rigid_body_mass.copy()

    @property
    def added_mass_matrix(self) -> FloatArray:
        return self._added_mass.copy()

    @property
    def mass_matrix(self) -> FloatArray:
        """Constant total generalized inertia matrix ``M = M_RB + M_A``."""
        return self._mass.copy()

    def split_state(self, state: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return ``(position, quaternion, generalized_velocity)`` views."""
        state = self.validate_state(state)
        return (
            state[self.POSITION_SLICE],
            state[self.QUATERNION_SLICE],
            state[self.VELOCITY_SLICE],
        )

    def project_state(self, state: FloatArray) -> FloatArray:
        """Normalize the quaternion after numerical integration."""
        state = self.validate_state(state).copy()
        state[self.QUATERNION_SLICE] = normalize_quaternion(state[self.QUATERNION_SLICE])
        return state

    def relative_velocity(self, state: FloatArray) -> FloatArray:
        """Return body velocity relative to a constant inertial water current."""
        _, quaternion, velocity = self.split_state(state)
        rotation = rotation_matrix_from_quaternion(quaternion)
        current_body = rotation.T @ self._current_velocity_inertial

        relative = velocity.copy()
        relative[:3] -= current_body
        return relative

    def coriolis_matrix(self, generalized_velocity: FloatArray) -> FloatArray:
        """Return total rigid-body plus added-mass Coriolis matrix."""
        velocity = np.asarray(generalized_velocity, dtype=float).reshape(6)
        linear_velocity = velocity[:3]
        angular_velocity = velocity[3:]

        mass = self.parameters.mass
        inertia = np.diag(self.parameters.rigid_body_inertia_diagonal)

        c_rb = np.zeros((6, 6), dtype=float)
        c_rb[:3, 3:] = -mass * skew(linear_velocity)
        c_rb[3:, :3] = -mass * skew(linear_velocity)
        c_rb[3:, 3:] = -skew(inertia @ angular_velocity)

        added_translation = self.parameters.added_mass_diagonal[:3]
        added_rotation = self.parameters.added_mass_diagonal[3:]
        added_linear_momentum = added_translation * linear_velocity
        added_angular_momentum = added_rotation * angular_velocity

        c_a = np.zeros((6, 6), dtype=float)
        c_a[:3, 3:] = -skew(added_linear_momentum)
        c_a[3:, :3] = -skew(added_linear_momentum)
        c_a[3:, 3:] = -skew(added_angular_momentum)

        return c_rb + c_a

    def damping_matrix(self, relative_velocity: FloatArray) -> FloatArray:
        """Return diagonal linear-plus-quadratic hydrodynamic damping."""
        relative_velocity = np.asarray(relative_velocity, dtype=float).reshape(6)
        diagonal = (
            self.parameters.linear_damping_diagonal
            + self.parameters.quadratic_damping_diagonal * np.abs(relative_velocity)
        )
        return np.diag(diagonal)

    def restoring_wrench(self, quaternion: FloatArray) -> FloatArray:
        """Return the hydrostatic restoring vector in the body frame."""
        roll, pitch, _ = quaternion_to_roll_pitch_yaw(quaternion)
        p = self.parameters

        weight = p.mass * p.gravity
        buoyancy = p.water_density * p.gravity * p.volume

        xg, yg, zg = p.center_of_gravity
        xb, yb, zb = p.center_of_buoyancy

        sin_roll = np.sin(roll)
        cos_roll = np.cos(roll)
        sin_pitch = np.sin(pitch)
        cos_pitch = np.cos(pitch)

        restoring = np.zeros(6, dtype=float)
        restoring[0] = (weight - buoyancy) * sin_pitch
        restoring[1] = -(weight - buoyancy) * cos_pitch * sin_roll
        restoring[2] = -(weight - buoyancy) * cos_pitch * cos_roll
        restoring[3] = (
            -(yg * weight - yb * buoyancy) * cos_pitch * cos_roll
            + (zg * weight - zb * buoyancy) * cos_pitch * sin_roll
        )
        restoring[4] = (zg * weight - zb * buoyancy) * sin_pitch + (
            xg * weight - xb * buoyancy
        ) * cos_pitch * cos_roll
        restoring[5] = (
            -(xg * weight - xb * buoyancy) * cos_pitch * sin_roll
            - (yg * weight - yb * buoyancy) * sin_pitch
        )
        return restoring

    def drift_wrench(self, state: FloatArray) -> FloatArray:
        """Return ``C(nu)nu + D(nu_r)nu_r + g(q)``."""
        _, quaternion, velocity = self.split_state(state)
        relative_velocity = self.relative_velocity(state)
        return (
            self.coriolis_matrix(velocity) @ velocity
            + self.damping_matrix(relative_velocity) @ relative_velocity
            + self.restoring_wrench(quaternion)
        )

    def generalized_acceleration(self, state: FloatArray, wrench_body: FloatArray) -> FloatArray:
        """Return ``nu_dot`` for a body-frame generalized wrench."""
        wrench_body = self.validate_control(wrench_body)
        return np.linalg.solve(self._mass, wrench_body - self.drift_wrench(state))

    def dynamics(self, state: FloatArray, control: FloatArray) -> FloatArray:
        state = self.validate_state(state)
        wrench_body = self.validate_control(control)
        _, quaternion_raw, velocity = self.split_state(state)

        quaternion = normalize_quaternion(quaternion_raw)
        linear_velocity_body = velocity[:3]
        angular_velocity_body = velocity[3:]

        rotation = rotation_matrix_from_quaternion(quaternion)
        position_dot = rotation @ linear_velocity_body
        quaternion_dot = quaternion_derivative_body_rates(quaternion, angular_velocity_body)
        velocity_dot = self.generalized_acceleration(state, wrench_body)

        return np.concatenate((position_dot, quaternion_dot, velocity_dot))

    def translational_dynamics_terms(self, state: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Return ``(M_v, h_v)`` from ``M_v v_dot + h_v = tau_v``.

        The current BlueROV2 parameterization has block-diagonal inertia, so
        this partition is exact. Coupling through Coriolis, damping, and
        restoring forces remains contained in ``h_v``.
        """
        drift = self.drift_wrench(state)
        return self._mass[:3, :3].copy(), drift[:3].copy()

    def rotational_dynamics_terms(self, state: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Return ``(M_omega, h_omega)`` for the rotational subsystem."""
        drift = self.drift_wrench(state)
        return self._mass[3:, 3:].copy(), drift[3:].copy()

    def _build_rigid_body_mass_matrix(self) -> FloatArray:
        """Build the rigid-body inertia matrix for the source parameterization."""
        p = self.parameters
        if not np.allclose(p.center_of_gravity, 0.0):
            raise NotImplementedError(
                "The current BlueROV2 model assumes the body-frame origin is at the "
                "center of gravity."
            )

        return np.diag(
            [
                p.mass,
                p.mass,
                p.mass,
                p.inertia_x,
                p.inertia_y,
                p.inertia_z,
            ]
        )
