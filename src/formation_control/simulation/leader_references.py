"""Leader-reference models for long-horizon formation validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from formation_control.control.bluerov2_leader import LeaderTrajectorySample
from formation_control.models.base import ContinuousTimeModel, FloatArray


def _vector3(value: FloatArray, *, name: str) -> FloatArray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {vector.shape}.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")
    return vector


def _positive_vector(
    value: float | FloatArray,
    *,
    name: str,
) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(3, float(array))
    elif array.shape != (3,):
        raise ValueError(f"{name} must be scalar or have shape (3,).")
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must be finite and positive.")
    return array


@dataclass(frozen=True)
class VelocityCommandReferenceFilter(ContinuousTimeModel):
    """Convert an inertial velocity command into a smooth ``(p,v,a)`` reference.

    The state is ``[p_r, v_r]`` and

        p_r_dot = v_r,
        v_r_dot = Lambda_v (v_cmd - v_r).

    Hence ``v_r_dot`` is the acceleration supplied to the leader CLF-QP.
    No numerical differentiation of the velocity command is required.
    """

    bandwidth: float | FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "bandwidth",
            _positive_vector(self.bandwidth, name="bandwidth"),
        )

    @property
    def state_dim(self) -> int:
        return 6

    @property
    def input_dim(self) -> int:
        return 3

    def initialize(
        self,
        position: FloatArray,
        *,
        velocity: FloatArray | None = None,
    ) -> FloatArray:
        position = _vector3(position, name="position")
        if velocity is None:
            velocity = np.zeros(3)
        else:
            velocity = _vector3(velocity, name="velocity")
        return np.concatenate((position, velocity))

    def split_state(
        self,
        state: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        state = self.validate_state(state)
        return state[:3], state[3:]

    def dynamics(
        self,
        state: FloatArray,
        control: FloatArray,
    ) -> FloatArray:
        position, velocity = self.split_state(state)
        _ = position
        command = self.validate_control(control)
        acceleration = self.bandwidth * (command - velocity)
        return np.concatenate((velocity, acceleration))

    def evaluate(
        self,
        state: FloatArray,
        velocity_command: FloatArray,
    ) -> LeaderTrajectorySample:
        position, velocity = self.split_state(state)
        velocity_command = self.validate_control(velocity_command)
        acceleration = self.bandwidth * (velocity_command - velocity)
        return LeaderTrajectorySample(
            position=position.copy(),
            velocity=velocity.copy(),
            acceleration=acceleration,
        )


@dataclass(frozen=True, slots=True)
class SmoothSpatialTrajectoryReference:
    """Analytic full trajectory providing position, velocity, acceleration."""

    initial_position: FloatArray
    forward_speed: float = 0.30
    lateral_amplitude: float = 1.20
    vertical_amplitude: float = 0.35
    lateral_frequency: float = 0.045
    vertical_frequency: float = 0.030
    time_constant: float = 6.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "initial_position",
            _vector3(
                self.initial_position,
                name="initial_position",
            ).copy(),
        )

        positive = {
            "forward_speed": self.forward_speed,
            "lateral_amplitude": self.lateral_amplitude,
            "vertical_amplitude": self.vertical_amplitude,
            "lateral_frequency": self.lateral_frequency,
            "vertical_frequency": self.vertical_frequency,
            "time_constant": self.time_constant,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")

    def evaluate(self, time: float) -> LeaderTrajectorySample:
        time = float(time)
        if not np.isfinite(time) or time < 0.0:
            raise ValueError("time must be finite and nonnegative.")

        exponential = np.exp(-time / self.time_constant)
        progress = time - self.time_constant * (1.0 - exponential)
        progress_rate = 1.0 - exponential
        progress_acceleration = exponential / self.time_constant

        lateral_phase = self.lateral_frequency * progress
        vertical_phase = self.vertical_frequency * progress

        derivative_progress = np.array(
            [
                self.forward_speed,
                self.lateral_amplitude * self.lateral_frequency * np.cos(lateral_phase),
                self.vertical_amplitude * self.vertical_frequency * np.cos(vertical_phase),
            ],
            dtype=float,
        )
        second_derivative_progress = np.array(
            [
                0.0,
                -self.lateral_amplitude * self.lateral_frequency**2 * np.sin(lateral_phase),
                -self.vertical_amplitude * self.vertical_frequency**2 * np.sin(vertical_phase),
            ],
            dtype=float,
        )

        return LeaderTrajectorySample(
            position=self.initial_position
            + np.array(
                [
                    self.forward_speed * progress,
                    self.lateral_amplitude * np.sin(lateral_phase),
                    self.vertical_amplitude * np.sin(vertical_phase),
                ],
                dtype=float,
            ),
            velocity=derivative_progress * progress_rate,
            acceleration=(
                second_derivative_progress * progress_rate**2
                + derivative_progress * progress_acceleration
            ),
        )
