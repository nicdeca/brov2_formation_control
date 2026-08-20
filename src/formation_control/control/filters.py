"""Continuous-time command filters.

The filter objects contain only immutable parameters; their dynamic state is
kept externally.  They implement the same ``ContinuousTimeModel`` interface as
the plant models, so the generic simulation integrators can advance filter
states without controller-specific integration code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from formation_control.models.base import ContinuousTimeModel, FloatArray


def _positive_vector(
    value: float | FloatArray,
    dimension: int,
    *,
    name: str,
) -> FloatArray:
    array = np.asarray(value, dtype=float)

    if array.ndim == 0:
        array = np.full(dimension, float(array))
    elif array.shape != (dimension,):
        raise ValueError(
            f"{name} must be a scalar or have shape ({dimension},), got {array.shape}."
        )

    if np.any(array <= 0.0):
        raise ValueError(f"{name} must be strictly positive.")

    return array


@dataclass(frozen=True)
class CommandFilterEvaluation:
    """Filtered command and its first derivative."""

    output: FloatArray
    output_derivative: FloatArray

    def __post_init__(self) -> None:
        output = np.asarray(self.output, dtype=float)
        derivative = np.asarray(self.output_derivative, dtype=float)

        if output.ndim != 1:
            raise ValueError("output must be one-dimensional.")
        if derivative.shape != output.shape:
            raise ValueError("output_derivative must have the same shape as output.")

        object.__setattr__(self, "output", output)
        object.__setattr__(self, "output_derivative", derivative)


@dataclass(frozen=True)
class FirstOrderCommandFilter(ContinuousTimeModel):
    """Diagonal first-order command filter.

    For desired command ``r`` and filtered command ``r_c``,

        r_c_dot = Lambda (r - r_c).

    ``bandwidth`` may be a positive scalar or one positive value per signal
    component.
    """

    signal_dim: int
    bandwidth: float | FloatArray

    def __post_init__(self) -> None:
        if self.signal_dim <= 0:
            raise ValueError("signal_dim must be positive.")
        bandwidth = _positive_vector(
            self.bandwidth,
            self.signal_dim,
            name="bandwidth",
        )
        object.__setattr__(self, "bandwidth", bandwidth)

    @property
    def state_dim(self) -> int:
        return self.signal_dim

    @property
    def input_dim(self) -> int:
        return self.signal_dim

    def initialize(self, command: FloatArray) -> FloatArray:
        """Initialize the filtered output at the supplied command."""
        return self.validate_control(command).copy()

    def dynamics(self, state: FloatArray, control: FloatArray) -> FloatArray:
        state = self.validate_state(state)
        command = self.validate_control(control)
        return self.bandwidth * (command - state)

    def evaluate(
        self,
        state: FloatArray,
        command: FloatArray,
    ) -> CommandFilterEvaluation:
        """Return the filtered command and its derivative."""
        state = self.validate_state(state)
        derivative = self.dynamics(state, command)
        return CommandFilterEvaluation(
            output=state.copy(),
            output_derivative=derivative,
        )


@dataclass(frozen=True)
class SecondOrderCommandFilter(ContinuousTimeModel):
    """Diagonal second-order command filter.

    For desired command ``r`` and filtered command ``r_c``,

        r_c_ddot
        + 2 zeta omega_n r_c_dot
        + omega_n^2 (r_c - r) = 0.

    The filter state is ``[r_c, r_c_dot]``.
    """

    signal_dim: int
    natural_frequency: float | FloatArray
    damping_ratio: float | FloatArray = 1.0

    def __post_init__(self) -> None:
        if self.signal_dim <= 0:
            raise ValueError("signal_dim must be positive.")

        natural_frequency = _positive_vector(
            self.natural_frequency,
            self.signal_dim,
            name="natural_frequency",
        )
        damping_ratio = _positive_vector(
            self.damping_ratio,
            self.signal_dim,
            name="damping_ratio",
        )

        object.__setattr__(self, "natural_frequency", natural_frequency)
        object.__setattr__(self, "damping_ratio", damping_ratio)

    @property
    def state_dim(self) -> int:
        return 2 * self.signal_dim

    @property
    def input_dim(self) -> int:
        return self.signal_dim

    def initialize(self, command: FloatArray) -> FloatArray:
        """Initialize at the command with zero filtered-command derivative."""
        command = self.validate_control(command)
        return np.concatenate((command, np.zeros(self.signal_dim)))

    def split_state(self, state: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Return ``(filtered_command, filtered_command_derivative)`` views."""
        state = self.validate_state(state)
        return state[: self.signal_dim], state[self.signal_dim :]

    def dynamics(self, state: FloatArray, control: FloatArray) -> FloatArray:
        command = self.validate_control(control)
        output, output_derivative = self.split_state(state)

        acceleration = (
            self.natural_frequency**2 * (command - output)
            - 2.0 * self.damping_ratio * self.natural_frequency * output_derivative
        )

        return np.concatenate((output_derivative, acceleration))

    def evaluate(
        self,
        state: FloatArray,
        command: FloatArray,
    ) -> CommandFilterEvaluation:
        """Return the filtered command and its first derivative."""
        self.validate_control(command)
        output, output_derivative = self.split_state(state)

        return CommandFilterEvaluation(
            output=output.copy(),
            output_derivative=output_derivative.copy(),
        )


CommandFilter = FirstOrderCommandFilter | SecondOrderCommandFilter
