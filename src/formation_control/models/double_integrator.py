"""Double-integrator model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import ContinuousTimeModel, FloatArray


@dataclass(frozen=True, slots=True)
class DoubleIntegratorModel(ContinuousTimeModel):
    """Point-mass model with state ``x = col(p, v)`` and input acceleration."""

    dimension: int = 3

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("dimension must be positive.")

    @property
    def state_dim(self) -> int:
        return 2 * self.dimension

    @property
    def input_dim(self) -> int:
        return self.dimension

    def split_state(self, state: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Return position and velocity views of the state."""
        state = self.validate_state(state)
        return state[: self.dimension], state[self.dimension :]

    def dynamics(self, state: FloatArray, control: FloatArray) -> FloatArray:
        _, velocity = self.split_state(state)
        acceleration = self.validate_control(control)
        return np.concatenate((velocity, acceleration))
