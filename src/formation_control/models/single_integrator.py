"""Single-integrator model."""

from __future__ import annotations

from dataclasses import dataclass

from .base import ContinuousTimeModel, FloatArray


@dataclass(frozen=True, slots=True)
class SingleIntegratorModel(ContinuousTimeModel):
    """Kinematic point-mass model ``p_dot = u`` in ``dimension`` dimensions."""

    dimension: int = 3

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("dimension must be positive.")

    @property
    def state_dim(self) -> int:
        return self.dimension

    @property
    def input_dim(self) -> int:
        return self.dimension

    def dynamics(self, state: FloatArray, control: FloatArray) -> FloatArray:
        self.validate_state(state)
        control = self.validate_control(control)
        return control.copy()
