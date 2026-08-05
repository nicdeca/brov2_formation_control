"""Stateful fixed-step simulation of a continuous-time model."""

from __future__ import annotations

from dataclasses import dataclass, field

from formation_control.models.base import ContinuousTimeModel, FloatArray

from .integrators import Integrator, RK4Integrator, _validate_step_size


@dataclass(slots=True)
class Simulator:
    """Advance one continuous-time model with a fixed integration step.

    The simulator owns only the numerical state and simulation clock. Control
    laws, logging, disturbances, and multi-agent orchestration intentionally
    remain outside this class.
    """

    model: ContinuousTimeModel
    initial_state: FloatArray
    dt: float
    integrator: Integrator = field(default_factory=RK4Integrator)
    initial_time: float = 0.0
    _state: FloatArray = field(init=False, repr=False)
    _time: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.dt = _validate_step_size(self.dt)
        self.initial_time = float(self.initial_time)
        self.initial_state = self.model.project_state(self.initial_state)
        self._state = self.initial_state.copy()
        self._time = self.initial_time

    @property
    def state(self) -> FloatArray:
        """Current state, returned as a copy."""
        return self._state.copy()

    @property
    def time(self) -> float:
        """Current simulation time."""
        return self._time

    def reset(
        self,
        state: FloatArray | None = None,
        *,
        time: float | None = None,
    ) -> FloatArray:
        """Reset the simulator and return the new state.

        When omitted, ``state`` and ``time`` revert to their constructor
        values.
        """
        reset_state = self.initial_state if state is None else state
        self._state = self.model.project_state(reset_state)
        self._time = self.initial_time if time is None else float(time)
        return self.state

    def step(self, control: FloatArray) -> FloatArray:
        """Apply a constant control over one integration interval."""
        control = self.model.validate_control(control)
        self._state = self.integrator.step(
            self.model,
            self._state,
            control,
            self.dt,
        )
        self._time += self.dt
        return self.state
