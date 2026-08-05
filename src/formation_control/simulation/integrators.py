"""Fixed-step numerical integrators for continuous-time models."""

from __future__ import annotations

from typing import Protocol

from formation_control.models.base import ContinuousTimeModel, FloatArray


class Integrator(Protocol):
    """Protocol implemented by fixed-step numerical integrators."""

    def step(
        self,
        model: ContinuousTimeModel,
        state: FloatArray,
        control: FloatArray,
        dt: float,
    ) -> FloatArray:
        """Advance ``model`` by one time step."""


def _validate_step_size(dt: float) -> float:
    dt = float(dt)
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    return dt


class EulerIntegrator:
    """Explicit first-order Euler integrator."""

    def step(
        self,
        model: ContinuousTimeModel,
        state: FloatArray,
        control: FloatArray,
        dt: float,
    ) -> FloatArray:
        dt = _validate_step_size(dt)
        state = model.validate_state(state)
        control = model.validate_control(control)

        next_state = state + dt * model.dynamics(state, control)
        return model.project_state(next_state)


class RK4Integrator:
    """Classical explicit fourth-order Runge--Kutta integrator."""

    def step(
        self,
        model: ContinuousTimeModel,
        state: FloatArray,
        control: FloatArray,
        dt: float,
    ) -> FloatArray:
        dt = _validate_step_size(dt)
        state = model.validate_state(state)
        control = model.validate_control(control)

        k1 = model.dynamics(state, control)
        k2 = model.dynamics(state + 0.5 * dt * k1, control)
        k3 = model.dynamics(state + 0.5 * dt * k2, control)
        k4 = model.dynamics(state + dt * k3, control)

        next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return model.project_state(next_state)
