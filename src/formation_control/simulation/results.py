"""Lightweight containers for fixed-step simulation data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from formation_control.models.base import FloatArray


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Time, state, and control histories from a fixed-step simulation.

    ``states[k]`` is the state at ``times[k]``. ``controls[k]`` is the input
    applied over the interval ``[times[k], times[k + 1]]``. Therefore, a
    result with ``N`` control intervals contains ``N + 1`` states.
    """

    times: FloatArray
    states: FloatArray
    controls: FloatArray

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float)
        states = np.asarray(self.states, dtype=float)
        controls = np.asarray(self.controls, dtype=float)

        if times.ndim != 1:
            raise ValueError("times must be a one-dimensional array.")
        if states.ndim != 2:
            raise ValueError("states must be a two-dimensional array.")
        if controls.ndim != 2:
            raise ValueError("controls must be a two-dimensional array.")
        if states.shape[0] != times.size:
            raise ValueError("states must contain one row for every time sample.")
        if controls.shape[0] != max(times.size - 1, 0):
            raise ValueError("controls must contain one row per simulation interval.")
        if times.size > 1 and np.any(np.diff(times) <= 0.0):
            raise ValueError("times must be strictly increasing.")

        object.__setattr__(self, "times", times.copy())
        object.__setattr__(self, "states", states.copy())
        object.__setattr__(self, "controls", controls.copy())

    @property
    def duration(self) -> float:
        """Elapsed simulated time."""
        if self.times.size < 2:
            return 0.0
        return float(self.times[-1] - self.times[0])

    @property
    def num_steps(self) -> int:
        """Number of simulated control intervals."""
        return self.controls.shape[0]
