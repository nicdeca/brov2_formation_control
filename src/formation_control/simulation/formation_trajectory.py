"""Multi-agent trajectory containers used by examples and visualization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from formation_control.models.base import FloatArray


@dataclass(frozen=True, slots=True)
class FormationTrajectory:
    """Time-aligned multi-agent state and control histories.

    ``positions[k, i]`` is the inertial position of agent ``i`` at
    ``times[k]``.

    Optional ``velocities`` follow the same sample convention and are intended
    for translational velocity histories.  Optional scalar-first
    ``quaternions`` have shape ``(n_samples, n_agents, 4)``.

    ``controls[k, i]`` is the input applied over
    ``[times[k], times[k + 1]]`` and therefore contains one fewer time sample.
    Its last dimension may differ from the position dimension.
    """

    times: FloatArray
    positions: FloatArray
    velocities: FloatArray | None = None
    controls: FloatArray | None = None
    quaternions: FloatArray | None = None

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float)
        positions = np.asarray(self.positions, dtype=float)

        if times.ndim != 1:
            raise ValueError("times must be one-dimensional.")
        if positions.ndim != 3:
            raise ValueError("positions must have shape (n_samples, n_agents, dimension).")
        if positions.shape[0] != times.size:
            raise ValueError("positions must contain one sample per time.")
        if times.size > 1 and np.any(np.diff(times) <= 0.0):
            raise ValueError("times must be strictly increasing.")
        if not np.all(np.isfinite(times)) or not np.all(np.isfinite(positions)):
            raise ValueError("times and positions must be finite.")

        velocities = None
        if self.velocities is not None:
            velocities = np.asarray(self.velocities, dtype=float)
            if velocities.shape != positions.shape:
                raise ValueError("velocities must have the same shape as positions.")
            if not np.all(np.isfinite(velocities)):
                raise ValueError("velocities must be finite.")

        quaternions = None
        if self.quaternions is not None:
            quaternions = np.asarray(self.quaternions, dtype=float)
            expected = (times.size, positions.shape[1], 4)
            if quaternions.shape != expected:
                raise ValueError(
                    f"quaternions must have shape {expected}, got {quaternions.shape}."
                )
            if not np.all(np.isfinite(quaternions)):
                raise ValueError("quaternions must be finite.")

            norms = np.linalg.norm(quaternions, axis=2)
            if np.any(norms <= np.finfo(float).eps):
                raise ValueError("quaternions must have nonzero norm.")
            quaternions = quaternions / norms[..., None]

        controls = None
        if self.controls is not None:
            controls = np.asarray(self.controls, dtype=float)
            expected_prefix = (max(times.size - 1, 0), positions.shape[1])
            if controls.ndim != 3 or controls.shape[:2] != expected_prefix:
                raise ValueError(
                    "controls must have shape (n_samples - 1, n_agents, control_dimension)."
                )
            if not np.all(np.isfinite(controls)):
                raise ValueError("controls must be finite.")

        object.__setattr__(self, "times", times.copy())
        object.__setattr__(self, "positions", positions.copy())
        object.__setattr__(
            self,
            "velocities",
            None if velocities is None else velocities.copy(),
        )
        object.__setattr__(
            self,
            "controls",
            None if controls is None else controls.copy(),
        )
        object.__setattr__(
            self,
            "quaternions",
            None if quaternions is None else quaternions.copy(),
        )

    @property
    def n_samples(self) -> int:
        return self.times.size

    @property
    def n_agents(self) -> int:
        return self.positions.shape[1]

    @property
    def dimension(self) -> int:
        return self.positions.shape[2]

    @property
    def duration(self) -> float:
        if self.times.size < 2:
            return 0.0
        return float(self.times[-1] - self.times[0])

    def path(self, agent: int) -> FloatArray:
        """Return the position history of one agent."""
        if not 0 <= agent < self.n_agents:
            raise IndexError(f"agent index {agent} is outside [0, {self.n_agents}).")
        return self.positions[:, agent, :].copy()
