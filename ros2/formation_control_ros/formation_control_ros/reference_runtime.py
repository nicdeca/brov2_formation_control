"""Leader-reference handling independent of ROS message types."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from formation_control.control.bluerov2_leader import LeaderTrajectorySample


@dataclass
class VelocityCommandReference:
    """Piecewise-constant velocity command -> smooth p/v/a reference.

    For each axis

        p_dot = v,
        v_dot = lambda (v_cmd - v).

    The exact discrete-time solution is used over each control interval, so
    ``a = lambda (v_cmd - v)`` is available without numerical differentiation.
    """

    position: np.ndarray
    velocity: np.ndarray
    bandwidth: np.ndarray

    @classmethod
    def initialize(
        cls,
        position: np.ndarray,
        velocity: np.ndarray,
        bandwidth: float | np.ndarray,
    ) -> "VelocityCommandReference":
        position = np.asarray(position, dtype=float).reshape(3)
        velocity = np.asarray(velocity, dtype=float).reshape(3)
        bandwidth_array = np.asarray(bandwidth, dtype=float)
        if bandwidth_array.ndim == 0:
            bandwidth_array = np.full(3, float(bandwidth_array))
        bandwidth_array = bandwidth_array.reshape(3)
        if np.any(bandwidth_array <= 0.0):
            raise ValueError("velocity-command bandwidth must be positive.")
        return cls(
            position=position.copy(),
            velocity=velocity.copy(),
            bandwidth=bandwidth_array,
        )

    def sample(self, command: np.ndarray) -> LeaderTrajectorySample:
        command = np.asarray(command, dtype=float).reshape(3)
        return LeaderTrajectorySample(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            acceleration=self.bandwidth * (command - self.velocity),
        )

    def advance(self, command: np.ndarray, dt: float) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        command = np.asarray(command, dtype=float).reshape(3)
        delta = self.velocity - command
        decay = np.exp(-self.bandwidth * dt)

        self.position = (
            self.position
            + command * dt
            + delta * (1.0 - decay) / self.bandwidth
        )
        self.velocity = command + decay * delta

    def project_to_box(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> None:
        """Project the integrated reference into an axis-aligned box.

        Any velocity component still pointing outward at an active bound is
        reset to zero. This prevents manual-command reference windup against
        the workspace barrier.
        """
        lower = np.asarray(lower, dtype=float).reshape(3)
        upper = np.asarray(upper, dtype=float).reshape(3)
        if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
            raise ValueError("reference bounds must be finite.")
        if np.any(lower >= upper):
            raise ValueError("reference lower bounds must be below upper bounds.")

        projected = np.clip(self.position, lower, upper)
        for index in range(3):
            if projected[index] <= lower[index] and self.velocity[index] < 0.0:
                self.velocity[index] = 0.0
            elif projected[index] >= upper[index] and self.velocity[index] > 0.0:
                self.velocity[index] = 0.0
        self.position = projected

