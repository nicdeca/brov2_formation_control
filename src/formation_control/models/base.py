"""Common interfaces for continuous-time dynamical models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class ContinuousTimeModel(ABC):
    """Base class for deterministic continuous-time control systems.

    A model exposes dynamics of the form ``x_dot = f(x, u)``.  State
    integration, process noise, and input saturation intentionally live
    outside this layer so that the same model can be reused by simulations,
    controllers, and ROS wrappers.
    """

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Dimension of the state vector."""

    @property
    @abstractmethod
    def input_dim(self) -> int:
        """Dimension of the control input."""

    @abstractmethod
    def dynamics(self, state: FloatArray, control: FloatArray) -> FloatArray:
        """Return the continuous-time state derivative."""

    def validate_state(self, state: FloatArray) -> FloatArray:
        """Return ``state`` as a 1-D float array after checking its size."""
        state = np.asarray(state, dtype=float)
        if state.shape != (self.state_dim,):
            raise ValueError(f"state must have shape ({self.state_dim},), got {state.shape}.")
        return state

    def validate_control(self, control: FloatArray) -> FloatArray:
        """Return ``control`` as a 1-D float array after checking its size."""
        control = np.asarray(control, dtype=float)
        if control.shape != (self.input_dim,):
            raise ValueError(f"control must have shape ({self.input_dim},), got {control.shape}.")
        return control

    def project_state(self, state: FloatArray) -> FloatArray:
        """Project a numerically integrated state onto the model state space.

        Euclidean-state models need no projection.  Manifold-valued models
        can override this method; e.g. the BlueROV2 model renormalizes its
        quaternion here.
        """
        return self.validate_state(state).copy()
