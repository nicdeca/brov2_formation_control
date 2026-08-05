"""Reusable rigid-body wireframe geometry for 3-D animations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from formation_control.models.base import FloatArray


def _segments(value: FloatArray, *, name: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 3 or array.shape[1:] != (2, 3):
        raise ValueError(f"{name} must have shape (n_segments, 2, 3), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


@dataclass(frozen=True)
class WireframePart:
    """One visual part of a rigid body, expressed in body coordinates."""

    segments_body: FloatArray
    linewidth: float = 1.0
    alpha: float = 1.0
    linestyle: str = "-"

    def __post_init__(self) -> None:
        segments = _segments(self.segments_body, name="segments_body")
        if self.linewidth <= 0.0:
            raise ValueError("linewidth must be positive.")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1].")

        object.__setattr__(self, "segments_body", segments.copy())


@dataclass(frozen=True)
class RigidBodyWireframe:
    """Collection of body-frame wireframe parts."""

    parts: tuple[WireframePart, ...]

    def __post_init__(self) -> None:
        if not self.parts:
            raise ValueError("a rigid-body wireframe requires at least one part.")

    @property
    def bounding_radius(self) -> float:
        """Maximum distance of any geometry point from the body origin."""
        return max(float(np.linalg.norm(part.segments_body, axis=2).max()) for part in self.parts)

    def transform(
        self,
        position: FloatArray,
        rotation_body_to_inertial: FloatArray,
    ) -> tuple[FloatArray, ...]:
        """Transform every wireframe part into inertial coordinates."""
        position = np.asarray(position, dtype=float)
        rotation = np.asarray(rotation_body_to_inertial, dtype=float)

        if position.shape != (3,):
            raise ValueError("position must have shape (3,).")
        if rotation.shape != (3, 3):
            raise ValueError("rotation_body_to_inertial must have shape (3, 3).")

        transformed = []
        for part in self.parts:
            segments = np.einsum(
                "ij,skj->ski",
                rotation,
                part.segments_body,
            )
            segments = segments + position[None, None, :]
            transformed.append(segments)

        return tuple(transformed)
