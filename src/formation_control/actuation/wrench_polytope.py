"""Achievable wrench polytope induced by bounded thruster forces."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

from formation_control.models.base import FloatArray


def _vector(value: FloatArray, dimension: int, *, name: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.shape != (dimension,):
        raise ValueError(f"{name} must have shape ({dimension},), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


@dataclass(frozen=True)
class HalfSpaceRepresentation:
    matrix: FloatArray
    vector: FloatArray

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=float)
        vector = np.asarray(self.vector, dtype=float)
        if matrix.ndim != 2 or vector.shape != (matrix.shape[0],):
            raise ValueError("invalid half-space dimensions")
        object.__setattr__(self, "matrix", matrix.copy())
        object.__setattr__(self, "vector", vector.copy())

    @property
    def n_facets(self) -> int:
        return self.matrix.shape[0]


@dataclass(frozen=True)
class ChebyshevBall:
    center: FloatArray
    radius: float

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=float)
        if center.shape != (6,):
            raise ValueError("center must have shape (6,)")
        if not np.isfinite(self.radius) or self.radius < 0.0:
            raise ValueError("radius must be finite and nonnegative")
        object.__setattr__(self, "center", center.copy())


@dataclass(frozen=True)
class WrenchProjection2D:
    first_index: int
    second_index: int
    vertices: FloatArray

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices, dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 2 or vertices.shape[0] < 3:
            raise ValueError("vertices must have shape (n>=3,2)")
        object.__setattr__(self, "vertices", vertices.copy())


@dataclass(frozen=True)
class AchievableWrenchPolytope:
    allocation_matrix: FloatArray
    lower_thruster_bounds: FloatArray
    upper_thruster_bounds: FloatArray

    def __post_init__(self) -> None:
        matrix = np.asarray(self.allocation_matrix, dtype=float)
        lower = np.asarray(self.lower_thruster_bounds, dtype=float)
        upper = np.asarray(self.upper_thruster_bounds, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != 6:
            raise ValueError("allocation_matrix must have shape (6,m)")
        if lower.shape != (matrix.shape[1],) or upper.shape != lower.shape:
            raise ValueError("thruster bound dimension mismatch")
        if np.any(lower >= upper):
            raise ValueError("lower bounds must be strictly below upper bounds")
        if np.linalg.matrix_rank(matrix) < 6:
            raise ValueError("allocation matrix must have rank six")
        object.__setattr__(self, "allocation_matrix", matrix.copy())
        object.__setattr__(self, "lower_thruster_bounds", lower.copy())
        object.__setattr__(self, "upper_thruster_bounds", upper.copy())

    @property
    def n_thrusters(self) -> int:
        return self.allocation_matrix.shape[1]

    @cached_property
    def corner_forces(self) -> FloatArray:
        n = self.n_thrusters
        binary = ((np.arange(2**n)[:, None] >> np.arange(n)[None, :]) & 1).astype(bool)
        return np.where(
            binary, self.upper_thruster_bounds[None, :], self.lower_thruster_bounds[None, :]
        )

    @cached_property
    def vertices(self) -> FloatArray:
        return self.corner_forces @ self.allocation_matrix.T

    @cached_property
    def halfspaces(self) -> HalfSpaceRepresentation:
        hull = ConvexHull(self.vertices, qhull_options="QJ")
        equations = np.asarray(hull.equations, dtype=float)
        matrix = equations[:, :-1]
        vector = -equations[:, -1]
        norms = np.linalg.norm(matrix, axis=1)
        keep = norms > np.finfo(float).eps
        matrix = matrix[keep] / norms[keep, None]
        vector = vector[keep] / norms[keep]
        packed = np.hstack((matrix, vector[:, None]))
        _, idx = np.unique(np.round(packed, 9), axis=0, return_index=True)
        idx = np.sort(idx)
        return HalfSpaceRepresentation(matrix[idx], vector[idx])

    @cached_property
    def chebyshev_ball(self) -> ChebyshevBall:
        A = self.halfspaces.matrix
        b = self.halfspaces.vector
        augmented = np.column_stack((A, np.linalg.norm(A, axis=1)))
        objective = np.zeros(7)
        objective[-1] = -1.0
        result = linprog(
            objective,
            A_ub=augmented,
            b_ub=b,
            bounds=[(None, None)] * 6 + [(0.0, None)],
            method="highs",
        )
        if not result.success:
            raise RuntimeError(f"Chebyshev-ball optimization failed: {result.message}")
        return ChebyshevBall(result.x[:6], float(result.x[-1]))

    def wrench_center(self) -> FloatArray:
        midpoint = 0.5 * (self.lower_thruster_bounds + self.upper_thruster_bounds)
        return self.allocation_matrix @ midpoint

    def support(self, direction: FloatArray) -> float:
        direction = _vector(direction, 6, name="direction")
        coeff = self.allocation_matrix.T @ direction
        force = np.where(coeff >= 0.0, self.upper_thruster_bounds, self.lower_thruster_bounds)
        return float(direction @ (self.allocation_matrix @ force))

    def directional_interval(
        self, direction: FloatArray, *, normalize: bool = True
    ) -> tuple[float, float]:
        direction = _vector(direction, 6, name="direction")
        if normalize:
            norm = np.linalg.norm(direction)
            if norm <= np.finfo(float).eps:
                raise ValueError("direction must be nonzero")
            direction = direction / norm
        return -self.support(-direction), self.support(direction)

    def canonical_axis_intervals(self) -> FloatArray:
        intervals = np.empty((6, 2))
        for i in range(6):
            direction = np.zeros(6)
            direction[i] = 1.0
            intervals[i] = self.directional_interval(direction, normalize=False)
        return intervals

    def contains(self, wrench: FloatArray, *, tolerance: float = 1e-7) -> bool:
        wrench = _vector(wrench, 6, name="wrench")
        hs = self.halfspaces
        return bool(np.all(hs.matrix @ wrench <= hs.vector + tolerance))

    def projection(self, first_index: int, second_index: int) -> WrenchProjection2D:
        if not 0 <= first_index < 6 or not 0 <= second_index < 6 or first_index == second_index:
            raise ValueError("projection indices must be distinct and lie in [0,6)")
        points = self.vertices[:, [first_index, second_index]]
        hull = ConvexHull(points, qhull_options="QJ")
        return WrenchProjection2D(first_index, second_index, points[hull.vertices])


def wrench_polytope_from_allocation(allocation) -> AchievableWrenchPolytope:
    return AchievableWrenchPolytope(
        allocation_matrix=np.asarray(allocation.matrix, dtype=float),
        lower_thruster_bounds=np.asarray(allocation.lower_bounds, dtype=float),
        upper_thruster_bounds=np.asarray(allocation.upper_bounds, dtype=float),
    )
