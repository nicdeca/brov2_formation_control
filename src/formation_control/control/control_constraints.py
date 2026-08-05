"""Polyhedral constraints for optimized control inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from formation_control.models.base import FloatArray


def _bound_vector(
    value: float | FloatArray | None,
    dimension: int,
    *,
    name: str,
) -> FloatArray | None:
    if value is None:
        return None

    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(dimension, float(array))
    elif array.shape != (dimension,):
        raise ValueError(
            f"{name} must be a scalar or have shape ({dimension},), got {array.shape}."
        )

    if np.any(np.isnan(array)):
        raise ValueError(f"{name} must not contain NaN.")

    return array


@dataclass(frozen=True)
class PolyhedralControlSet:
    """Box and linear-inequality constraints on a control vector.

    The admissible set is

        lower <= u <= upper,
        G u <= h.

    Any subset of these constraints may be omitted.
    """

    dimension: int
    lower: FloatArray | None = None
    upper: FloatArray | None = None
    inequality_matrix: FloatArray | None = None
    inequality_bound: FloatArray | None = None

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("dimension must be positive.")

        lower = _bound_vector(
            self.lower,
            self.dimension,
            name="lower",
        )
        upper = _bound_vector(
            self.upper,
            self.dimension,
            name="upper",
        )

        if lower is not None and upper is not None and np.any(lower > upper):
            raise ValueError("lower bounds must not exceed upper bounds.")

        matrix = self.inequality_matrix
        bound = self.inequality_bound

        if (matrix is None) != (bound is None):
            raise ValueError("inequality_matrix and inequality_bound must be supplied together.")

        if matrix is not None:
            matrix = np.asarray(matrix, dtype=float)
            bound = np.asarray(bound, dtype=float)

            if matrix.ndim != 2 or matrix.shape[1] != self.dimension:
                raise ValueError("inequality_matrix must have one column per control input.")
            if bound.shape != (matrix.shape[0],):
                raise ValueError("inequality_bound must have one entry per inequality.")
            if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(bound)):
                raise ValueError("linear inequality data must contain only finite values.")

        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "inequality_matrix", matrix)
        object.__setattr__(self, "inequality_bound", bound)

    @classmethod
    def unconstrained(cls, dimension: int) -> PolyhedralControlSet:
        """Return ``R^dimension``."""
        return cls(dimension=dimension)

    @classmethod
    def box(
        cls,
        lower: float | FloatArray,
        upper: float | FloatArray,
        *,
        dimension: int | None = None,
    ) -> PolyhedralControlSet:
        """Construct a box-constrained control set.

        If scalar bounds are used, ``dimension`` must be supplied.  If vectors
        are used, the dimension is inferred and any supplied value is checked.
        """
        lower_array = np.asarray(lower, dtype=float)
        upper_array = np.asarray(upper, dtype=float)

        inferred_dimensions = {array.size for array in (lower_array, upper_array) if array.ndim > 0}

        if len(inferred_dimensions) > 1:
            raise ValueError("lower and upper vector bounds must have equal size.")

        if dimension is None:
            if not inferred_dimensions:
                raise ValueError("dimension is required when both bounds are scalars.")
            dimension = inferred_dimensions.pop()
        elif inferred_dimensions and dimension not in inferred_dimensions:
            raise ValueError("supplied dimension does not match vector bounds.")

        return cls(
            dimension=dimension,
            lower=lower,
            upper=upper,
        )

    @classmethod
    def polyhedron(
        cls,
        inequality_matrix: FloatArray,
        inequality_bound: FloatArray,
    ) -> PolyhedralControlSet:
        """Construct ``{u | G u <= h}``."""
        matrix = np.asarray(inequality_matrix, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("inequality_matrix must be two-dimensional.")

        return cls(
            dimension=matrix.shape[1],
            inequality_matrix=matrix,
            inequality_bound=inequality_bound,
        )

    def contains(self, control: FloatArray, *, tolerance: float = 1e-9) -> bool:
        """Return whether a control lies in the admissible set."""
        if tolerance < 0.0:
            raise ValueError("tolerance must be nonnegative.")

        control = np.asarray(control, dtype=float)
        if control.shape != (self.dimension,):
            raise ValueError(f"control must have shape ({self.dimension},), got {control.shape}.")

        if self.lower is not None and np.any(control < self.lower - tolerance):
            return False
        if self.upper is not None and np.any(control > self.upper + tolerance):
            return False
        if (
            self.inequality_matrix is not None
            and self.inequality_bound is not None
            and np.any(self.inequality_matrix @ control > self.inequality_bound + tolerance)
        ):
            return False

        return True
