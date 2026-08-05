"""BlueROV2 Heavy eight-thruster geometry and force allocation.

The vehicle uses four horizontal T200 thrusters in the default 45-degree
vectored layout and four vertical T200 thrusters.  The physical generalized
body wrench is

    tau = B f,

where ``f in R^8`` contains signed axial thruster forces and each column of
``B`` is

    B_i = [e_i; r_i x e_i].

Here ``r_i`` is the thruster application point relative to the vehicle origin
and ``e_i`` is the unit direction of positive thrust, both in body
coordinates.

The mounting coordinates in :meth:`BlueROV2HeavyThrusterConfiguration.default_45deg`
are the calibrated geometry used by the project's previous BlueROV2 Heavy
model.  They are symmetric to sub-millimetre rounding and are now the single
source of truth for both control allocation and visualization.

T200 full-throttle limits are asymmetric.  Positive/negative force limits can
be selected at the published 12, 16, or 20 V operating points and optionally
derated without changing the allocation geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear

from formation_control.models.base import FloatArray

_KGF_TO_NEWTON = 9.80665


def _vector(value: FloatArray, dimension: int, *, name: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.shape != (dimension,):
        raise ValueError(f"{name} must have shape ({dimension},), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _matrix(value: FloatArray, shape: tuple[int, int], *, name: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


@dataclass(frozen=True)
class T200ForceLimits:
    """Per-thruster signed force limits in newtons.

    ``forward`` is the maximum force along the configured positive thrust
    direction.  ``reverse`` is the positive magnitude available opposite that
    direction.
    """

    forward: float
    reverse: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.forward) or self.forward <= 0.0:
            raise ValueError("forward thrust limit must be finite and positive.")
        if not np.isfinite(self.reverse) or self.reverse <= 0.0:
            raise ValueError("reverse thrust limit must be finite and positive.")

    @classmethod
    def from_voltage(
        cls,
        voltage: int = 16,
        *,
        derating: float = 1.0,
    ) -> T200ForceLimits:
        """Return published full-throttle T200 limits at 12, 16, or 20 V."""
        thrust_kgf = {
            12: (3.71, 2.92),
            16: (5.25, 4.10),
            20: (6.70, 5.05),
        }

        if voltage not in thrust_kgf:
            raise ValueError("voltage must be one of 12, 16, or 20 V.")
        if not np.isfinite(derating) or not 0.0 < derating <= 1.0:
            raise ValueError("derating must lie in (0, 1].")

        forward_kgf, reverse_kgf = thrust_kgf[voltage]
        return cls(
            forward=derating * forward_kgf * _KGF_TO_NEWTON,
            reverse=derating * reverse_kgf * _KGF_TO_NEWTON,
        )

    @property
    def lower(self) -> float:
        """Signed lower force bound."""
        return -self.reverse

    @property
    def upper(self) -> float:
        """Signed upper force bound."""
        return self.forward


@dataclass(frozen=True)
class BlueROV2HeavyThrusterConfiguration:
    """Physical positions, directions, and limits of the eight thrusters."""

    positions_body: FloatArray
    directions_body: FloatArray
    force_limits: T200ForceLimits
    names: tuple[str, ...] = (
        "T1",
        "T2",
        "T3",
        "T4",
        "T5",
        "T6",
        "T7",
        "T8",
    )

    def __post_init__(self) -> None:
        positions = _matrix(
            self.positions_body,
            (8, 3),
            name="positions_body",
        )
        directions = _matrix(
            self.directions_body,
            (8, 3),
            name="directions_body",
        )

        norms = np.linalg.norm(directions, axis=1)
        if np.any(norms <= np.finfo(float).eps):
            raise ValueError("thruster directions must be nonzero.")
        directions = directions / norms[:, None]

        if len(self.names) != 8:
            raise ValueError("exactly eight thruster names are required.")
        if len(set(self.names)) != 8:
            raise ValueError("thruster names must be unique.")

        object.__setattr__(self, "positions_body", positions.copy())
        object.__setattr__(self, "directions_body", directions.copy())

    @classmethod
    def default_45deg(
        cls,
        *,
        voltage: int = 16,
        derating: float = 1.0,
    ) -> BlueROV2HeavyThrusterConfiguration:
        """Return the project's BlueROV2 Heavy default 45-degree geometry.

        Thrusters 1--4 are the horizontal vectored thrusters and 5--8 are the
        four vertical Heavy-kit thrusters.
        """
        diagonal = 1.0 / np.sqrt(2.0)

        positions = np.array(
            [
                [0.156, 0.111, 0.085],
                [0.156, -0.111, 0.085],
                [-0.156, 0.111, 0.085],
                [-0.156, -0.111, 0.085],
                [0.120, 0.218, 0.000],
                [0.120, -0.218, 0.000],
                [-0.120, 0.218, 0.000],
                [-0.120, -0.218, 0.000],
            ],
            dtype=float,
        )

        directions = np.array(
            [
                [diagonal, -diagonal, 0.0],
                [diagonal, diagonal, 0.0],
                [-diagonal, -diagonal, 0.0],
                [-diagonal, diagonal, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=float,
        )

        return cls(
            positions_body=positions,
            directions_body=directions,
            force_limits=T200ForceLimits.from_voltage(
                voltage,
                derating=derating,
            ),
        )

    @property
    def n_thrusters(self) -> int:
        return 8

    @property
    def lower_bounds(self) -> FloatArray:
        return np.full(self.n_thrusters, self.force_limits.lower)

    @property
    def upper_bounds(self) -> FloatArray:
        return np.full(self.n_thrusters, self.force_limits.upper)


@dataclass(frozen=True)
class ThrusterAllocationResult:
    """Result of a bounded or unbounded wrench allocation."""

    forces: FloatArray
    achieved_wrench: FloatArray
    requested_wrench: FloatArray
    residual_norm: float
    bounded: bool

    def __post_init__(self) -> None:
        forces = _vector(self.forces, 8, name="forces")
        achieved = _vector(self.achieved_wrench, 6, name="achieved_wrench")
        requested = _vector(self.requested_wrench, 6, name="requested_wrench")

        if not np.isfinite(self.residual_norm) or self.residual_norm < 0.0:
            raise ValueError("residual_norm must be finite and nonnegative.")

        object.__setattr__(self, "forces", forces.copy())
        object.__setattr__(self, "achieved_wrench", achieved.copy())
        object.__setattr__(self, "requested_wrench", requested.copy())

    @property
    def exact(self) -> bool:
        return self.residual_norm <= 1e-8


@dataclass(frozen=True)
class BlueROV2HeavyThrusterAllocation:
    """Allocation map ``tau = B f`` for the BlueROV2 Heavy."""

    configuration: BlueROV2HeavyThrusterConfiguration

    @classmethod
    def default_45deg(
        cls,
        *,
        voltage: int = 16,
        derating: float = 1.0,
    ) -> BlueROV2HeavyThrusterAllocation:
        return cls(
            configuration=BlueROV2HeavyThrusterConfiguration.default_45deg(
                voltage=voltage,
                derating=derating,
            )
        )

    @property
    def matrix(self) -> FloatArray:
        """Return the 6x8 physical allocation matrix."""
        directions = self.configuration.directions_body
        positions = self.configuration.positions_body
        moments = np.cross(positions, directions)

        return np.vstack((directions.T, moments.T))

    @property
    def rank(self) -> int:
        return int(np.linalg.matrix_rank(self.matrix))

    @property
    def lower_bounds(self) -> FloatArray:
        return self.configuration.lower_bounds

    @property
    def upper_bounds(self) -> FloatArray:
        return self.configuration.upper_bounds

    def wrench(self, forces: FloatArray) -> FloatArray:
        """Map eight signed thruster forces to body wrench."""
        forces = _vector(forces, 8, name="forces")
        return self.matrix @ forces

    def contains(self, forces: FloatArray, *, tolerance: float = 1e-9) -> bool:
        """Return whether all thruster forces satisfy their physical bounds."""
        forces = _vector(forces, 8, name="forces")
        if tolerance < 0.0:
            raise ValueError("tolerance must be nonnegative.")

        return bool(
            np.all(forces >= self.lower_bounds - tolerance)
            and np.all(forces <= self.upper_bounds + tolerance)
        )

    def utilization(self, forces: FloatArray) -> FloatArray:
        """Return signed-force utilization magnitudes in ``[0, +inf)``.

        Positive forces are normalized by forward capability, negative forces
        by reverse capability.
        """
        forces = _vector(forces, 8, name="forces")
        denominators = np.where(
            forces >= 0.0,
            self.upper_bounds,
            -self.lower_bounds,
        )
        return np.abs(forces) / denominators

    def minimum_norm(
        self,
        wrench: FloatArray,
    ) -> ThrusterAllocationResult:
        """Unconstrained Moore--Penrose minimum-norm allocation."""
        requested = _vector(wrench, 6, name="wrench")
        forces = np.linalg.pinv(self.matrix) @ requested
        achieved = self.wrench(forces)

        return ThrusterAllocationResult(
            forces=forces,
            achieved_wrench=achieved,
            requested_wrench=requested,
            residual_norm=float(np.linalg.norm(achieved - requested)),
            bounded=False,
        )

    def bounded_least_squares(
        self,
        wrench: FloatArray,
        *,
        tolerance: float = 1e-10,
    ) -> ThrusterAllocationResult:
        """Allocate a wrench under asymmetric physical thruster limits.

        This helper is mainly useful for feedforward/equilibrium wrenches.  The
        follower CLF-QP optimizes directly in thruster coordinates and therefore
        imposes the same bounds without a separate allocation stage.
        """
        requested = _vector(wrench, 6, name="wrench")
        if tolerance <= 0.0:
            raise ValueError("tolerance must be positive.")

        result = lsq_linear(
            self.matrix,
            requested,
            bounds=(self.lower_bounds, self.upper_bounds),
            tol=tolerance,
            lsmr_tol=tolerance,
        )
        forces = np.asarray(result.x, dtype=float)
        achieved = self.wrench(forces)

        return ThrusterAllocationResult(
            forces=forces,
            achieved_wrench=achieved,
            requested_wrench=requested,
            residual_norm=float(np.linalg.norm(achieved - requested)),
            bounded=True,
        )

    def corner_wrenches(self) -> FloatArray:
        """Return all 2^8 wrench-polytope corner images.

        These points can be passed to ``scipy.spatial.ConvexHull`` when an
        H-representation or visualization of the achievable wrench polytope is
        needed later.
        """
        binary = ((np.arange(2**8)[:, None] >> np.arange(8)[None, :]) & 1).astype(bool)
        forces = np.where(
            binary,
            self.upper_bounds[None, :],
            self.lower_bounds[None, :],
        )
        return forces @ self.matrix.T
