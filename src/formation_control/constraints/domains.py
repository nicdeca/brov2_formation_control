"""Conservative and physical admissible-domain parameters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _validate_enlargement(value: float, maximum: float, *, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= maximum:
        raise ValueError(f"{name} must lie in [0, {maximum}].")
    return value


@dataclass(frozen=True)
class DistanceDomain:
    """Physical and conservative distance bounds.

    The physical domain is

        d_min < ||p_ij|| < d_max,

    while the conservative domain is

        d_min_conservative < ||p_ij|| < d_max_conservative.

    The conservative domain must be a strict subset of the physical one.
    By default, additive relaxation is defined on the direct-distance
    constraints used in the paper.  Set ``squared=True`` to retain the legacy
    squared-distance parametrization.
    """

    d_min: float
    d_max: float
    d_min_conservative: float
    d_max_conservative: float
    squared: bool = False

    def __post_init__(self) -> None:
        if self.d_min < 0.0:
            raise ValueError("d_min must be nonnegative.")
        if not self.d_min < self.d_max:
            raise ValueError("Require d_min < d_max.")
        if not self.d_min < self.d_min_conservative:
            raise ValueError("Require d_min < d_min_conservative.")
        if not self.d_max_conservative < self.d_max:
            raise ValueError("Require d_max_conservative < d_max.")
        if not self.d_min_conservative < self.d_max_conservative:
            raise ValueError("Conservative distance interval must be nonempty.")

    @property
    def collision_enlargement_max(self) -> float:
        """Maximum additive enlargement of the collision constraint."""
        if self.squared:
            return self.d_min_conservative**2 - self.d_min**2
        return self.d_min_conservative - self.d_min

    @property
    def range_enlargement_max(self) -> float:
        """Maximum additive enlargement of the range constraint."""
        if self.squared:
            return self.d_max**2 - self.d_max_conservative**2
        return self.d_max - self.d_max_conservative

    def effective_minimum_distance(self, enlargement: float) -> float:
        """Return the minimum distance associated with ``h_delta^c + rho``."""
        enlargement = _validate_enlargement(
            enlargement,
            self.collision_enlargement_max,
            name="collision enlargement",
        )
        if self.squared:
            return float(np.sqrt(self.d_min_conservative**2 - enlargement))
        return float(self.d_min_conservative - enlargement)

    def effective_maximum_distance(self, enlargement: float) -> float:
        """Return the maximum distance associated with ``h_Delta^c + rho``."""
        enlargement = _validate_enlargement(
            enlargement,
            self.range_enlargement_max,
            name="range enlargement",
        )
        if self.squared:
            return float(np.sqrt(self.d_max_conservative**2 + enlargement))
        return float(self.d_max_conservative + enlargement)


@dataclass(frozen=True)
class FieldOfViewDomain:
    """Physical and conservative normalized image-plane limits.

    Normalized coordinates are defined relative to the physical camera FoV, so
    the physical domain is ``|alpha_h| < 1`` and ``|alpha_v| < 1``.
    Conservative limits must lie strictly inside that domain.
    """

    alpha_h_conservative: float
    alpha_v_conservative: float

    def __post_init__(self) -> None:
        for name, value in (
            ("alpha_h_conservative", self.alpha_h_conservative),
            ("alpha_v_conservative", self.alpha_v_conservative),
        ):
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie strictly between 0 and 1.")

    @property
    def horizontal_enlargement_max(self) -> float:
        return 1.0 - self.alpha_h_conservative**2

    @property
    def vertical_enlargement_max(self) -> float:
        return 1.0 - self.alpha_v_conservative**2

    def effective_horizontal_limit(self, enlargement: float) -> float:
        """Return the enlarged normalized horizontal FoV limit."""
        enlargement = _validate_enlargement(
            enlargement,
            self.horizontal_enlargement_max,
            name="horizontal FoV enlargement",
        )
        return float(np.sqrt(self.alpha_h_conservative**2 + enlargement))

    def effective_vertical_limit(self, enlargement: float) -> float:
        """Return the enlarged normalized vertical FoV limit."""
        enlargement = _validate_enlargement(
            enlargement,
            self.vertical_enlargement_max,
            name="vertical FoV enlargement",
        )
        return float(np.sqrt(self.alpha_v_conservative**2 + enlargement))
