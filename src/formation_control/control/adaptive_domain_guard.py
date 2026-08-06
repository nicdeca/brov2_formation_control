"""Sampled-data safeguard for adaptive barrier domains.

For each adaptive scalar constraint

    h_a(z, rho) = h_c(z) + rho,
    0 <= rho <= rho_max,

the logarithmic barrier is defined only while ``h_a > 0``.

In continuous time a suitably designed adaptation law should prevent
``h_a`` from reaching zero.  In sampled-data simulation/experiments, however,
the physical state may move across the current adaptive boundary between two
control updates.  The controller must then enlarge the *computational*
admissible domain before evaluating the next logarithmic barrier.

The minimal monotone correction is

    rho_safe = max(rho, margin - h_c(z), 0),

clipped by ``rho_max``.  Because this correction can only increase ``rho`` and
the recentered adaptive barrier satisfies

    partial beta_bar / partial rho <= 0,

the jump can only decrease the barrier contribution.  If the required
enlargement exceeds ``rho_max``, the physical admissible domain itself has
been exhausted and the event is reported rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from formation_control.models.base import FloatArray

ADAPTIVE_DOMAIN_CHANNELS = (
    "collision",
    "range",
    "horizontal_fov",
    "vertical_fov",
)


class AdaptiveDomainLimitError(RuntimeError):
    """Raised when no admissible enlargement can contain the current state."""


@dataclass(frozen=True)
class AdaptiveDomainGuardResult:
    """Result of one sampled-data adaptive-domain projection."""

    enlargement: FloatArray
    minimum_required: FloatArray
    correction: FloatArray

    def __post_init__(self) -> None:
        enlargement = np.asarray(self.enlargement, dtype=float)
        required = np.asarray(self.minimum_required, dtype=float)
        correction = np.asarray(self.correction, dtype=float)

        if enlargement.ndim != 1:
            raise ValueError("enlargement must be one-dimensional.")
        if required.shape != enlargement.shape:
            raise ValueError("minimum_required must match enlargement shape.")
        if correction.shape != enlargement.shape:
            raise ValueError("correction must match enlargement shape.")

        object.__setattr__(self, "enlargement", enlargement.copy())
        object.__setattr__(self, "minimum_required", required.copy())
        object.__setattr__(self, "correction", correction.copy())

    @property
    def activated(self) -> bool:
        """Return whether any enlargement channel had to be corrected."""
        return bool(np.any(self.correction > 0.0))

    @property
    def maximum_correction(self) -> float:
        return float(np.max(self.correction, initial=0.0))


def project_adaptive_domain(
    enlargement: FloatArray,
    conservative_constraint_values: FloatArray,
    maximum_enlargement: FloatArray,
    *,
    enabled: FloatArray | None = None,
    margin: float = 1e-6,
    tolerance: float = 1e-10,
) -> AdaptiveDomainGuardResult:
    """Project ``rho`` just enough to keep all enabled adaptive barriers valid.

    Parameters
    ----------
    enlargement:
        Current adaptive state ``rho``.
    conservative_constraint_values:
        Current values ``h_c(z)`` of the conservative scalar constraints.
    maximum_enlargement:
        Physical reserve ``rho_max`` for each channel.
    enabled:
        Optional Boolean mask. Disabled channels are left unchanged.
    margin:
        Strict positive floor imposed on ``h_c + rho`` after projection.
    tolerance:
        Numerical tolerance used when deciding whether the physical reserve
        has been exhausted.
    """
    rho = np.asarray(enlargement, dtype=float)
    base = np.asarray(conservative_constraint_values, dtype=float)
    maximum = np.asarray(maximum_enlargement, dtype=float)

    if rho.ndim != 1:
        raise ValueError("enlargement must be one-dimensional.")
    if base.shape != rho.shape:
        raise ValueError("conservative_constraint_values must match enlargement shape.")
    if maximum.shape != rho.shape:
        raise ValueError("maximum_enlargement must match enlargement shape.")
    if not np.all(np.isfinite(rho)):
        raise ValueError("enlargement must be finite.")
    if not np.all(np.isfinite(base)):
        raise ValueError("conservative constraint values must be finite.")
    if not np.all(np.isfinite(maximum)) or np.any(maximum < 0.0):
        raise ValueError("maximum_enlargement must be finite and nonnegative.")
    if not np.isfinite(margin) or margin <= 0.0:
        raise ValueError("margin must be finite and strictly positive.")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and nonnegative.")

    if enabled is None:
        active = np.ones(rho.shape, dtype=bool)
    else:
        active = np.asarray(enabled, dtype=bool)
        if active.shape != rho.shape:
            raise ValueError("enabled must match enlargement shape.")

    bounded_rho = np.clip(rho, 0.0, maximum)
    minimum_required = bounded_rho.copy()

    geometric_requirement = np.maximum(margin - base, 0.0)
    minimum_required[active] = np.maximum(
        bounded_rho[active],
        geometric_requirement[active],
    )

    impossible = active & (minimum_required > maximum + tolerance)
    if np.any(impossible):
        indices = np.flatnonzero(impossible)
        descriptions = []
        for index in indices:
            name = (
                ADAPTIVE_DOMAIN_CHANNELS[index]
                if index < len(ADAPTIVE_DOMAIN_CHANNELS)
                else f"channel_{index}"
            )
            descriptions.append(
                f"{name}: h_c={base[index]:.6g}, "
                f"required rho={minimum_required[index]:.6g}, "
                f"rho_max={maximum[index]:.6g}"
            )
        raise AdaptiveDomainLimitError(
            "The current state cannot be placed strictly inside the adaptive "
            "domain without exceeding the physical admissible-domain reserve: "
            + "; ".join(descriptions)
        )

    projected = bounded_rho.copy()
    projected[active] = np.minimum(
        minimum_required[active],
        maximum[active],
    )
    correction = projected - bounded_rho

    return AdaptiveDomainGuardResult(
        enlargement=projected,
        minimum_required=minimum_required,
        correction=correction,
    )
