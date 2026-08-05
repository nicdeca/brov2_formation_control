"""Semantic red-to-green coloring for sensing-edge robustness."""

from __future__ import annotations

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .style import MATLAB_GREEN, MATLAB_RED, MATLAB_YELLOW


def validate_edge_quality(
    quality: np.ndarray | None,
    *,
    n_samples: int,
    n_edges: int,
) -> np.ndarray | None:
    """Validate a normalized edge-quality history.

    Quality must lie in ``[0, 1]``:

    * ``1``: robustly satisfied;
    * ``0``: at or beyond the physical disconnection boundary.
    """
    if quality is None:
        return None

    array = np.asarray(quality, dtype=float)
    expected = (n_samples, n_edges)
    if array.shape != expected:
        raise ValueError(f"edge_quality must have shape {expected}, got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError("edge_quality must contain only finite values.")
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError("edge_quality values must lie in [0, 1].")

    return array


def edge_quality_colormap() -> LinearSegmentedColormap:
    """Return a MATLAB-inspired red-yellow-green connection colormap."""
    return LinearSegmentedColormap.from_list(
        "connection_quality",
        (MATLAB_RED, MATLAB_YELLOW, MATLAB_GREEN),
    )


def edge_quality_color(quality: float) -> tuple[float, float, float, float]:
    """Map normalized connection quality to an RGBA edge color."""
    value = float(np.clip(quality, 0.0, 1.0))
    return edge_quality_colormap()(value)
