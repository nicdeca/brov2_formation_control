"""Consistent MATLAB-like plotting style for simulations and papers."""

from __future__ import annotations

import matplotlib as mpl
from cycler import cycler

# MATLAB R2014b+ default color order.
MATLAB_COLORS: tuple[tuple[float, float, float], ...] = (
    (0.0000, 0.4470, 0.7410),
    (0.8500, 0.3250, 0.0980),
    (0.9290, 0.6940, 0.1250),
    (0.4940, 0.1840, 0.5560),
    (0.4660, 0.6740, 0.1880),
    (0.3010, 0.7450, 0.9330),
    (0.6350, 0.0780, 0.1840),
)

MATLAB_GREEN = MATLAB_COLORS[4]
MATLAB_YELLOW = MATLAB_COLORS[2]
MATLAB_RED = MATLAB_COLORS[6]


def apply_visualization_style(*, paper_quality: bool = False) -> None:
    """Apply the project-wide plotting style.

    The style deliberately does not require an external LaTeX installation.
    STIX serif fonts and STIX mathtext provide a LaTeX-like appearance while
    remaining portable across development machines and CI.

    ``paper_quality`` increases rendering/save resolution and slightly
    strengthens lines and fonts for publication figures.
    """
    if paper_quality:
        font_size = 10.0
        line_width = 1.5
        figure_dpi = 160
        save_dpi = 600
    else:
        font_size = 10.0
        line_width = 1.25
        figure_dpi = 110
        save_dpi = 300

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "STIX Two Text",
                "STIXGeneral",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "font.size": font_size,
            "axes.titlesize": font_size + 1.0,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size - 1.0,
            "ytick.labelsize": font_size - 1.0,
            "legend.fontsize": font_size - 1.0,
            "axes.prop_cycle": cycler(color=MATLAB_COLORS),
            "axes.linewidth": 0.8,
            "lines.linewidth": line_width,
            "lines.markersize": 5.0,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.28,
            "figure.dpi": figure_dpi,
            "savefig.dpi": save_dpi,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.usetex": False,
        }
    )
