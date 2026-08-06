from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

MATLAB_BLUE = (0.0000, 0.4470, 0.7410)
MATLAB_ORANGE = (0.8500, 0.3250, 0.0980)
MATLAB_PURPLE = (0.4940, 0.1840, 0.5560)
MATLAB_GREEN = (0.4660, 0.6740, 0.1880)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.labelsize": 11,
        "figure.dpi": 160,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# Representative values.
alpha_h_c = 0.72
alpha_v_c = 0.62
alpha_h = 0.48
alpha_v = 0.34

figure, axes = plt.subplots(figsize=(5.6, 4.7))

# Physical normalized field of view: |alpha_h| < 1, |alpha_v| < 1.
axes.add_patch(
    Rectangle(
        (-1.0, -1.0),
        2.0,
        2.0,
        fill=False,
        linewidth=1.8,
        edgecolor="0.20",
    )
)

# Conservative field of view:
# |alpha_h| < alpha_h^c, |alpha_v| < alpha_v^c.
axes.add_patch(
    Rectangle(
        (-alpha_h_c, -alpha_v_c),
        2.0 * alpha_h_c,
        2.0 * alpha_v_c,
        fill=False,
        linewidth=2.0,
        linestyle="--",
        edgecolor=MATLAB_GREEN,
    )
)

axes.axhline(0.0, color="0.70", linewidth=0.8)
axes.axvline(0.0, color="0.70", linewidth=0.8)

# Representative target projection.
axes.plot(
    [alpha_h, alpha_h],
    [0.0, alpha_v],
    linestyle=":",
    linewidth=1.2,
    color=MATLAB_BLUE,
)
axes.plot(
    [0.0, alpha_h],
    [alpha_v, alpha_v],
    linestyle=":",
    linewidth=1.2,
    color=MATLAB_BLUE,
)
axes.scatter(
    [alpha_h],
    [alpha_v],
    s=55,
    color=MATLAB_BLUE,
    zorder=5,
)
axes.annotate(
    r"$\boldsymbol{\alpha}_{ij}=(\alpha_h,\alpha_v)$",
    xy=(alpha_h, alpha_v),
    xytext=(0.56, 0.49),
    arrowprops={
        "arrowstyle": "->",
        "linewidth": 1.0,
        "color": MATLAB_BLUE,
    },
    color=MATLAB_BLUE,
)

# Horizontal normalized coordinate.
axes.add_patch(
    FancyArrowPatch(
        (0.0, -0.10),
        (alpha_h, -0.10),
        arrowstyle="<->",
        mutation_scale=10,
        linewidth=1.2,
        color=MATLAB_ORANGE,
    )
)
axes.text(
    alpha_h / 2.0,
    -0.16,
    r"$\alpha_h$",
    ha="center",
    va="top",
    color=MATLAB_ORANGE,
)

# Vertical normalized coordinate.
axes.add_patch(
    FancyArrowPatch(
        (-0.10, 0.0),
        (-0.10, alpha_v),
        arrowstyle="<->",
        mutation_scale=10,
        linewidth=1.2,
        color=MATLAB_PURPLE,
    )
)
axes.text(
    -0.16,
    alpha_v / 2.0,
    r"$\alpha_v$",
    ha="right",
    va="center",
    color=MATLAB_PURPLE,
)

# Conservative boundaries.
axes.text(
    alpha_h_c,
    -alpha_v_c - 0.07,
    r"$\alpha_h^c$",
    ha="center",
    va="top",
    color=MATLAB_GREEN,
)
axes.text(
    -alpha_h_c,
    -alpha_v_c - 0.07,
    r"$-\alpha_h^c$",
    ha="center",
    va="top",
    color=MATLAB_GREEN,
)
axes.text(
    -alpha_h_c - 0.06,
    alpha_v_c,
    r"$\alpha_v^c$",
    ha="right",
    va="center",
    color=MATLAB_GREEN,
)
axes.text(
    -alpha_h_c - 0.06,
    -alpha_v_c,
    r"$-\alpha_v^c$",
    ha="right",
    va="center",
    color=MATLAB_GREEN,
)

# Physical normalized boundaries.
axes.text(1.0, 0.04, r"$1$", ha="center", va="bottom")
axes.text(-1.0, 0.04, r"$-1$", ha="center", va="bottom")
axes.text(0.04, 1.0, r"$1$", ha="left", va="center")
axes.text(0.04, -1.0, r"$-1$", ha="left", va="center")

axes.text(
    0.0,
    0.83,
    "physical FoV",
    ha="center",
    va="center",
    color="0.25",
)
axes.text(
    0.0,
    -0.79,
    "conservative FoV",
    ha="center",
    va="center",
    color=MATLAB_GREEN,
)

axes.scatter([0.0], [0.0], s=18, color="0.15", zorder=5)
axes.text(
    0.04,
    0.04,
    "optical center",
    ha="left",
    va="bottom",
    color="0.25",
)

axes.set_xlabel(r"normalized horizontal image coordinate $\alpha_h$")
axes.set_ylabel(r"normalized vertical image coordinate $\alpha_v$")
axes.set_xlim(-1.15, 1.15)
axes.set_ylim(-1.15, 1.15)
axes.set_aspect("equal", adjustable="box")
axes.set_xticks([])
axes.set_yticks([])

for spine in axes.spines.values():
    spine.set_visible(False)

figure.tight_layout()

output_directory = Path("outputs/fov_coordinates")
output_directory.mkdir(parents=True, exist_ok=True)

figure.savefig(
    output_directory / "normalized_fov_coordinates.pdf",
    bbox_inches="tight",
)
figure.savefig(
    output_directory / "normalized_fov_coordinates.png",
    dpi=600,
    bbox_inches="tight",
)

plt.show()
