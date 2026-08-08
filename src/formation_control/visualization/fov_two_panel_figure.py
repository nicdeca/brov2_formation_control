from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ---------------------------------------------------------------------------
# Output and style
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("outputs/fov_two_panel")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# MATLAB R2014b+ palette
MATLAB_BLUE = (0.0000, 0.4470, 0.7410)
MATLAB_ORANGE = (0.8500, 0.3250, 0.0980)
MATLAB_YELLOW = (0.9290, 0.6940, 0.1250)
MATLAB_PURPLE = (0.4940, 0.1840, 0.5560)
MATLAB_GREEN = (0.4660, 0.6740, 0.1880)
MATLAB_CYAN = (0.3010, 0.7450, 0.9330)
MATLAB_RED = (0.6350, 0.0780, 0.1840)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 14.5,
        "axes.labelsize": 14.5,
        "figure.dpi": 160,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.usetex": False,
    }
)

# ---------------------------------------------------------------------------
# Representative geometry
# ---------------------------------------------------------------------------

# Conservative normalized FoV limits.
alpha_h_c = 0.72
alpha_v_c = 0.62

# Representative normalized coordinates of robot j in camera i.
alpha_h = 0.48
alpha_v = 0.34

# Choose a camera-frame depth. The representative point is then consistent
# with alpha_h = p_y^C / p_x^C and alpha_v = p_z^C / p_x^C.
p_x = 1.75
p_target = np.array(
    [
        p_x,
        alpha_h * p_x,
        alpha_v * p_x,
    ]
)

# Representative projection plane used only for the 3-D illustration.
projection_depth = 1.55

physical_y = projection_depth
physical_z = projection_depth

conservative_y = projection_depth * alpha_h_c
conservative_z = projection_depth * alpha_v_c

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

figure = plt.figure(figsize=(10.8, 4.5))
grid = figure.add_gridspec(
    1,
    2,
    width_ratios=[1.02, 1.0],
    wspace=-0.06,
)

# ===========================================================================
# (a) Camera-frame geometry
# ===========================================================================

axes_3d = figure.add_subplot(grid[0, 0], projection="3d")

camera_origin = np.zeros(3)

# Camera origin.
axes_3d.scatter(
    [0.0],
    [0.0],
    [0.0],
    s=30,
    color="0.15",
    zorder=6,
)
axes_3d.text(
    -0.05,
    -0.06,
    -0.10,
    r"$C_i$",
    color="0.20",
)


def draw_frustum(
    y_half: float,
    z_half: float,
    *,
    color,
    linestyle: str,
    linewidth: float,
    alpha: float = 1.0,
) -> np.ndarray:
    """Draw a rectangular camera frustum and return its plane corners."""
    corners = np.array(
        [
            [projection_depth, -y_half, -z_half],
            [projection_depth, y_half, -z_half],
            [projection_depth, y_half, z_half],
            [projection_depth, -y_half, z_half],
        ]
    )

    for corner in corners:
        axes_3d.plot(
            [0.0, corner[0]],
            [0.0, corner[1]],
            [0.0, corner[2]],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
        )

    loop = np.vstack((corners, corners[0]))
    axes_3d.plot(
        loop[:, 0],
        loop[:, 1],
        loop[:, 2],
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
    )

    return corners


# Physical FoV.
physical_corners = draw_frustum(
    physical_y,
    physical_z,
    color="0.35",
    linestyle="-",
    linewidth=1.1,
    alpha=0.8,
)

# Very light image-plane patch to make the geometry easier to read.
axes_3d.add_collection3d(
    Poly3DCollection(
        [
            [
                physical_corners[0],
                physical_corners[1],
                physical_corners[2],
                physical_corners[3],
            ]
        ],
        facecolors=[(0.7, 0.7, 0.7, 0.05)],
        edgecolors="none",
    )
)

# Conservative FoV.
draw_frustum(
    conservative_y,
    conservative_z,
    color=MATLAB_GREEN,
    linestyle="--",
    linewidth=1.8,
)

# Relative camera-frame position.
axes_3d.plot(
    [0.0, p_target[0]],
    [0.0, p_target[1]],
    [0.0, p_target[2]],
    color=MATLAB_BLUE,
    linewidth=2.0,
)
axes_3d.scatter(
    [p_target[0]],
    [p_target[1]],
    [p_target[2]],
    s=42,
    color=MATLAB_BLUE,
    zorder=7,
)
axes_3d.text(
    p_target[0] + 0.05,
    p_target[1] + 0.03,
    p_target[2] + 0.03,
    r"$j$",
    color=MATLAB_BLUE,
)
axes_3d.text(
    0.82,
    0.22,
    0.25,
    r"$\mathbf{p}_{ij}^{C}$",
    color=MATLAB_BLUE,
)

# Projection onto the representative plane x_C = projection_depth.
projection_scale = projection_depth / p_target[0]
projected_point = np.array(
    [
        projection_depth,
        projection_scale * p_target[1],
        projection_scale * p_target[2],
    ]
)

axes_3d.scatter(
    [projected_point[0]],
    [projected_point[1]],
    [projected_point[2]],
    s=28,
    color=MATLAB_ORANGE,
    zorder=7,
)
axes_3d.plot(
    [p_target[0], projected_point[0]],
    [p_target[1], projected_point[1]],
    [p_target[2], projected_point[2]],
    linestyle=":",
    linewidth=1.0,
    color=MATLAB_ORANGE,
)

# ---------------------------------------------------------------------------
# Short colored camera-frame triad.
#
# All axes are intentionally much shorter than the FoV geometry so that they
# indicate the frame without visually competing with p_ij^C.
# ---------------------------------------------------------------------------

axis_length = 0.50

# x_C: optical axis
axes_3d.quiver(
    0.0,
    0.0,
    0.0,
    axis_length,
    0.0,
    0.0,
    color=MATLAB_BLUE,
    arrow_length_ratio=0.18,
    linewidth=1.5,
)
axes_3d.text(
    axis_length + 0.045,
    0.0,
    0.0,
    r"$x_C$",
    color=MATLAB_BLUE,
)

# y_C: horizontal image direction
axes_3d.quiver(
    0.0,
    0.0,
    0.0,
    0.0,
    axis_length,
    0.0,
    color=MATLAB_ORANGE,
    arrow_length_ratio=0.18,
    linewidth=1.5,
)
axes_3d.text(
    0.0,
    axis_length + 0.055,
    0.0,
    r"$y_C$",
    color=MATLAB_ORANGE,
)

# z_C: vertical image direction
axes_3d.quiver(
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    axis_length,
    color=MATLAB_PURPLE,
    arrow_length_ratio=0.18,
    linewidth=1.5,
)
axes_3d.text(
    0.0,
    0.0,
    axis_length + 0.055,
    r"$z_C$",
    color=MATLAB_PURPLE,
)

# Region labels.
axes_3d.text(
    1.18,
    -1.28,
    1.08,
    "physical FoV",
    color="0.30",
)
axes_3d.text(
    1.48,
    -0.95,
    -0.95,
    "conservative FoV",
    color=MATLAB_GREEN,
)

axes_3d.set_xlim(-0.05, 2.15)
axes_3d.set_ylim(-1.35, 1.35)
axes_3d.set_zlim(-1.25, 1.25)
axes_3d.set_box_aspect(
    (2.2, 2.5, 2.2),
    zoom=1.16,
)
axes_3d.view_init(elev=18, azim=-58)
axes_3d.set_axis_off()

axes_3d.text2D(
    0.02,
    0.96,
    "(a)",
    transform=axes_3d.transAxes,
    fontweight="bold",
)

# ===========================================================================
# (b) Normalized image plane
# ===========================================================================

axes = figure.add_subplot(grid[0, 1])

# Physical normalized FoV.
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

# Conservative normalized FoV.
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

# Representative normalized projection.
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
    s=50,
    color=MATLAB_BLUE,
    zorder=5,
)
axes.annotate(
    r"$(\alpha_h,\alpha_v)$",
    xy=(alpha_h, alpha_v),
    xytext=(0.60, 0.50),
    arrowprops={
        "arrowstyle": "->",
        "linewidth": 1.0,
        "color": MATLAB_BLUE,
    },
    color=MATLAB_BLUE,
)

# Horizontal normalized image coordinate.
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

# Vertical normalized image coordinate.
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

# Conservative limits.
axes.text(
    alpha_h_c,
    -alpha_v_c - 0.065,
    r"$\alpha_h^c$",
    ha="center",
    va="top",
    color=MATLAB_GREEN,
)
axes.text(
    -alpha_h_c,
    -alpha_v_c - 0.065,
    r"$-\alpha_h^c$",
    ha="center",
    va="top",
    color=MATLAB_GREEN,
)
axes.text(
    -alpha_h_c - 0.055,
    alpha_v_c,
    r"$\alpha_v^c$",
    ha="right",
    va="center",
    color=MATLAB_GREEN,
)
axes.text(
    -alpha_h_c - 0.055,
    -alpha_v_c,
    r"$-\alpha_v^c$",
    ha="right",
    va="center",
    color=MATLAB_GREEN,
)

# Physical normalized limits.
axes.text(1.0, 0.04, r"$1$", ha="center", va="bottom")
axes.text(-1.0, 0.04, r"$-1$", ha="center", va="bottom")
axes.text(0.04, 1.0, r"$1$", ha="left", va="center")
axes.text(0.04, -1.0, r"$-1$", ha="left", va="center")

# Optical center.
axes.scatter(
    [0.0],
    [0.0],
    s=18,
    color="0.15",
    zorder=5,
)
axes.text(
    0.04,
    0.04,
    "optical center",
    ha="left",
    va="bottom",
    color="0.25",
)

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

axes.set_xlabel(r"normalized horizontal coordinate $\alpha_h$")
axes.set_ylabel(r"normalized vertical coordinate $\alpha_v$")
axes.set_xlim(-1.15, 1.15)
axes.set_ylim(-1.15, 1.15)
axes.set_aspect("equal", adjustable="box")
axes.set_xticks([])
axes.set_yticks([])

for spine in axes.spines.values():
    spine.set_visible(False)

axes.text(
    0.02,
    0.96,
    "(b)",
    transform=axes.transAxes,
    ha="left",
    va="top",
    fontweight="bold",
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

figure.subplots_adjust(
    left=0.01,
    right=0.995,
    bottom=0.10,
    top=0.99,
    wspace=-0.06,
)

figure.savefig(
    OUTPUT_DIR / "camera_fov_coordinates_two_panel.pdf",
    bbox_inches="tight",
)
figure.savefig(
    OUTPUT_DIR / "camera_fov_coordinates_two_panel.png",
    dpi=600,
    bbox_inches="tight",
)
figure.savefig(
    OUTPUT_DIR / "camera_fov_coordinates_two_panel.svg",
    bbox_inches="tight",
)

plt.show()
