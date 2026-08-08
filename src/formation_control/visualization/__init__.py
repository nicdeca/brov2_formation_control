"""Visualization helpers for formation-control examples."""

from .barrier_functions import (
    DistanceBarrierSweep,
    DistanceBarrierTuning,
    FoVBarrierSweep,
    FoVBarrierTuning,
    distance_barrier_sweep,
    fov_barrier_sweep,
    plot_recentered_barrier_figure,
    plot_recentered_barrier_tuning,
)
from .bluerov2_heavy import BlueROV2HeavyVisualGeometry
from .domain_relaxation_plot import (
    FoVDomainRelaxationCurves,
    fov_domain_relaxation_curves,
    plot_fov_domain_relaxation,
)
from .edge_quality import edge_quality_color, edge_quality_colormap, validate_edge_quality
from .formation_2d import FormationAnimation2D, animate_formation_2d, plot_formation_2d
from .formation_3d import FormationAnimation3D, animate_formation_3d, plot_formation_3d
from .io import save_animation, save_figure
from .style import MATLAB_COLORS, MATLAB_GREEN, MATLAB_RED, MATLAB_YELLOW, apply_visualization_style
from .vehicle_geometry import RigidBodyWireframe, WireframePart
from .wrench_polytope import (
    WRENCH_LABELS,
    plot_wrench_axis_authority,
    plot_wrench_polytope_projections,
)

__all__ = [
    "FoVDomainRelaxationCurves",
    "BlueROV2HeavyVisualGeometry",
    "DistanceBarrierSweep",
    "DistanceBarrierTuning",
    "FoVBarrierSweep",
    "FoVBarrierTuning",
    "FormationAnimation2D",
    "FormationAnimation3D",
    "MATLAB_COLORS",
    "MATLAB_GREEN",
    "MATLAB_RED",
    "MATLAB_YELLOW",
    "RigidBodyWireframe",
    "WRENCH_LABELS",
    "WireframePart",
    "animate_formation_2d",
    "animate_formation_3d",
    "apply_visualization_style",
    "distance_barrier_sweep",
    "edge_quality_color",
    "edge_quality_colormap",
    "fov_barrier_sweep",
    "fov_domain_relaxation_curves",
    "plot_formation_2d",
    "plot_formation_3d",
    "plot_fov_domain_relaxation",
    "plot_recentered_barrier_figure",
    "plot_recentered_barrier_tuning",
    "plot_wrench_axis_authority",
    "plot_wrench_polytope_projections",
    "save_animation",
    "save_figure",
    "validate_edge_quality",
]
