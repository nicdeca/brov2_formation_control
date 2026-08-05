import matplotlib as mpl

from formation_control.visualization import (
    MATLAB_COLORS,
    apply_visualization_style,
)


def test_matlab_palette_is_installed():
    apply_visualization_style()

    cycle = mpl.rcParams["axes.prop_cycle"].by_key()["color"]

    assert len(MATLAB_COLORS) == 7
    assert tuple(cycle[0]) == MATLAB_COLORS[0]


def test_paper_quality_increases_save_resolution():
    apply_visualization_style(paper_quality=False)
    normal_dpi = mpl.rcParams["savefig.dpi"]

    apply_visualization_style(paper_quality=True)
    paper_dpi = mpl.rcParams["savefig.dpi"]

    assert paper_dpi > normal_dpi
