import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from formation_control.visualization import save_figure  # noqa: E402


def test_save_figure_creates_parent_directory(tmp_path):
    figure, axes = plt.subplots()
    axes.plot([0.0, 1.0], [0.0, 1.0])

    path = save_figure(
        figure,
        tmp_path / "nested" / "figure.png",
        paper_quality=True,
    )

    assert path.exists()
    assert path.suffix == ".png"
    plt.close(figure)


def test_save_figure_default_suffix_depends_on_quality(tmp_path):
    figure, _ = plt.subplots()

    normal = save_figure(
        figure,
        tmp_path / "normal",
        paper_quality=False,
    )
    paper = save_figure(
        figure,
        tmp_path / "paper",
        paper_quality=True,
    )

    assert normal.suffix == ".png"
    assert paper.suffix == ".pdf"
    plt.close(figure)
