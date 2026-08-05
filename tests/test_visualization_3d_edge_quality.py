import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from formation_control.graphs import DirectedSensingGraph  # noqa: E402
from formation_control.simulation import FormationTrajectory  # noqa: E402
from formation_control.visualization import (  # noqa: E402
    animate_formation_3d,
    plot_formation_3d,
)


def make_trajectory():
    return FormationTrajectory(
        times=np.array([0.0, 0.1, 0.2]),
        positions=np.array(
            [
                [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.2]],
                [[0.0, 0.0, 0.0], [-0.9, 0.1, 0.2]],
                [[0.0, 0.0, 0.0], [-0.8, 0.2, 0.2]],
            ]
        ),
    )


def test_static_plot_accepts_edge_quality():
    trajectory = make_trajectory()
    graph = DirectedSensingGraph.rooted_star(2)
    quality = np.array([[1.0], [0.5], [0.0]])

    figure, _ = plot_formation_3d(
        trajectory,
        graph,
        edge_quality=quality,
    )

    plt.close(figure)


def test_animation_accepts_time_varying_edge_quality():
    trajectory = make_trajectory()
    graph = DirectedSensingGraph.rooted_star(2)
    quality = np.array([[1.0], [0.5], [0.0]])

    result = animate_formation_3d(
        trajectory,
        graph,
        edge_quality=quality,
    )

    assert result.animation is not None
    result.animation._draw_was_started = True
    plt.close(result.figure)
