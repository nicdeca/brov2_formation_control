import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from formation_control.graphs import DirectedSensingGraph  # noqa: E402
from formation_control.simulation import FormationTrajectory  # noqa: E402
from formation_control.visualization import (  # noqa: E402
    BlueROV2HeavyVisualGeometry,
    animate_formation_3d,
    plot_formation_3d,
)


def make_trajectory():
    return FormationTrajectory(
        times=np.array([0.0, 0.1]),
        positions=np.array(
            [
                [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.2]],
                [[0.0, 0.0, 0.0], [-0.8, 0.1, 0.3]],
            ]
        ),
        quaternions=np.tile(
            np.array([1.0, 0.0, 0.0, 0.0]),
            (2, 2, 1),
        ),
    )


def test_static_plot_accepts_bluerov_geometry():
    trajectory = make_trajectory()
    graph = DirectedSensingGraph.rooted_star(2)
    geometry = BlueROV2HeavyVisualGeometry().wireframe()

    figure, _ = plot_formation_3d(
        trajectory,
        graph,
        vehicle_geometry=geometry,
        show_body_forward=False,
    )

    plt.close(figure)


def test_animation_accepts_bluerov_geometry():
    trajectory = make_trajectory()
    graph = DirectedSensingGraph.rooted_star(2)
    geometry = BlueROV2HeavyVisualGeometry().wireframe()

    result = animate_formation_3d(
        trajectory,
        graph,
        vehicle_geometry=geometry,
        show_body_forward=False,
    )

    assert result.animation is not None
    result.animation._draw_was_started = True
    plt.close(result.figure)
