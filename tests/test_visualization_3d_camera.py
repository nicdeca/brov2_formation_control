import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from formation_control.geometry import PinholeCamera  # noqa: E402
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
                [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.3]],
                [[0.0, 0.0, 0.0], [-0.9, 0.1, 0.4]],
                [[0.0, 0.0, 0.0], [-0.8, 0.2, 0.5]],
            ]
        ),
        quaternions=np.tile(
            np.array([1.0, 0.0, 0.0, 0.0]),
            (3, 2, 1),
        ),
    )


def test_static_3d_plot_accepts_camera_frustum():
    trajectory = make_trajectory()
    graph = DirectedSensingGraph.rooted_star(2)
    camera = PinholeCamera.from_degrees(45.0, 30.0)

    figure, axes = plot_formation_3d(
        trajectory,
        graph,
        camera=camera,
        camera_agents=(1,),
    )

    assert figure is not None
    assert hasattr(axes, "get_zlim")
    plt.close(figure)


def test_3d_animation_accepts_camera_frustum_and_frame_stride():
    trajectory = make_trajectory()
    graph = DirectedSensingGraph.rooted_star(2)
    camera = PinholeCamera.from_degrees(45.0, 30.0)

    result = animate_formation_3d(
        trajectory,
        graph,
        camera=camera,
        camera_agents=(1,),
        frame_stride=2,
    )

    assert result.animation is not None
    result.animation._draw_was_started = True
    plt.close(result.figure)
