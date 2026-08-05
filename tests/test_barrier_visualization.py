import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from formation_control.potentials import RecenteredLogBarrier  # noqa: E402
from formation_control.visualization import (  # noqa: E402
    DistanceBarrierTuning,
    FoVBarrierTuning,
    distance_barrier_sweep,
    fov_barrier_sweep,
    plot_recentered_barrier_figure,
    plot_recentered_barrier_tuning,
)


def test_distance_barrier_has_zero_minimum_at_desired_distance():
    tuning = DistanceBarrierTuning(
        minimum_distance=0.8,
        maximum_distance=3.0,
        desired_distance=1.8,
    )
    sweep = distance_barrier_sweep(tuning, n_samples=4001)

    index = int(np.argmin(np.abs(sweep.distance - tuning.desired_distance)))

    assert sweep.total[index] < 1e-5
    assert abs(sweep.derivative_total[index]) < 2e-2


def test_fov_barrier_is_zero_and_flat_at_image_center():
    tuning = FoVBarrierTuning(
        horizontal_limit=0.72,
        vertical_limit=0.72,
    )
    sweep = fov_barrier_sweep(tuning, n_samples=4001)

    index = int(np.argmin(np.abs(sweep.coordinate)))

    assert sweep.horizontal[index] < 1e-10
    assert abs(sweep.derivative_horizontal[index]) < 1e-10


def test_distance_sweep_matches_scalar_recentered_barrier():
    tuning = DistanceBarrierTuning(
        minimum_distance=1.0,
        maximum_distance=4.0,
        desired_distance=2.0,
        collision_weight=0.3,
        range_weight=0.7,
    )
    sweep = distance_barrier_sweep(tuning, n_samples=101)

    index = 37
    distance = sweep.distance[index]
    h = distance**2 - tuning.minimum_distance**2
    h_d = tuning.desired_distance**2 - tuning.minimum_distance**2

    expected = tuning.collision_weight * RecenteredLogBarrier(h_d).value(h)

    assert sweep.collision[index] == pytest.approx(expected)


def test_paper_figure_constructs():
    figure, axes = plot_recentered_barrier_figure(
        paper_quality=True,
    )

    assert len(axes) == 3
    plt.close(figure)


def test_tuning_figure_constructs():
    figure, axes = plot_recentered_barrier_tuning()

    assert len(axes) == 2
    plt.close(figure)
