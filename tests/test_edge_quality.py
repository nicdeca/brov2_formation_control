import numpy as np
import pytest

from formation_control.visualization import (
    edge_quality_color,
    validate_edge_quality,
)


def test_edge_quality_validation():
    quality = np.array(
        [
            [1.0, 0.5],
            [0.25, 0.0],
        ]
    )

    actual = validate_edge_quality(
        quality,
        n_samples=2,
        n_edges=2,
    )

    np.testing.assert_allclose(actual, quality)


def test_edge_quality_rejects_out_of_range_values():
    with pytest.raises(ValueError):
        validate_edge_quality(
            np.array([[1.1]]),
            n_samples=1,
            n_edges=1,
        )


def test_robust_edge_is_greener_than_disconnected_edge():
    robust = edge_quality_color(1.0)
    disconnected = edge_quality_color(0.0)

    assert robust[1] > robust[0]
    assert disconnected[0] > disconnected[1]
