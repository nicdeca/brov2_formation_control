import numpy as np

from formation_control.visualization import BlueROV2HeavyVisualGeometry


def test_bluerov2_heavy_wireframe_has_multiple_parts():
    geometry = BlueROV2HeavyVisualGeometry().wireframe()

    assert len(geometry.parts) >= 4
    assert geometry.bounding_radius > 0.25


def test_wireframe_transform_matches_rigid_body_transform():
    geometry = BlueROV2HeavyVisualGeometry().wireframe()
    position = np.array([1.0, -2.0, 0.5])
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    transformed = geometry.transform(position, rotation)

    first_body_point = geometry.parts[0].segments_body[0, 0]
    expected = position + rotation @ first_body_point

    np.testing.assert_allclose(
        transformed[0][0, 0],
        expected,
    )
