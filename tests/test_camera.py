import numpy as np
import pytest

from formation_control.geometry import CameraExtrinsics, PinholeCamera, skew


def exp_so3(rotation_vector):
    rotation_vector = np.asarray(rotation_vector, dtype=float)
    angle = np.linalg.norm(rotation_vector)

    if angle < 1e-14:
        return np.eye(3) + skew(rotation_vector)

    axis = rotation_vector / angle
    axis_skew = skew(axis)
    return np.eye(3) + np.sin(angle) * axis_skew + (1.0 - np.cos(angle)) * (axis_skew @ axis_skew)


def finite_difference_matrix(function, value, output_dim, epsilon=1e-7):
    value = np.asarray(value, dtype=float)
    jacobian = np.zeros((output_dim, value.size))

    for index in range(value.size):
        perturbation = np.zeros_like(value)
        perturbation[index] = epsilon
        jacobian[:, index] = (function(value + perturbation) - function(value - perturbation)) / (
            2.0 * epsilon
        )

    return jacobian


def test_identity_camera_geometry():
    camera = PinholeCamera.from_degrees(45.0, 45.0)

    observation = camera.observe(
        observer_position=np.zeros(3),
        observer_rotation=np.eye(3),
        target_position=np.array([2.0, 1.0, -0.5]),
    )

    np.testing.assert_allclose(
        observation.relative_position_inertial,
        np.array([2.0, 1.0, -0.5]),
    )
    np.testing.assert_allclose(
        observation.relative_position_body,
        np.array([2.0, 1.0, -0.5]),
    )
    np.testing.assert_allclose(
        observation.point_camera,
        np.array([2.0, 1.0, -0.5]),
    )
    assert observation.image_point.alpha_h == pytest.approx(0.5)
    assert observation.image_point.alpha_v == pytest.approx(-0.25)


def test_camera_translation_is_expressed_in_body_frame():
    extrinsics = CameraExtrinsics(
        rotation_camera_from_body=np.eye(3),
        position_camera_in_body=np.array([0.2, 0.1, -0.1]),
    )
    camera = PinholeCamera.from_degrees(
        45.0,
        45.0,
        extrinsics=extrinsics,
    )

    observation = camera.observe(
        observer_position=np.zeros(3),
        observer_rotation=np.eye(3),
        target_position=np.array([2.0, 0.5, 0.3]),
    )

    np.testing.assert_allclose(
        observation.point_camera,
        np.array([1.8, 0.4, 0.4]),
    )


def test_image_projection_requires_positive_depth():
    camera = PinholeCamera.from_degrees(45.0, 45.0)

    with pytest.raises(ValueError):
        camera.normalized_image_point(np.array([0.0, 1.0, 0.0]))

    with pytest.raises(ValueError):
        camera.normalized_image_point(np.array([-1.0, 0.0, 0.0]))


def test_image_jacobian_matches_finite_difference():
    camera = PinholeCamera.from_degrees(50.0, 35.0)
    point_camera = np.array([2.3, 0.4, -0.7])

    analytic = camera.image_jacobian(point_camera)
    numeric = finite_difference_matrix(
        lambda point: camera.normalized_image_point(point).as_array(),
        point_camera,
        output_dim=2,
    )

    np.testing.assert_allclose(analytic, numeric, atol=1e-8)


def test_position_jacobians_match_finite_difference():
    camera = PinholeCamera.from_degrees(
        50.0,
        35.0,
        extrinsics=CameraExtrinsics(
            rotation_camera_from_body=exp_so3(np.array([0.1, -0.2, 0.15])),
            position_camera_in_body=np.array([0.2, -0.1, 0.05]),
        ),
    )
    observer_position = np.array([0.3, -0.4, 0.2])
    observer_rotation = exp_so3(np.array([-0.2, 0.1, 0.25]))
    target_position = np.array([3.0, 0.7, -0.1])

    analytic = camera.jacobians(
        observer_position,
        observer_rotation,
        target_position,
    )

    numeric_observer = finite_difference_matrix(
        lambda position: camera.observe(
            position,
            observer_rotation,
            target_position,
        ).image_point.as_array(),
        observer_position,
        output_dim=2,
    )
    numeric_target = finite_difference_matrix(
        lambda position: camera.observe(
            observer_position,
            observer_rotation,
            position,
        ).image_point.as_array(),
        target_position,
        output_dim=2,
    )

    np.testing.assert_allclose(
        analytic.image_wrt_observer_position,
        numeric_observer,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        analytic.image_wrt_target_position,
        numeric_target,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        analytic.image_wrt_target_position,
        -analytic.image_wrt_observer_position,
        atol=1e-12,
    )


def test_body_orientation_jacobian_matches_finite_difference():
    camera = PinholeCamera.from_degrees(
        55.0,
        40.0,
        extrinsics=CameraExtrinsics(
            rotation_camera_from_body=exp_so3(np.array([0.05, 0.08, -0.03])),
            position_camera_in_body=np.array([0.15, 0.02, -0.04]),
        ),
    )
    observer_position = np.array([0.1, -0.2, 0.3])
    observer_rotation = exp_so3(np.array([-0.15, 0.2, 0.1]))
    target_position = np.array([3.2, 0.4, 0.1])

    analytic = camera.jacobians(
        observer_position,
        observer_rotation,
        target_position,
    ).image_wrt_observer_orientation_body

    numeric = finite_difference_matrix(
        lambda delta_theta: camera.observe(
            observer_position,
            observer_rotation @ exp_so3(delta_theta),
            target_position,
        ).image_point.as_array(),
        np.zeros(3),
        output_dim=2,
    )

    np.testing.assert_allclose(analytic, numeric, atol=1e-7)


def test_image_gradient_pullback_matches_directional_derivative():
    camera = PinholeCamera.from_degrees(50.0, 40.0)
    observer_position = np.array([0.2, -0.1, 0.0])
    observer_rotation = exp_so3(np.array([0.0, 0.1, -0.05]))
    target_position = np.array([3.0, 0.6, -0.2])
    gradient_image = np.array([1.4, -0.7])

    jacobians = camera.jacobians(
        observer_position,
        observer_rotation,
        target_position,
    )
    gradient = jacobians.pullback_image_gradient(gradient_image)

    direction_position = np.array([0.3, -0.5, 0.2])
    direction_rotation = np.array([-0.2, 0.1, 0.4])
    epsilon = 1e-7

    def scalar_value(position, rotation):
        image = camera.observe(
            position,
            rotation,
            target_position,
        ).image_point.as_array()
        return float(gradient_image @ image)

    numeric_position = (
        scalar_value(
            observer_position + epsilon * direction_position,
            observer_rotation,
        )
        - scalar_value(
            observer_position - epsilon * direction_position,
            observer_rotation,
        )
    ) / (2.0 * epsilon)
    analytic_position = float(gradient.observer_position @ direction_position)

    numeric_rotation = (
        scalar_value(
            observer_position,
            observer_rotation @ exp_so3(epsilon * direction_rotation),
        )
        - scalar_value(
            observer_position,
            observer_rotation @ exp_so3(-epsilon * direction_rotation),
        )
    ) / (2.0 * epsilon)
    analytic_rotation = float(gradient.observer_orientation_body @ direction_rotation)

    assert analytic_position == pytest.approx(numeric_position, abs=1e-7)
    assert analytic_rotation == pytest.approx(numeric_rotation, abs=1e-7)
