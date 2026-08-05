import numpy as np
import pytest

from formation_control.constraints import (
    HorizontalFieldOfViewConstraint,
    MaximumDistanceConstraint,
    MinimumDistanceConstraint,
    NormalizedImagePoint,
    VerticalFieldOfViewConstraint,
)
from formation_control.geometry import PinholeCamera, skew
from formation_control.potentials import (
    ConstraintBarrierPotential,
    EdgePotential,
    ImageCenteringPotential,
    PositionTrackingPotential,
    RelativePositionPotential,
)


def exp_so3(rotation_vector):
    rotation_vector = np.asarray(rotation_vector, dtype=float)
    angle = np.linalg.norm(rotation_vector)

    if angle < 1e-14:
        return np.eye(3) + skew(rotation_vector)

    axis = rotation_vector / angle
    axis_skew = skew(axis)
    return np.eye(3) + np.sin(angle) * axis_skew + (1.0 - np.cos(angle)) * (axis_skew @ axis_skew)


def build_full_edge_potential():
    desired_relative_position = np.array([2.0, 0.0, 0.0])
    desired_image = NormalizedImagePoint(alpha_h=0.0, alpha_v=0.0)

    collision_constraint = MinimumDistanceConstraint(0.5)
    range_constraint = MaximumDistanceConstraint(4.0)
    horizontal_constraint = HorizontalFieldOfViewConstraint(0.8)
    vertical_constraint = VerticalFieldOfViewConstraint(0.7)

    return EdgePotential(
        formation=RelativePositionPotential.isotropic(
            desired_relative_position,
            gain=2.0,
        ),
        collision_barrier=ConstraintBarrierPotential.from_reference(
            collision_constraint,
            desired_relative_position,
            weight=0.7,
        ),
        range_barrier=ConstraintBarrierPotential.from_reference(
            range_constraint,
            desired_relative_position,
            weight=0.9,
        ),
        image_centering=ImageCenteringPotential(
            horizontal_gain=1.3,
            vertical_gain=1.6,
        ),
        horizontal_fov_barrier=ConstraintBarrierPotential.from_reference(
            horizontal_constraint,
            desired_image,
            weight=0.4,
        ),
        vertical_fov_barrier=ConstraintBarrierPotential.from_reference(
            vertical_constraint,
            desired_image,
            weight=0.5,
        ),
        camera=PinholeCamera.from_degrees(50.0, 40.0),
    )


def finite_difference_position_gradient(
    potential,
    observer_position,
    observer_rotation,
    target_position,
    *,
    with_respect_to,
    epsilon=1e-7,
):
    gradient = np.zeros(3)

    for index in range(3):
        perturbation = np.zeros(3)
        perturbation[index] = epsilon

        observer_plus = observer_position.copy()
        observer_minus = observer_position.copy()
        target_plus = target_position.copy()
        target_minus = target_position.copy()

        if with_respect_to == "observer":
            observer_plus += perturbation
            observer_minus -= perturbation
        elif with_respect_to == "target":
            target_plus += perturbation
            target_minus -= perturbation
        else:
            raise ValueError("invalid differentiation variable")

        value_plus = potential.evaluate(
            observer_plus,
            observer_rotation,
            target_plus,
        ).value
        value_minus = potential.evaluate(
            observer_minus,
            observer_rotation,
            target_minus,
        ).value

        gradient[index] = (value_plus - value_minus) / (2.0 * epsilon)

    return gradient


def test_edge_potential_requires_at_least_one_term():
    with pytest.raises(ValueError):
        EdgePotential()


def test_camera_is_required_for_camera_terms():
    with pytest.raises(ValueError):
        EdgePotential(image_centering=ImageCenteringPotential())


def test_relative_only_edge_potential_has_expected_gradient_signs():
    desired = np.array([1.0, 0.0, 0.0])
    potential = EdgePotential(formation=RelativePositionPotential.isotropic(desired, gain=2.0))

    evaluation = potential.evaluate(
        observer_position=np.array([0.2, -0.1, 0.0]),
        observer_rotation=np.eye(3),
        target_position=np.array([1.6, 0.3, -0.2]),
    )

    relative_position = np.array([1.4, 0.4, -0.2])
    gradient_relative = 2.0 * (relative_position - desired)

    np.testing.assert_allclose(
        evaluation.observer_position_gradient,
        -gradient_relative,
    )
    np.testing.assert_allclose(
        evaluation.target_position_gradient,
        gradient_relative,
    )
    np.testing.assert_allclose(
        evaluation.observer_orientation_gradient_body,
        np.zeros(3),
    )
    assert evaluation.observation is None


def test_full_edge_potential_is_zero_at_nominal_configuration():
    potential = build_full_edge_potential()

    evaluation = potential.evaluate(
        observer_position=np.zeros(3),
        observer_rotation=np.eye(3),
        target_position=np.array([2.0, 0.0, 0.0]),
    )

    assert evaluation.value == pytest.approx(0.0, abs=1e-14)
    np.testing.assert_allclose(
        evaluation.observer_position_gradient,
        np.zeros(3),
        atol=1e-14,
    )
    np.testing.assert_allclose(
        evaluation.target_position_gradient,
        np.zeros(3),
        atol=1e-14,
    )
    np.testing.assert_allclose(
        evaluation.observer_orientation_gradient_body,
        np.zeros(3),
        atol=1e-14,
    )


def test_full_edge_position_gradients_match_finite_differences():
    potential = build_full_edge_potential()
    observer_position = np.array([0.2, -0.3, 0.1])
    observer_rotation = exp_so3(np.array([0.08, -0.1, 0.12]))
    target_position = np.array([2.4, 0.25, -0.15])

    evaluation = potential.evaluate(
        observer_position,
        observer_rotation,
        target_position,
    )

    observer_numeric = finite_difference_position_gradient(
        potential,
        observer_position,
        observer_rotation,
        target_position,
        with_respect_to="observer",
    )
    target_numeric = finite_difference_position_gradient(
        potential,
        observer_position,
        observer_rotation,
        target_position,
        with_respect_to="target",
    )

    np.testing.assert_allclose(
        evaluation.observer_position_gradient,
        observer_numeric,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        evaluation.target_position_gradient,
        target_numeric,
        atol=1e-7,
    )


def test_full_edge_orientation_gradient_matches_finite_difference():
    potential = build_full_edge_potential()
    observer_position = np.array([0.2, -0.3, 0.1])
    observer_rotation = exp_so3(np.array([0.08, -0.1, 0.12]))
    target_position = np.array([2.4, 0.25, -0.15])

    analytic = potential.evaluate(
        observer_position,
        observer_rotation,
        target_position,
    ).observer_orientation_gradient_body

    epsilon = 1e-7
    numeric = np.zeros(3)

    for index in range(3):
        direction = np.zeros(3)
        direction[index] = 1.0

        rotation_plus = observer_rotation @ exp_so3(epsilon * direction)
        rotation_minus = observer_rotation @ exp_so3(-epsilon * direction)

        value_plus = potential.evaluate(
            observer_position,
            rotation_plus,
            target_position,
        ).value
        value_minus = potential.evaluate(
            observer_position,
            rotation_minus,
            target_position,
        ).value

        numeric[index] = (value_plus - value_minus) / (2.0 * epsilon)

    np.testing.assert_allclose(analytic, numeric, atol=1e-7)


def test_edge_component_values_sum_to_total():
    potential = build_full_edge_potential()

    evaluation = potential.evaluate(
        observer_position=np.array([0.0, 0.0, 0.0]),
        observer_rotation=exp_so3(np.array([0.0, 0.0, 0.08])),
        target_position=np.array([2.2, 0.15, -0.1]),
    )

    assert sum(evaluation.components.values()) == pytest.approx(evaluation.value)
    assert set(evaluation.components) == {
        "formation",
        "collision_barrier",
        "range_barrier",
        "image_centering",
        "horizontal_fov_barrier",
        "vertical_fov_barrier",
    }


def test_position_tracking_potential():
    desired = np.array([1.0, -0.5, 0.2])
    potential = PositionTrackingPotential.isotropic(desired, gain=3.0)

    evaluation = potential.evaluate(np.array([1.2, -0.1, 0.0]))
    error = np.array([0.2, 0.4, -0.2])

    assert evaluation.value == pytest.approx(1.5 * float(error @ error))
    np.testing.assert_allclose(evaluation.gradient, 3.0 * error)
