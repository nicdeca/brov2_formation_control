"""Instantaneous kinematics of the conservative sensing constraints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from formation_control.geometry import PinholeCamera, rotation_matrix_from_quaternion
from formation_control.models import BlueROV2Model
from formation_control.models.base import FloatArray

from .domains import DistanceDomain, FieldOfViewDomain


@dataclass(frozen=True)
class SensingConstraintKinematics:
    """Values and time derivatives of the four conservative constraints.

    Channel order is

        [collision, range, horizontal_fov, vertical_fov].
    """

    values: FloatArray
    rates: FloatArray
    image_coordinates: FloatArray
    image_rates: FloatArray
    camera_depth: float

    def __post_init__(self) -> None:
        for name in ("values", "rates", "image_coordinates", "image_rates"):
            array = np.asarray(getattr(self, name), dtype=float)
            expected = (4,) if name in ("values", "rates") else (2,)
            if array.shape != expected:
                raise ValueError(f"{name} must have shape {expected}.")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must contain only finite values.")
            object.__setattr__(self, name, array.copy())

        if not np.isfinite(self.camera_depth):
            raise ValueError("camera_depth must be finite.")


def evaluate_sensing_constraint_kinematics(
    model: BlueROV2Model,
    camera: PinholeCamera,
    distance_domain: DistanceDomain,
    fov_domain: FieldOfViewDomain,
    observer_state: FloatArray,
    target_state: FloatArray,
) -> SensingConstraintKinematics:
    """Evaluate ``h_c`` and ``h_c_dot`` from the current measured state.

    Since the constraints depend only on configuration, their first derivative
    depends on the measured generalized velocities but not on the thruster
    forces.  The resulting quantities can therefore be used as coefficients in
    the same CLF-QP without creating an algebraic loop.
    """
    observer_position, observer_quaternion, observer_velocity = model.split_state(observer_state)
    target_position, target_quaternion, target_velocity = model.split_state(target_state)

    observer_rotation = rotation_matrix_from_quaternion(observer_quaternion)
    target_rotation = rotation_matrix_from_quaternion(target_quaternion)

    observer_linear_inertial = observer_rotation @ observer_velocity[:3]
    target_linear_inertial = target_rotation @ target_velocity[:3]
    relative_position = target_position - observer_position
    relative_velocity = target_linear_inertial - observer_linear_inertial

    distance_squared = float(relative_position @ relative_position)
    distance_squared_rate = 2.0 * float(relative_position @ relative_velocity)

    observation = camera.observe(
        observer_position,
        observer_rotation,
        target_position,
    )
    jacobians = camera.jacobians(
        observer_position,
        observer_rotation,
        target_position,
    )

    image_rate = (
        jacobians.image_wrt_observer_position @ observer_linear_inertial
        + jacobians.image_wrt_target_position @ target_linear_inertial
        + jacobians.image_wrt_observer_orientation_body @ observer_velocity[3:]
    )
    image = observation.image_point.as_array()

    values = np.array(
        [
            distance_squared - distance_domain.d_min_conservative**2,
            distance_domain.d_max_conservative**2 - distance_squared,
            fov_domain.alpha_h_conservative**2 - image[0] ** 2,
            fov_domain.alpha_v_conservative**2 - image[1] ** 2,
        ],
        dtype=float,
    )
    rates = np.array(
        [
            distance_squared_rate,
            -distance_squared_rate,
            -2.0 * image[0] * image_rate[0],
            -2.0 * image[1] * image_rate[1],
        ],
        dtype=float,
    )

    return SensingConstraintKinematics(
        values=values,
        rates=rates,
        image_coordinates=image,
        image_rates=image_rate,
        camera_depth=float(observation.point_camera[0]),
    )
