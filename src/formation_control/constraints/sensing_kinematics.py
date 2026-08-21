"""Instantaneous values and kinematics of the conservative sensing constraints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from formation_control.geometry import PinholeCamera, rotation_matrix_from_quaternion
from formation_control.models import BlueROV2Model
from formation_control.models.base import FloatArray

from .domains import DistanceDomain, FieldOfViewDomain


@dataclass(frozen=True)
class SensingConstraintValues:
    """Values of the four conservative sensing constraints.

    Channel order is

        [collision, range, horizontal_fov, vertical_fov].

    This configuration-only object intentionally contains no constraint rates,
    so it can be evaluated without the target velocity.
    """

    values: FloatArray
    image_coordinates: FloatArray
    camera_depth: float

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.shape != (4,):
            raise ValueError("values must have shape (4,).")
        if not np.all(np.isfinite(values)):
            raise ValueError("values must contain only finite values.")
        object.__setattr__(self, "values", values.copy())

        image = np.asarray(self.image_coordinates, dtype=float)
        if image.shape != (2,):
            raise ValueError("image_coordinates must have shape (2,).")
        if not np.all(np.isfinite(image)):
            raise ValueError("image_coordinates must contain only finite values.")
        object.__setattr__(self, "image_coordinates", image.copy())

        if not np.isfinite(self.camera_depth):
            raise ValueError("camera_depth must be finite.")


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


def evaluate_sensing_constraint_values(
    camera: PinholeCamera,
    distance_domain: DistanceDomain,
    fov_domain: FieldOfViewDomain,
    observer_state: FloatArray,
    target_position: FloatArray,
) -> SensingConstraintValues:
    """Evaluate the conservative sensing constraints from configuration only.

    The observer pose and target position are sufficient because the four
    constraints depend only on relative configuration.  In particular, the
    target velocity is not required.
    """
    observer = np.asarray(observer_state, dtype=float)
    if observer.ndim != 1 or observer.size < 7:
        raise ValueError(
            "observer_state must be one-dimensional and contain at least "
            "position and quaternion entries."
        )
    if not np.all(np.isfinite(observer[:7])):
        raise ValueError(
            "observer position and quaternion must contain only finite values."
        )

    target = np.asarray(target_position, dtype=float)
    if target.shape != (3,):
        raise ValueError("target_position must have shape (3,).")
    if not np.all(np.isfinite(target)):
        raise ValueError("target_position must contain only finite values.")

    observer_position = observer[:3]
    observer_quaternion = observer[3:7]
    observer_rotation = rotation_matrix_from_quaternion(observer_quaternion)

    relative_position = target - observer_position
    distance_squared = float(relative_position @ relative_position)

    observation = camera.observe(
        observer_position,
        observer_rotation,
        target,
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

    return SensingConstraintValues(
        values=values,
        image_coordinates=image,
        camera_depth=float(observation.point_camera[0]),
    )


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
    forces.  This derivative-based evaluator is retained for diagnostics and
    code paths that explicitly have access to both vehicle velocities.
    """
    observer_position, observer_quaternion, observer_velocity = model.split_state(
        observer_state
    )
    target_position, target_quaternion, target_velocity = model.split_state(
        target_state
    )

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
