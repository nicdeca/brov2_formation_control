"""Pinhole-camera geometry and analytic Jacobians.

Conventions
-----------
The observer pose is described by position ``p_i`` in the inertial frame and
rotation ``R_i`` mapping body-frame vectors to the inertial frame.

For a target point ``p_j`` in the inertial frame,

    p_ij^I = p_j - p_i,
    p_ij^B = R_i.T @ p_ij^I,
    p_ij^C = R_CB @ (p_ij^B - p_CB),

where ``R_CB`` maps body-frame vectors to the camera frame and ``p_CB`` is the
camera origin expressed in the body frame.

The camera optical axis is ``+x_C``.  The normalized image coordinates are

    alpha_h = y_C / (x_C tan(theta_h)),
    alpha_v = z_C / (x_C tan(theta_v)),

so the physical field of view is ``|alpha_h| < 1`` and ``|alpha_v| < 1``.

Orientation Jacobians use a right/body perturbation

    R_i(epsilon) = R_i exp(epsilon [delta_theta]_x).

Accordingly, gradients with respect to orientation are body-frame tangent
vectors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .rotations import skew

FloatArray = NDArray[np.float64]


def _vector3(value: FloatArray, *, name: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _rotation_matrix(value: FloatArray, *, name: str) -> FloatArray:
    rotation = np.asarray(value, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {rotation.shape}.")
    if not np.all(np.isfinite(rotation)):
        raise ValueError(f"{name} must contain only finite values.")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9):
        raise ValueError(f"{name} must be orthogonal.")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-9):
        raise ValueError(f"{name} must have determinant +1.")
    return rotation


@dataclass(frozen=True)
class NormalizedImagePoint:
    """Normalized horizontal and vertical image coordinates."""

    alpha_h: float
    alpha_v: float

    def as_array(self) -> FloatArray:
        """Return ``[alpha_h, alpha_v]``."""
        return np.array([self.alpha_h, self.alpha_v], dtype=float)


@dataclass(frozen=True)
class CameraExtrinsics:
    """Rigid transform from robot body frame to camera frame.

    Parameters
    ----------
    rotation_camera_from_body:
        ``R_CB`` mapping body-frame vectors to the camera frame.
    position_camera_in_body:
        ``p_CB``, the camera origin expressed in body coordinates.
    """

    rotation_camera_from_body: FloatArray
    position_camera_in_body: FloatArray

    def __post_init__(self) -> None:
        rotation = _rotation_matrix(
            self.rotation_camera_from_body,
            name="rotation_camera_from_body",
        )
        position = _vector3(
            self.position_camera_in_body,
            name="position_camera_in_body",
        )
        object.__setattr__(self, "rotation_camera_from_body", rotation)
        object.__setattr__(self, "position_camera_in_body", position)

    @classmethod
    def identity(cls) -> CameraExtrinsics:
        """Return a camera colocated and aligned with the body frame."""
        return cls(
            rotation_camera_from_body=np.eye(3),
            position_camera_in_body=np.zeros(3),
        )


@dataclass(frozen=True)
class CameraObservation:
    """Geometric quantities associated with one observed target."""

    relative_position_inertial: FloatArray
    relative_position_body: FloatArray
    point_camera: FloatArray
    image_point: NormalizedImagePoint


@dataclass(frozen=True)
class PoseGradient:
    """Pullback of an image-space gradient to observer/target pose variables."""

    observer_position: FloatArray
    target_position: FloatArray
    observer_orientation_body: FloatArray


@dataclass(frozen=True)
class CameraJacobians:
    """Analytic Jacobians for one camera observation."""

    point_camera_wrt_observer_position: FloatArray
    point_camera_wrt_target_position: FloatArray
    point_camera_wrt_observer_orientation_body: FloatArray
    image_wrt_point_camera: FloatArray

    @property
    def image_wrt_observer_position(self) -> FloatArray:
        return self.image_wrt_point_camera @ self.point_camera_wrt_observer_position

    @property
    def image_wrt_target_position(self) -> FloatArray:
        return self.image_wrt_point_camera @ self.point_camera_wrt_target_position

    @property
    def image_wrt_observer_orientation_body(self) -> FloatArray:
        return self.image_wrt_point_camera @ self.point_camera_wrt_observer_orientation_body

    def pullback_image_gradient(self, gradient_image: FloatArray) -> PoseGradient:
        """Map ``dV/d[alpha_h, alpha_v]`` to pose gradients."""
        gradient_image = np.asarray(gradient_image, dtype=float)
        if gradient_image.shape != (2,):
            raise ValueError(f"gradient_image must have shape (2,), got {gradient_image.shape}.")

        return PoseGradient(
            observer_position=self.image_wrt_observer_position.T @ gradient_image,
            target_position=self.image_wrt_target_position.T @ gradient_image,
            observer_orientation_body=(self.image_wrt_observer_orientation_body.T @ gradient_image),
        )


@dataclass(frozen=True)
class PinholeCamera:
    """Forward-looking pinhole camera with rectangular field of view."""

    horizontal_half_angle: float
    vertical_half_angle: float
    extrinsics: CameraExtrinsics = CameraExtrinsics.identity()

    def __post_init__(self) -> None:
        for name, angle in (
            ("horizontal_half_angle", self.horizontal_half_angle),
            ("vertical_half_angle", self.vertical_half_angle),
        ):
            if not 0.0 < angle < 0.5 * np.pi:
                raise ValueError(f"{name} must lie strictly between 0 and pi/2.")

    @classmethod
    def from_degrees(
        cls,
        horizontal_half_angle: float,
        vertical_half_angle: float,
        *,
        extrinsics: CameraExtrinsics | None = None,
    ) -> PinholeCamera:
        """Construct a camera from half-angles specified in degrees."""
        return cls(
            horizontal_half_angle=float(np.deg2rad(horizontal_half_angle)),
            vertical_half_angle=float(np.deg2rad(vertical_half_angle)),
            extrinsics=CameraExtrinsics.identity() if extrinsics is None else extrinsics,
        )

    @property
    def horizontal_scale(self) -> float:
        """Return ``tan(theta_h)``."""
        return float(np.tan(self.horizontal_half_angle))

    @property
    def vertical_scale(self) -> float:
        """Return ``tan(theta_v)``."""
        return float(np.tan(self.vertical_half_angle))

    def point_in_camera(
        self,
        observer_position: FloatArray,
        observer_rotation: FloatArray,
        target_position: FloatArray,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return inertial-relative, body-relative, and camera-frame vectors."""
        observer_position = _vector3(observer_position, name="observer_position")
        target_position = _vector3(target_position, name="target_position")
        observer_rotation = _rotation_matrix(
            observer_rotation,
            name="observer_rotation",
        )

        relative_inertial = target_position - observer_position
        relative_body = observer_rotation.T @ relative_inertial
        point_camera = self.extrinsics.rotation_camera_from_body @ (
            relative_body - self.extrinsics.position_camera_in_body
        )

        return relative_inertial, relative_body, point_camera

    def normalized_image_point(self, point_camera: FloatArray) -> NormalizedImagePoint:
        """Project a positive-depth camera-frame point to normalized coordinates."""
        point_camera = _vector3(point_camera, name="point_camera")
        depth, horizontal, vertical = point_camera

        if depth <= 0.0:
            raise ValueError("normalized image coordinates require strictly positive camera depth.")

        return NormalizedImagePoint(
            alpha_h=float(horizontal / (depth * self.horizontal_scale)),
            alpha_v=float(vertical / (depth * self.vertical_scale)),
        )

    def observe(
        self,
        observer_position: FloatArray,
        observer_rotation: FloatArray,
        target_position: FloatArray,
    ) -> CameraObservation:
        """Return all geometric quantities for an observed target."""
        relative_inertial, relative_body, point_camera = self.point_in_camera(
            observer_position,
            observer_rotation,
            target_position,
        )

        return CameraObservation(
            relative_position_inertial=relative_inertial,
            relative_position_body=relative_body,
            point_camera=point_camera,
            image_point=self.normalized_image_point(point_camera),
        )

    def image_jacobian(self, point_camera: FloatArray) -> FloatArray:
        """Return ``d[alpha_h, alpha_v] / d p^C``."""
        point_camera = _vector3(point_camera, name="point_camera")
        depth, horizontal, vertical = point_camera

        if depth <= 0.0:
            raise ValueError("image Jacobian requires strictly positive camera depth.")

        return np.array(
            [
                [
                    -horizontal / (depth**2 * self.horizontal_scale),
                    1.0 / (depth * self.horizontal_scale),
                    0.0,
                ],
                [
                    -vertical / (depth**2 * self.vertical_scale),
                    0.0,
                    1.0 / (depth * self.vertical_scale),
                ],
            ],
            dtype=float,
        )

    def jacobians(
        self,
        observer_position: FloatArray,
        observer_rotation: FloatArray,
        target_position: FloatArray,
    ) -> CameraJacobians:
        """Return analytic position/orientation and image Jacobians."""
        _, relative_body, point_camera = self.point_in_camera(
            observer_position,
            observer_rotation,
            target_position,
        )
        observer_rotation = _rotation_matrix(
            observer_rotation,
            name="observer_rotation",
        )

        rotation_camera_from_inertial = (
            self.extrinsics.rotation_camera_from_body @ observer_rotation.T
        )

        point_wrt_observer_position = -rotation_camera_from_inertial
        point_wrt_target_position = rotation_camera_from_inertial

        # For R(eps) = R exp(eps [delta_theta]_x),
        # d(R.T p)/d delta_theta = [R.T p]_x.
        point_wrt_orientation = self.extrinsics.rotation_camera_from_body @ skew(relative_body)

        return CameraJacobians(
            point_camera_wrt_observer_position=point_wrt_observer_position,
            point_camera_wrt_target_position=point_wrt_target_position,
            point_camera_wrt_observer_orientation_body=point_wrt_orientation,
            image_wrt_point_camera=self.image_jacobian(point_camera),
        )
