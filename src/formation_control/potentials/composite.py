"""Composite configuration potentials for directed sensing edges.

An edge ``(i, j)`` means that robot ``i`` observes robot ``j``.  The relative
position convention is

    p_ij = p_j - p_i.

The edge potential can combine:

* a formation-regulation term in ``p_ij``;
* minimum- and maximum-distance barriers in ``p_ij``;
* a smooth image-centering objective in normalized camera coordinates;
* horizontal and vertical field-of-view barriers.

Camera-dependent gradients are pulled back analytically to the observer
position, target position, and observer body-frame orientation tangent.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from formation_control.geometry import CameraObservation, PinholeCamera

from .base import EuclideanPotential

FloatArray = NDArray[np.float64]


def _vector3(value: FloatArray, *, name: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


@dataclass(frozen=True)
class EdgePotentialEvaluation:
    """Value, gradients, and diagnostics for one directed edge."""

    value: float
    observer_position_gradient: FloatArray
    target_position_gradient: FloatArray
    observer_orientation_gradient_body: FloatArray
    components: Mapping[str, float]
    enlargement_derivatives: Mapping[str, float] | None = None
    observation: CameraObservation | None = None

    def __post_init__(self) -> None:
        for name in (
            "observer_position_gradient",
            "target_position_gradient",
            "observer_orientation_gradient_body",
        ):
            gradient = _vector3(getattr(self, name), name=name)
            object.__setattr__(self, name, gradient)

        if not np.isfinite(self.value):
            raise ValueError("edge-potential value must be finite.")

        components = dict(self.components)
        if not all(np.isfinite(value) for value in components.values()):
            raise ValueError("all component values must be finite.")

        object.__setattr__(
            self,
            "components",
            MappingProxyType(components),
        )

        enlargement_derivatives = (
            {} if self.enlargement_derivatives is None else dict(self.enlargement_derivatives)
        )
        if not all(np.isfinite(value) for value in enlargement_derivatives.values()):
            raise ValueError("all enlargement derivatives must be finite.")
        object.__setattr__(
            self,
            "enlargement_derivatives",
            MappingProxyType(enlargement_derivatives),
        )


@dataclass(frozen=True)
class EdgePotential:
    """Composite potential associated with one sensing edge.

    Each term is optional.  At least one term must be supplied.

    Relative-position terms are differentiated with respect to
    ``p_ij = p_j - p_i``.  Camera terms are expressed in normalized image
    coordinates and are pulled back through the camera model.
    """

    formation: EuclideanPotential[FloatArray] | None = None
    collision_barrier: EuclideanPotential[FloatArray] | None = None
    range_barrier: EuclideanPotential[FloatArray] | None = None
    image_centering: EuclideanPotential[object] | None = None
    horizontal_fov_barrier: EuclideanPotential[object] | None = None
    vertical_fov_barrier: EuclideanPotential[object] | None = None
    camera: PinholeCamera | None = None

    def __post_init__(self) -> None:
        has_relative_term = any(
            term is not None
            for term in (
                self.formation,
                self.collision_barrier,
                self.range_barrier,
            )
        )
        has_camera_term = any(
            term is not None
            for term in (
                self.image_centering,
                self.horizontal_fov_barrier,
                self.vertical_fov_barrier,
            )
        )

        if not has_relative_term and not has_camera_term:
            raise ValueError("EdgePotential requires at least one potential term.")

        if has_camera_term and self.camera is None:
            raise ValueError("camera is required when camera-dependent terms are used.")

    @property
    def uses_camera(self) -> bool:
        """Return whether the potential contains a camera-dependent term."""
        return any(
            term is not None
            for term in (
                self.image_centering,
                self.horizontal_fov_barrier,
                self.vertical_fov_barrier,
            )
        )

    def evaluate(
        self,
        observer_position: FloatArray,
        observer_rotation: FloatArray,
        target_position: FloatArray,
    ) -> EdgePotentialEvaluation:
        """Evaluate the complete edge potential and its pose gradients."""
        observer_position = _vector3(
            observer_position,
            name="observer_position",
        )
        target_position = _vector3(
            target_position,
            name="target_position",
        )

        relative_position = target_position - observer_position

        total_value = 0.0
        observer_position_gradient = np.zeros(3)
        target_position_gradient = np.zeros(3)
        observer_orientation_gradient_body = np.zeros(3)
        components: dict[str, float] = {}
        enlargement_derivatives: dict[str, float] = {}

        relative_terms = (
            ("formation", self.formation),
            ("collision_barrier", self.collision_barrier),
            ("range_barrier", self.range_barrier),
        )

        for name, potential in relative_terms:
            if potential is None:
                continue

            evaluation = potential.evaluate(relative_position)
            gradient_relative = _vector3(
                evaluation.gradient,
                name=f"{name} gradient",
            )

            total_value += evaluation.value
            observer_position_gradient -= gradient_relative
            target_position_gradient += gradient_relative
            components[name] = evaluation.value
            derivative = getattr(
                potential,
                "enlargement_derivative",
                None,
            )
            if callable(derivative):
                enlargement_derivatives[name] = float(derivative(relative_position))

        observation = None

        if self.uses_camera:
            assert self.camera is not None
            observation = self.camera.observe(
                observer_position,
                observer_rotation,
                target_position,
            )
            jacobians = self.camera.jacobians(
                observer_position,
                observer_rotation,
                target_position,
            )

            camera_terms = (
                ("image_centering", self.image_centering),
                ("horizontal_fov_barrier", self.horizontal_fov_barrier),
                ("vertical_fov_barrier", self.vertical_fov_barrier),
            )

            for name, potential in camera_terms:
                if potential is None:
                    continue

                evaluation = potential.evaluate(observation.image_point)
                gradient_image = np.asarray(evaluation.gradient, dtype=float)
                if gradient_image.shape != (2,):
                    raise ValueError(
                        f"{name} gradient must have shape (2,), got {gradient_image.shape}."
                    )

                pose_gradient = jacobians.pullback_image_gradient(gradient_image)

                total_value += evaluation.value
                observer_position_gradient += pose_gradient.observer_position
                target_position_gradient += pose_gradient.target_position
                observer_orientation_gradient_body += pose_gradient.observer_orientation_body
                components[name] = evaluation.value
                derivative = getattr(
                    potential,
                    "enlargement_derivative",
                    None,
                )
                if callable(derivative):
                    enlargement_derivatives[name] = float(derivative(observation.image_point))

        return EdgePotentialEvaluation(
            value=total_value,
            observer_position_gradient=observer_position_gradient,
            target_position_gradient=target_position_gradient,
            observer_orientation_gradient_body=(observer_orientation_gradient_body),
            components=components,
            enlargement_derivatives=enlargement_derivatives,
            observation=observation,
        )
