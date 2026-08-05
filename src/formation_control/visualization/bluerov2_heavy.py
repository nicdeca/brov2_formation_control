"""Approximate BlueROV2 Heavy wireframe for 3-D visualization.

The model is intentionally lightweight rather than CAD-level.  It captures the
recognizable geometry of the vehicle:

* the Heavy outer frame;
* electronics and battery enclosures;
* top buoyancy blocks and lower ballast rails;
* four vectored horizontal T200 thrusters;
* four external vertical T200 thrusters;
* front camera and light pods.

Dimensions are in metres and are chosen to match the overall BlueROV2 Heavy
envelope and T200 scale closely enough for simulation visualization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from formation_control.actuation import BlueROV2HeavyThrusterConfiguration
from formation_control.models.base import FloatArray

from .vehicle_geometry import RigidBodyWireframe, WireframePart


def _box_segments(
    center: FloatArray,
    size: FloatArray,
) -> FloatArray:
    center = np.asarray(center, dtype=float)
    size = np.asarray(size, dtype=float)

    half = 0.5 * size
    vertices = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=float,
    )
    vertices = center + vertices * half

    edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )

    return np.asarray([[vertices[first], vertices[second]] for first, second in edges])


def _orthogonal_basis(axis: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)

    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(axis @ reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])

    first = np.cross(axis, reference)
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)

    return axis, first, second


def _cylinder_segments(
    center: FloatArray,
    axis: FloatArray,
    *,
    length: float,
    radius: float,
    n_circle: int = 10,
    longitudinal_lines: int = 4,
) -> FloatArray:
    """Return a sparse wireframe cylinder."""
    center = np.asarray(center, dtype=float)
    axis, first, second = _orthogonal_basis(axis)

    angles = np.linspace(
        0.0,
        2.0 * np.pi,
        n_circle,
        endpoint=False,
    )
    radial = np.array(
        [radius * (np.cos(angle) * first + np.sin(angle) * second) for angle in angles]
    )

    start_center = center - 0.5 * length * axis
    end_center = center + 0.5 * length * axis
    start_ring = start_center + radial
    end_ring = end_center + radial

    segments = []

    for ring in (start_ring, end_ring):
        for index in range(n_circle):
            segments.append(
                [
                    ring[index],
                    ring[(index + 1) % n_circle],
                ]
            )

    step = max(1, n_circle // longitudinal_lines)
    for index in range(0, n_circle, step):
        segments.append([start_ring[index], end_ring[index]])

    segments.append([start_center, end_center])

    return np.asarray(segments, dtype=float)


def _combine(*groups: FloatArray) -> FloatArray:
    nonempty = [group for group in groups if group.size]
    if not nonempty:
        return np.empty((0, 2, 3))
    return np.concatenate(nonempty, axis=0)


@dataclass(frozen=True)
class BlueROV2HeavyVisualGeometry:
    """Lightweight geometry parameters for a BlueROV2 Heavy.

    Thruster positions and axes are sourced from the same physical
    configuration used by control allocation.
    """

    thruster_configuration: BlueROV2HeavyThrusterConfiguration = (
        BlueROV2HeavyThrusterConfiguration.default_45deg()
    )
    length: float = 0.457
    width: float = 0.575
    height: float = 0.254

    electronics_length: float = 0.30
    electronics_radius: float = 0.055

    battery_length: float = 0.27
    battery_radius: float = 0.041

    thruster_length: float = 0.113
    thruster_radius: float = 0.050

    def __post_init__(self) -> None:
        values = (
            self.length,
            self.width,
            self.height,
            self.electronics_length,
            self.electronics_radius,
            self.battery_length,
            self.battery_radius,
            self.thruster_length,
            self.thruster_radius,
        )
        if any(value <= 0.0 for value in values):
            raise ValueError("all BlueROV2 visual dimensions must be positive.")

    def wireframe(self) -> RigidBodyWireframe:
        """Build the body-frame wireframe.

        Body ``+x`` is forward, ``+y`` is starboard, and ``+z`` follows the
        vehicle/body convention used by the dynamics.
        """
        # Main Heavy frame.  The vertical-thruster guards define the large
        # lateral envelope, so the box uses the full Heavy width.
        frame = _box_segments(
            np.zeros(3),
            np.array(
                [
                    self.length,
                    self.width,
                    self.height,
                ]
            ),
        )

        # Inner central chassis, making the frame more recognizable than a
        # single bounding box.
        inner_frame = _box_segments(
            np.array([0.0, 0.0, 0.0]),
            np.array([0.40, 0.27, 0.20]),
        )

        # Main electronics tube and lower battery tube.
        electronics = _cylinder_segments(
            np.array([0.0, 0.0, 0.015]),
            np.array([1.0, 0.0, 0.0]),
            length=self.electronics_length,
            radius=self.electronics_radius,
            n_circle=12,
        )
        battery = _cylinder_segments(
            np.array([-0.015, 0.0, -0.072]),
            np.array([1.0, 0.0, 0.0]),
            length=self.battery_length,
            radius=self.battery_radius,
            n_circle=10,
        )

        # Approximate top buoyancy blocks/fairings.
        buoyancy = _combine(
            _box_segments(
                np.array([-0.07, -0.070, 0.094]),
                np.array([0.24, 0.095, 0.055]),
            ),
            _box_segments(
                np.array([-0.07, 0.070, 0.094]),
                np.array([0.24, 0.095, 0.055]),
            ),
        )

        # Lower ballast rails/weights.
        ballast = _combine(
            _box_segments(
                np.array([0.0, -0.085, -0.108]),
                np.array([0.30, 0.030, 0.025]),
            ),
            _box_segments(
                np.array([0.0, 0.085, -0.108]),
                np.array([0.30, 0.030, 0.025]),
            ),
        )

        # The allocation geometry is the single source of truth for all
        # eight T200 positions and thrust axes.
        thruster_positions = self.thruster_configuration.positions_body
        thruster_directions = self.thruster_configuration.directions_body

        horizontal_thrusters = _combine(
            *[
                _cylinder_segments(
                    thruster_positions[index],
                    thruster_directions[index],
                    length=self.thruster_length,
                    radius=self.thruster_radius,
                    n_circle=8,
                )
                for index in range(4)
            ]
        )

        vertical_thrusters = _combine(
            *[
                _cylinder_segments(
                    thruster_positions[index],
                    thruster_directions[index],
                    length=self.thruster_length,
                    radius=self.thruster_radius,
                    n_circle=8,
                )
                for index in range(4, 8)
            ]
        )

        # Forward camera body and the two front light pods.
        camera = _box_segments(
            np.array([0.205, 0.0, 0.015]),
            np.array([0.045, 0.060, 0.045]),
        )
        lights = _combine(
            _cylinder_segments(
                np.array([0.190, -0.095, 0.040]),
                np.array([1.0, 0.0, 0.0]),
                length=0.055,
                radius=0.022,
                n_circle=8,
            ),
            _cylinder_segments(
                np.array([0.190, 0.095, 0.040]),
                np.array([1.0, 0.0, 0.0]),
                length=0.055,
                radius=0.022,
                n_circle=8,
            ),
        )

        return RigidBodyWireframe(
            parts=(
                WireframePart(
                    _combine(frame, inner_frame),
                    linewidth=1.2,
                    alpha=0.75,
                ),
                WireframePart(
                    _combine(electronics, battery),
                    linewidth=1.0,
                    alpha=0.9,
                ),
                WireframePart(
                    _combine(buoyancy, ballast),
                    linewidth=0.9,
                    alpha=0.65,
                ),
                WireframePart(
                    _combine(
                        horizontal_thrusters,
                        vertical_thrusters,
                    ),
                    linewidth=0.9,
                    alpha=0.85,
                ),
                WireframePart(
                    _combine(camera, lights),
                    linewidth=1.0,
                    alpha=0.9,
                ),
            )
        )
