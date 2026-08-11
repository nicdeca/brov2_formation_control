"""Explicit frame conversions used at the ROS 2 boundary.

The ROS-independent ``formation_control`` core uses:

* inertial coordinates: project convention, currently NWU in the experiments;
* body coordinates: FLU (x forward, y left, z up);
* quaternion: scalar-first ``[q_w, q_x, q_y, q_z]``;
* rotation matrix: body -> inertial;
* generalized velocity: ``[v_body, omega_body]``.

The old ROS 2/PX4 workspace implicitly mixed several conventions.  This module
makes every conversion explicit and keeps those conversions OUTSIDE the core
controller.

Supported incoming MoCap/world conventions
------------------------------------------

``core_nwu``
    The incoming Odometry position/orientation already uses the project's
    inertial frame.  This matches how the student's MoCap node was previously
    interpreted and is therefore the default for the first integration tests.

``ros_enu``
    Standard ROS ENU coordinates.  The conversion to project NWU is

        [N, W, U]^T = [[ 0,  1, 0],
                       [-1,  0, 0],
                       [ 0,  0, 1]] [E, N, U]^T.

Body-frame convention
---------------------

The project core uses FLU.  The ROS Odometry child frame is assumed FLU.
PX4 thrust/torque setpoints use FRD, hence

    [x_FRD, y_FRD, z_FRD] = diag(1, -1, -1) [x_FLU, y_FLU, z_FLU].

Both force and torque are transformed with this proper 180-degree rotation.

Odometry twist convention
-------------------------

ROS ``nav_msgs/Odometry`` defines the twist in ``child_frame_id``.  Some MoCap
bridges nevertheless publish world-frame velocities.  The adapter therefore
has an explicit ``twist_frame`` parameter:

``body``
    Treat linear and angular velocities as FLU body coordinates.

``world``
    Treat them as incoming world-frame vectors and rotate them into the core
    body frame.

Do not change signs ad hoc in controller nodes.  Any new frame convention
belongs here, with a unit test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
WorldFrame = Literal["core_nwu", "ros_enu"]
TwistFrame = Literal["body", "world"]

_ENU_TO_NWU = np.array(
    [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=float,
)

_FLU_TO_FRD = np.diag([1.0, -1.0, -1.0])

# PX4 VehicleOdometry uses NED for pose/velocity when the corresponding
# enum value is 1 and FRD for the body frame.
_NED_TO_NWU = np.diag([1.0, -1.0, -1.0])
_FRD_TO_FLU = np.diag([1.0, -1.0, -1.0])


def _vector3(value: FloatArray, *, name: str) -> FloatArray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {vector.shape}.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain finite values.")
    return vector


def _rotation_matrix_from_xyzw(quaternion_xyzw: FloatArray) -> FloatArray:
    """Return body->world rotation from a ROS xyzw quaternion."""
    qx, qy, qz, qw = np.asarray(quaternion_xyzw, dtype=float).reshape(4)
    norm = float(np.linalg.norm([qw, qx, qy, qz]))
    if norm <= np.finfo(float).eps:
        raise ValueError("quaternion norm must be nonzero.")
    qw, qx, qy, qz = np.array([qw, qx, qy, qz]) / norm
    return np.array(
        [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qw * qz),
                2.0 * (qx * qz + qw * qy),
            ],
            [
                2.0 * (qx * qy + qw * qz),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qw * qx),
            ],
            [
                2.0 * (qx * qz - qw * qy),
                2.0 * (qy * qz + qw * qx),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ],
        dtype=float,
    )


def _scalar_first_quaternion_from_rotation(rotation: FloatArray) -> FloatArray:
    """Convert a proper rotation matrix to scalar-first quaternion."""
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3).")

    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        qw = (matrix[2, 1] - matrix[1, 2]) / scale
        qx = 0.25 * scale
        qy = (matrix[0, 1] + matrix[1, 0]) / scale
        qz = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        qw = (matrix[0, 2] - matrix[2, 0]) / scale
        qx = (matrix[0, 1] + matrix[1, 0]) / scale
        qy = 0.25 * scale
        qz = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        qw = (matrix[1, 0] - matrix[0, 1]) / scale
        qx = (matrix[0, 2] + matrix[2, 0]) / scale
        qy = (matrix[1, 2] + matrix[2, 1]) / scale
        qz = 0.25 * scale

    quaternion = np.array([qw, qx, qy, qz], dtype=float)
    quaternion /= np.linalg.norm(quaternion)

    # q and -q describe the same rotation.  Keeping q_w nonnegative reduces
    # sign jumps in logged state histories.
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return quaternion


@dataclass(frozen=True, slots=True)
class FrameConvention:
    """Conversions between ROS/MoCap, core, and PX4 coordinate conventions."""

    world_frame: WorldFrame = "core_nwu"
    twist_frame: TwistFrame = "body"

    def __post_init__(self) -> None:
        if self.world_frame not in ("core_nwu", "ros_enu"):
            raise ValueError(
                "world_frame must be 'core_nwu' or 'ros_enu'."
            )
        if self.twist_frame not in ("body", "world"):
            raise ValueError("twist_frame must be 'body' or 'world'.")

    @property
    def core_from_ros_world(self) -> FloatArray:
        if self.world_frame == "core_nwu":
            return np.eye(3)
        return _ENU_TO_NWU.copy()

    def world_vector_to_core(self, vector: FloatArray) -> FloatArray:
        return self.core_from_ros_world @ _vector3(
            vector,
            name="world vector",
        )

    def world_position_to_core(self, position: FloatArray) -> FloatArray:
        return self.world_vector_to_core(position)

    def rotation_to_core(
        self,
        quaternion_xyzw: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        """Return ``(R_body_to_core_world, q_core_scalar_first)``."""
        rotation_ros = _rotation_matrix_from_xyzw(quaternion_xyzw)
        rotation_core = self.core_from_ros_world @ rotation_ros
        quaternion_core = _scalar_first_quaternion_from_rotation(
            rotation_core
        )
        return rotation_core, quaternion_core

    def twist_to_core_body(
        self,
        linear: FloatArray,
        angular: FloatArray,
        *,
        rotation_body_to_core_world: FloatArray,
    ) -> FloatArray:
        linear = _vector3(linear, name="linear velocity")
        angular = _vector3(angular, name="angular velocity")

        if self.twist_frame == "body":
            linear_body = linear
            angular_body = angular
        else:
            linear_world = self.world_vector_to_core(linear)
            angular_world = self.world_vector_to_core(angular)
            rotation = np.asarray(
                rotation_body_to_core_world,
                dtype=float,
            ).reshape(3, 3)
            linear_body = rotation.T @ linear_world
            angular_body = rotation.T @ angular_world

        return np.concatenate((linear_body, angular_body))

    @staticmethod
    def px4_ned_position_to_core_nwu(
        position_ned: FloatArray,
    ) -> FloatArray:
        """Convert PX4 NED position to project NWU position."""
        return _NED_TO_NWU @ _vector3(
            position_ned,
            name="PX4 NED position",
        )

    @staticmethod
    def px4_vehicle_odometry_to_core(
        *,
        position_ned: FloatArray,
        quaternion_wxyz_frd_to_ned: FloatArray,
        velocity_ned: FloatArray,
        angular_velocity_frd: FloatArray,
    ) -> FloatArray:
        """Convert a PX4 VehicleOdometry sample to core ``[p,q,nu]``.

        Preconditions
        -------------
        ``pose_frame == POSE_FRAME_NED`` and
        ``velocity_frame == VELOCITY_FRAME_NED``.

        PX4 supplies:
        - position in NED;
        - quaternion [w,x,y,z] mapping body FRD -> NED;
        - linear velocity in NED;
        - angular velocity in body FRD.

        The core requires:
        - position in NWU;
        - quaternion [w,x,y,z] mapping body FLU -> NWU;
        - linear and angular velocity in body FLU.
        """
        position_core = _NED_TO_NWU @ _vector3(
            position_ned,
            name="PX4 NED position",
        )

        quaternion = np.asarray(
            quaternion_wxyz_frd_to_ned,
            dtype=float,
        )
        if quaternion.shape != (4,):
            raise ValueError("PX4 quaternion must have shape (4,).")
        # helper expects ROS xyzw
        rotation_frd_to_ned = _rotation_matrix_from_xyzw(
            np.array(
                [
                    quaternion[1],
                    quaternion[2],
                    quaternion[3],
                    quaternion[0],
                ],
                dtype=float,
            )
        )

        # v_NWU = S_world R_NED<-FRD S_body v_FLU.
        rotation_flu_to_nwu = (
            _NED_TO_NWU
            @ rotation_frd_to_ned
            @ _FRD_TO_FLU
        )
        quaternion_core = _scalar_first_quaternion_from_rotation(
            rotation_flu_to_nwu
        )

        velocity_core_world = _NED_TO_NWU @ _vector3(
            velocity_ned,
            name="PX4 NED velocity",
        )
        linear_velocity_flu = (
            rotation_flu_to_nwu.T @ velocity_core_world
        )
        angular_velocity_flu = _FRD_TO_FLU @ _vector3(
            angular_velocity_frd,
            name="PX4 FRD angular velocity",
        )

        return np.concatenate(
            (
                position_core,
                quaternion_core,
                linear_velocity_flu,
                angular_velocity_flu,
            )
        )

    @staticmethod
    def core_body_vector_to_px4_frd(vector: FloatArray) -> FloatArray:
        """Transform a core FLU body vector into PX4 FRD coordinates."""
        return _FLU_TO_FRD @ _vector3(vector, name="body vector")

    @staticmethod
    def core_wrench_to_px4_frd(wrench: FloatArray) -> FloatArray:
        wrench = np.asarray(wrench, dtype=float)
        if wrench.shape != (6,):
            raise ValueError("wrench must have shape (6,).")
        return np.concatenate(
            (
                _FLU_TO_FRD @ wrench[:3],
                _FLU_TO_FRD @ wrench[3:],
            )
        )
