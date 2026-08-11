"""ROS 2 message adapters for the controller state."""

from __future__ import annotations

import numpy as np
from nav_msgs.msg import Odometry
from numpy.typing import NDArray
from px4_msgs.msg import VehicleOdometry

from .frame_conventions import FrameConvention

FloatArray = NDArray[np.float64]


def odometry_to_core_state(
    message: Odometry,
    frames: FrameConvention,
) -> FloatArray:
    """Convert generic ``nav_msgs/Odometry`` to core ``[p,q,nu]``."""
    position_ros = np.array(
        [
            message.pose.pose.position.x,
            message.pose.pose.position.y,
            message.pose.pose.position.z,
        ],
        dtype=float,
    )
    quaternion_xyzw = np.array(
        [
            message.pose.pose.orientation.x,
            message.pose.pose.orientation.y,
            message.pose.pose.orientation.z,
            message.pose.pose.orientation.w,
        ],
        dtype=float,
    )
    linear = np.array(
        [
            message.twist.twist.linear.x,
            message.twist.twist.linear.y,
            message.twist.twist.linear.z,
        ],
        dtype=float,
    )
    angular = np.array(
        [
            message.twist.twist.angular.x,
            message.twist.twist.angular.y,
            message.twist.twist.angular.z,
        ],
        dtype=float,
    )

    position_core = frames.world_position_to_core(position_ros)
    rotation_core, quaternion_core = frames.rotation_to_core(
        quaternion_xyzw
    )
    velocity_core = frames.twist_to_core_body(
        linear,
        angular,
        rotation_body_to_core_world=rotation_core,
    )
    return np.concatenate(
        (
            position_core,
            quaternion_core,
            velocity_core,
        )
    )


def px4_vehicle_odometry_to_core_state(
    message: VehicleOdometry,
) -> FloatArray:
    """Convert PX4 ``VehicleOdometry`` to the core state.

    The current BlueROV SITL publishes:
    - ``pose_frame == 1``: NED;
    - ``velocity_frame == 1``: NED.

    We reject any other frame rather than silently applying an incorrect
    transform.
    """
    pose_frame_ned = int(
        getattr(VehicleOdometry, "POSE_FRAME_NED", 1)
    )
    velocity_frame_ned = int(
        getattr(VehicleOdometry, "VELOCITY_FRAME_NED", 1)
    )

    if int(message.pose_frame) != pose_frame_ned:
        raise ValueError(
            "Unsupported PX4 pose frame "
            f"{message.pose_frame}; expected NED ({pose_frame_ned})."
        )
    if int(message.velocity_frame) != velocity_frame_ned:
        raise ValueError(
            "Unsupported PX4 velocity frame "
            f"{message.velocity_frame}; expected NED "
            f"({velocity_frame_ned})."
        )

    return FrameConvention.px4_vehicle_odometry_to_core(
        position_ned=np.asarray(message.position, dtype=float),
        quaternion_wxyz_frd_to_ned=np.asarray(
            message.q,
            dtype=float,
        ),
        velocity_ned=np.asarray(message.velocity, dtype=float),
        angular_velocity_frd=np.asarray(
            message.angular_velocity,
            dtype=float,
        ),
    )
