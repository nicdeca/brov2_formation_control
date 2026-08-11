import numpy as np

from formation_control_ros.frame_conventions import FrameConvention


def test_core_nwu_identity_world_conversion():
    frames = FrameConvention(
        world_frame="core_nwu",
        twist_frame="body",
    )
    vector = np.array([1.0, -2.0, 3.0])
    np.testing.assert_allclose(
        frames.world_vector_to_core(vector),
        vector,
    )


def test_ros_enu_to_core_nwu():
    frames = FrameConvention(
        world_frame="ros_enu",
        twist_frame="body",
    )
    # East, North, Up -> North, West, Up
    np.testing.assert_allclose(
        frames.world_vector_to_core(
            np.array([2.0, 3.0, 4.0])
        ),
        np.array([3.0, -2.0, 4.0]),
    )


def test_core_flu_to_px4_frd_wrench():
    frames = FrameConvention()
    wrench_flu = np.array(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    )
    np.testing.assert_allclose(
        frames.core_wrench_to_px4_frd(wrench_flu),
        np.array([1.0, -2.0, -3.0, 4.0, -5.0, -6.0]),
    )


def test_ros_identity_quaternion_becomes_scalar_first_core_quaternion():
    frames = FrameConvention(world_frame="core_nwu")
    rotation, quaternion = frames.rotation_to_core(
        np.array([0.0, 0.0, 0.0, 1.0])
    )
    np.testing.assert_allclose(rotation, np.eye(3))
    np.testing.assert_allclose(
        quaternion,
        np.array([1.0, 0.0, 0.0, 0.0]),
    )


def test_px4_ned_frd_identity_pose_maps_to_core_nwu_flu_identity():
    frames = FrameConvention()
    state = frames.px4_vehicle_odometry_to_core(
        position_ned=np.array([1.0, 2.0, 3.0]),
        quaternion_wxyz_frd_to_ned=np.array([1.0, 0.0, 0.0, 0.0]),
        velocity_ned=np.array([1.0, 0.0, 0.0]),
        angular_velocity_frd=np.array([1.0, 2.0, 3.0]),
    )
    np.testing.assert_allclose(state[:3], np.array([1.0, -2.0, -3.0]))
    np.testing.assert_allclose(state[3:7], np.array([1.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(state[7:10], np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(state[10:13], np.array([1.0, -2.0, -3.0]))
