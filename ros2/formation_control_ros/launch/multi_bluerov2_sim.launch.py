#!/usr/bin/env python3
"""Launch N PX4 SITL BlueROV2 Heavy vehicles in one Gazebo world."""

from __future__ import annotations

import os
import shlex
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration

MAX_ROBOTS = 6


def _default_px4_dir() -> str:
    candidates = [
        os.environ.get("PX4_AUTOPILOT_DIR"),
        "~/PX4-Autopilot",
        "~/Gits/KTH-PX4/PX4-Autopilot",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.is_dir():
            return str(path.resolve())
    return str(Path("~/Github/PX4-Autopilot").expanduser())


DEFAULT_PX4_DIR = _default_px4_dir()


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")

# Gazebo ENU poses: x,y,z,roll,pitch,yaw.
#
# The world has been re-anchored so that PX4 local NED uses the same pool
# origin/orientation as the real Marinarium convention.  These poses are the
# rigidly transformed versions of the previous validated spawn geometry.
# The -90 deg Gazebo yaw preserves each vehicle's orientation relative to the
# rotated tank.
DEFAULT_POSES = [
    "-1.050,2.675,-1.275,0,0,-1.57079632679",  # robot 1
    "-0.400,4.175,-1.275,0,0,-1.57079632679",  # robot 2
    "-1.700,4.775,-1.275,0,0,-1.57079632679",  # robot 3
    "-0.050,5.325,-1.275,0,0,-1.57079632679",  # robot 4
    "-1.550,5.775,-1.275,0,0,-1.57079632679",  # robot 5
    "-1.200,5.925,-1.275,0,0,-1.57079632679",  # robot 6
]


def _launch_setup(context, *args, **kwargs):
    px4_dir = (
        Path(LaunchConfiguration("px4_dir").perform(context)).expanduser().resolve()
    )
    px4_binary = px4_dir / "build" / "px4_sitl_uuv" / "bin" / "px4"

    if not px4_dir.is_dir():
        raise RuntimeError(
            f"PX4-Autopilot directory not found: {px4_dir}\n"
            "Pass px4_dir:=/absolute/path/to/PX4-Autopilot."
        )
    if not px4_binary.is_file():
        raise RuntimeError(
            f"PX4 SITL UUV binary not found: {px4_binary}\n"
            f"Build with: cd {px4_dir} && make px4_sitl_uuv"
        )

    robot_count = int(LaunchConfiguration("robot_count").perform(context))
    if not 1 <= robot_count <= MAX_ROBOTS:
        raise RuntimeError(
            f"robot_count must lie in [1, {MAX_ROBOTS}], got {robot_count}."
        )

    world = LaunchConfiguration("world").perform(context).strip()
    recording_mode = _as_bool(
        LaunchConfiguration("recording_mode").perform(context)
    )
    if recording_mode:
        world = LaunchConfiguration("recording_world").perform(context).strip()
        if not world:
            raise RuntimeError("recording_world must be nonempty in recording mode.")

    spawn_delay = float(LaunchConfiguration("spawn_delay").perform(context))
    if spawn_delay < 0.0:
        raise RuntimeError("spawn_delay must be nonnegative.")

    robot_names = [
        LaunchConfiguration(f"robot_{index + 1}_name").perform(context).strip()
        for index in range(robot_count)
    ]
    if any(not name for name in robot_names):
        raise RuntimeError("All robot names must be nonempty.")
    if len(set(robot_names)) != len(robot_names):
        raise RuntimeError(
            "Robot names must be unique; got "
            + ", ".join(repr(name) for name in robot_names)
            + "."
        )

    actions = []
    for index in range(robot_count):
        instance = index
        namespace = robot_names[index]
        pose = LaunchConfiguration(f"rov_{index + 1}_pose").perform(context)
        delay = spawn_delay * index

        command = (
            f"sleep {delay} && " f"exec {shlex.quote(str(px4_binary))} -i {instance}"
        )

        vehicle_env = {
            **os.environ,
            "PX4_SYS_AUTOSTART": "60002",
            "PX4_SIM_MODEL": "gz_uuv_bluerov2_heavy",
            "PX4_GZ_MODEL_POSE": pose,
            "PX4_UXRCE_DDS_NS": namespace,
            "PX4_GZ_WORLD": world,
        }
        if instance > 0:
            vehicle_env["PX4_GZ_STANDALONE"] = "1"

        actions.append(
            ExecuteProcess(
                cmd=["bash", "-c", command],
                cwd=str(px4_dir),
                env=vehicle_env,
                output="screen",
                emulate_tty=True,
                name=f"px4_{namespace}",
                sigterm_timeout="5",
                sigkill_timeout="5",
            )
        )

    return actions


def generate_launch_description() -> LaunchDescription:
    actions = [
        DeclareLaunchArgument(
            "px4_dir",
            default_value=DEFAULT_PX4_DIR,
            description="Path to the PX4-Autopilot checkout.",
        ),
        DeclareLaunchArgument(
            "world",
            default_value="kth_marinarium_docking",
            description="Gazebo world name without .sdf.",
        ),
        DeclareLaunchArgument(
            "recording_mode",
            default_value="false",
            description=(
                "Use the dedicated Gazebo state-recording world. This records the "
                "Gazebo simulation state without live video encoding, avoiding "
                "the rendering lag caused by the GUI VideoRecorder."
            ),
        ),
        DeclareLaunchArgument(
            "recording_world",
            default_value="kth_marinarium_docking_record",
            description=(
                "Gazebo world name used when recording_mode:=true. The world should "
                "contain gz::sim::systems::LogRecord and no live VideoRecorder."
            ),
        ),
        DeclareLaunchArgument(
            "robot_count",
            default_value="3",
            description=f"Number of BlueROVs to launch (1..{MAX_ROBOTS}).",
        ),
        DeclareLaunchArgument(
            "spawn_delay",
            default_value="5.0",
            description="Delay [s] between successive PX4 instances.",
        ),
    ]

    for index in range(MAX_ROBOTS):
        actions.append(
            DeclareLaunchArgument(
                f"robot_{index + 1}_name",
                default_value=f"itrl_rov_{index + 1}",
                description=f"ROS/PX4 namespace of robot {index + 1}.",
            )
        )
        actions.append(
            DeclareLaunchArgument(
                f"rov_{index + 1}_pose",
                default_value=DEFAULT_POSES[index],
                description=(
                    f"Gazebo ENU pose of robot {index + 1}: "
                    "x,y,z,roll,pitch,yaw."
                ),
            )
        )

    actions.append(OpaqueFunction(function=_launch_setup))
    return LaunchDescription(actions)
