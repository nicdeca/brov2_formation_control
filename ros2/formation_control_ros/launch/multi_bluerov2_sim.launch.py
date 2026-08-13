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

# Gazebo ENU poses: x,y,z,roll,pitch,yaw.
#
# The first three defaults put both followers on the camera-visible side of
# robot 1.  They are deliberately close, but not identical, to the desired
# experiment triangle so the INITIALIZE phase still performs a real maneuver.
DEFAULT_POSES = [
    "-1.15,-2.20,-95.70,0,0,0",  # itrl_rov_1
    "-2.65,-1.55,-95.70,0,0,0",  # itrl_rov_2
    "-3.25,-2.85,-95.70,0,0,0",  # itrl_rov_3
    "-3.80,-1.20,-95.70,0,0,0",
    "-3.80,-3.50,-95.70,0,0,0",
    "-4.40,-2.35,-95.70,0,0,0",
]


def _launch_setup(context, *args, **kwargs):
    px4_dir = Path(
        LaunchConfiguration("px4_dir").perform(context)
    ).expanduser().resolve()
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

    robot_count = int(
        LaunchConfiguration("robot_count").perform(context)
    )
    if not 1 <= robot_count <= MAX_ROBOTS:
        raise RuntimeError(
            f"robot_count must lie in [1, {MAX_ROBOTS}], got {robot_count}."
        )

    world = LaunchConfiguration("world").perform(context)
    spawn_delay = float(
        LaunchConfiguration("spawn_delay").perform(context)
    )
    if spawn_delay < 0.0:
        raise RuntimeError("spawn_delay must be nonnegative.")

    actions = []
    for index in range(robot_count):
        instance = index
        namespace = f"itrl_rov_{index + 1}"
        pose = LaunchConfiguration(
            f"rov_{index + 1}_pose"
        ).perform(context)
        delay = spawn_delay * index

        command = (
            f"sleep {delay} && "
            f"exec {shlex.quote(str(px4_binary))} -i {instance}"
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
                f"rov_{index + 1}_pose",
                default_value=DEFAULT_POSES[index],
                description=(
                    f"Gazebo ENU pose of itrl_rov_{index + 1}: "
                    "x,y,z,roll,pitch,yaw."
                ),
            )
        )

    actions.append(OpaqueFunction(function=_launch_setup))
    return LaunchDescription(actions)
