# SITL simulation setup

## 1. Pool-aligned Gazebo world

The current Marinarium worlds are re-anchored so that PX4 local NED matches the
intended real-pool origin/orientation. The simulated tank remains approximately
`7.4 x 4.55 x 2.55 m`; it was not scaled to the reported real-pool dimensions.

Read `pool_coordinate_frame.md` before changing any absolute positions.

## 2. Terminal 1 — Gazebo + PX4 SITL

The canonical simulator launch is `multi_bluerov2_sim.launch.py`.

Two robots with generic SITL names:

```bash
cd ~/discower_ws
source /opt/ros/jazzy/setup.bash
source ~/discower_ws/install/setup.bash

ros2 launch formation_control_ros multi_bluerov2_sim.launch.py \
  robot_count:=2 \
  px4_dir:=/home/nicola/Gits/KTH-PX4/PX4-Autopilot
```

Two robots using hardware-like names:

```bash
ros2 launch formation_control_ros multi_bluerov2_sim.launch.py \
  robot_count:=2 \
  robot_1_name:=splash \
  robot_2_name:=glub \
  px4_dir:=/home/nicola/Gits/KTH-PX4/PX4-Autopilot
```

Five robots:

```bash
ros2 launch formation_control_ros multi_bluerov2_sim.launch.py \
  robot_count:=5 \
  px4_dir:=/home/nicola/Gits/KTH-PX4/PX4-Autopilot
```

`robot_1_name` ... `robot_6_name` control the PX4/ROS namespaces. The simulator
also accepts `rov_1_pose` ... `rov_6_pose`, which are internal Gazebo ENU poses.
Do not reinterpret those pose arguments as controller-core NWU coordinates.

## 3. Terminal 2 — Micro XRCE-DDS agent

```bash
micro-xrce-dds-agent udp4 -p 8888
```

## 4. QGroundControl and namespace check

Open one QGroundControl instance and verify every intended vehicle appears.
Before arming, check that the ROS namespaces exactly match the simulator names:

```bash
ros2 topic list | grep '/fmu/out/vehicle_odometry'
```

A common failure is to spawn `splash,bubble` while launching controllers for
`splash,glub`; the second controller then has no PX4/offboard stream.

Useful rate checks:

```bash
ros2 topic hz /splash/fmu/out/vehicle_odometry
ros2 topic hz /glub/fmu/out/vehicle_odometry
```

If using generic names, substitute `itrl_rov_1`, `itrl_rov_2`, etc.

## 5. Verify the new pool coordinates

Before launching control, inspect one raw PX4 sample:

```bash
ros2 topic echo /splash/fmu/out/vehicle_odometry --once
```

For the first default spawn, the raw NED position should be of order

```text
x ~ 2.7 m, y ~ -1.0 m, z ~ 1.3 m
```

rather than the old `z ~ -95 m` world coordinates.

## 6. Two-robot controller launch in SITL

`two_robot_experiment.launch.py` defaults robot dynamics to `auto`. `auto` is
for known physical names and must **not** be allowed to select real-hardware
models in SITL. Pass the Gazebo configuration explicitly.

With `splash` / `glub` namespaces:

```bash
ros2 launch formation_control_ros two_robot_experiment.launch.py \
  dry_run:=false \
  leader_reference_mode:=velocity \
  leader:=splash \
  follower:=glub \
  leader_robot_configuration:=gazebo \
  follower_robot_configuration:=gazebo \
  workspace_barrier_enabled:=true \
  workspace_adaptive:=true
```

With generic namespaces, likewise pass both `*_robot_configuration:=gazebo`.

## 7. Five-robot controller launch

```bash
ros2 launch formation_control_ros five_robot_tree_experiment.launch.py \
  dry_run:=false \
  leader_reference_mode:=velocity \
  workspace_barrier_enabled:=true \
  workspace_adaptive:=true
```

The five-robot launch already fixes all controller dynamics to `gazebo`.

Run the mission with:

```bash
python scripts/run_five_robot_tree_experiment.py \
  --leader itrl_rov_1 \
  --profile full
```

The current five-robot runner uses wall time and does **not** expose a
`--gazebo-timer` argument. The launch-level `gazebo_timer` option is therefore
not part of the normal scripted five-robot workflow.

## 8. Adaptive two-robot SITL mission

Use the dedicated launch rather than changing the normal two-robot experiment:

```bash
ros2 launch formation_control_ros two_robot_adaptive_mission.launch.py \
  dry_run:=false
```

Then:

```bash
python scripts/run_two_robot_adaptive_mission.py --leader itrl_rov_1
```

For simulation-time scheduling, set `gazebo_timer:=true` in the launch and pass
`--gazebo-timer` to this adaptive mission runner.
