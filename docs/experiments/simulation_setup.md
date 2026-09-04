# SITL simulation setup

This page documents the common Gazebo/PX4 setup and the two supported state
paths:

1. direct PX4 `VehicleOdometry`;
2. simulated MoCap pose + gyro passed through the same in-repository estimator
   used in real experiments.

## Common SITL setup

### Terminal 1 — Gazebo + PX4 SITL

Two robots with physical names:

```bash
ros2 launch formation_control_ros multi_bluerov2_sim.launch.py \
  robot_count:=2 \
  robot_1_name:=splash \
  robot_2_name:=glub \
  px4_dir:=/home/nicola/Gits/KTH-PX4/PX4-Autopilot
```

Three robots:

```bash
ros2 launch formation_control_ros multi_bluerov2_sim.launch.py \
  robot_count:=3 \
  robot_1_name:=splash \
  robot_2_name:=glub \
  robot_3_name:=bubble \
  px4_dir:=/home/nicola/Gits/KTH-PX4/PX4-Autopilot
```

Using the physical names is intentional: the same controller, recorder, and
mission commands can then be reused in the pool.

### Terminal 2 — Micro XRCE-DDS agent

```bash
micro-xrce-dds-agent udp4 -p 8888
```

### QGroundControl

Before enabling actuation:

- confirm every intended vehicle appears;
- confirm every vehicle can arm;
- confirm Offboard becomes available;
- verify QGC IDs correspond to the ROS namespaces;
- verify PX4 odometry is alive.

Examples:

```bash
ros2 topic hz /splash/fmu/out/vehicle_odometry
ros2 topic hz /glub/fmu/out/vehicle_odometry
```

---

# SITL mode A — direct PX4 state

## Two robots

```bash
ros2 launch formation_control_ros two_robot_experiment.launch.py \
  dry_run:=true \
  leader:=splash \
  follower:=glub \
  leader_robot_configuration:=gazebo \
  follower_robot_configuration:=gazebo \
  leader_reference_mode:=velocity \
  state_source:=px4 \
  workspace_barrier_enabled:=true \
  workspace_adaptive:=true
```

## Three robots

```bash
ros2 launch formation_control_ros three_robot_experiment.launch.py \
  dry_run:=true \
  leader:=splash \
  follower_left:=glub \
  follower_right:=bubble \
  leader_robot_configuration:=gazebo \
  follower_left_robot_configuration:=gazebo \
  follower_right_robot_configuration:=gazebo \
  leader_reference_mode:=velocity \
  state_source:=px4 \
  workspace_barrier_enabled:=true \
  workspace_adaptive:=true
```

Because the names `splash`, `glub`, and `bubble` normally select physical
dynamics in `auto` mode, the direct-PX4 SITL commands explicitly force
`gazebo`.

After checking state/frame/controller diagnostics, repeat with:

```text
dry_run:=false
```

---

# SITL mode B — simulated MoCap + in-repository estimator

This mode validates the same estimator/controller interface used in the pool.

The simulated adapter creates:

```text
/mocap/<robot>/pose   # raw NED world / FRD body, matching the lab MoCap
/mocap/<robot>/imu    # pseudo gyro in FRD for SITL frame-path validation
```

from PX4 SITL state. The estimator applies the same NED/FRD -> NWU/FLU
conversion used in the pool and then publishes:

```text
/mocap/<robot>/pose_core
/mocap/<robot>/odom_ekf
```

with the fixed output contract:

```text
world = core NWU
body  = FLU
```

## Two robots

```bash
ros2 launch formation_control_ros \
  two_robot_experiment_with_ekf_sitl.launch.py \
  leader:=splash \
  follower:=glub \
  dry_run:=true \
  leader_reference_mode:=velocity \
  workspace_barrier_enabled:=true \
  workspace_adaptive:=true
```

## Three robots

```bash
ros2 launch formation_control_ros \
  three_robot_experiment_with_ekf_sitl.launch.py \
  leader:=splash \
  follower_left:=glub \
  follower_right:=bubble \
  dry_run:=true \
  leader_reference_mode:=velocity \
  workspace_barrier_enabled:=true \
  workspace_adaptive:=true
```

The EKF-SITL wrappers force `gazebo` dynamics automatically.

### Verify the estimator path

For example:

```bash
ros2 topic echo /mocap/glub/pose --once
ros2 topic echo /mocap/glub/imu --once
ros2 topic echo /mocap/glub/pose_core --once
ros2 topic echo /mocap/glub/odom_ekf --once
```

Healthy simulated MoCap/EKF operation is intentionally quiet after startup.
Warnings are printed only if a stream stalls, a measurement is rejected, or
gyro fusion falls back.

After the dry-run check, repeat with:

```text
dry_run:=false
```

---

# What SITL with the estimator does and does not test

The simulated MoCap pose and gyro are generated from PX4 SITL
`VehicleOdometry`, but are intentionally republished in the laboratory raw
NED/FRD convention before entering the estimator. Therefore this mode is useful
for testing:

- topic and namespace wiring;
- PX4 NED/FRD -> core NWU/FLU conversion;
- MoCap estimator initialization;
- gyro fusion and angular-rate conventions;
- controller/phase-manager use of `nav_msgs/Odometry`;
- recorder/export/three-way comparison plumbing.

It is **not** an independent estimator-performance benchmark because both the
simulated sensor and PX4 comparison originate from the same SITL state.

---

# Recorder examples

## PX4 state

```bash
scripts/record_formation_experiment.sh \
  --name two_robot_sitl_px4 \
  --robots splash,glub \
  --edge glub:splash \
  --state-source px4
```

## MoCap-estimator state in SITL

```bash
scripts/record_formation_experiment.sh \
  --name two_robot_sitl_ekf \
  --robots splash,glub \
  --edge glub:splash \
  --state-source nav_msgs \
  --state-topic-template '/mocap/{robot}/odom_ekf' \
  --mocap-world-frame core_nwu \
  --odom-twist-frame body \
  --imu-topic-template '/mocap/{robot}/imu'
```

The resulting estimator-comparison plots can contain PX4, estimator output,
and transformed raw MoCap on the same axes.
