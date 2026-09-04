# Three-robot formation experiment

## Topology

Physical assignment:

```text
             splash
             /    \
          glub    bubble
```

Directed follower-to-parent edges:

```text
glub   -> splash
bubble -> splash
```

Current pool-aligned initialization positions:

```text
splash = [2.675,  0.050, -1.450]
glub   = [4.475, -0.650, -1.450]
bubble = [4.475,  0.750, -1.450]
```

Nominal parent-minus-follower vectors:

```text
glub   -> splash: [-1.800,  0.700, 0.000]
bubble -> splash: [-1.800, -0.700, 0.000]
```


The common initialization depth is exposed as

```text
initialization_z:=-1.45
```

and is forwarded to the phase manager and all three controller nodes.

## Four supported state-estimation test modes

| environment | state source | controller launch |
| --- | --- | --- |
| SITL | PX4 | `three_robot_experiment.launch.py` |
| SITL | simulated MoCap + estimator | `three_robot_experiment_with_ekf_sitl.launch.py` |
| real robots | PX4 | `three_robot_experiment.launch.py` |
| real robots | real MoCap + estimator | `three_robot_experiment_with_ekf.launch.py` |

Always start with `dry_run:=true` when validating a new state-estimator/frame
configuration, then repeat with `dry_run:=false` for the initialization
maneuver.

---

## 1. SITL using PX4 state directly

Start Gazebo/PX4 with physical names:

```bash
ros2 launch formation_control_ros multi_bluerov2_sim.launch.py \
  robot_count:=3 \
  robot_1_name:=splash \
  robot_2_name:=glub \
  robot_3_name:=bubble \
  px4_dir:=/home/nicola/Gits/KTH-PX4/PX4-Autopilot
```

Start DDS:

```bash
micro-xrce-dds-agent udp4 -p 8888
```

Dry-run controller:

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

Change to `dry_run:=false` after verification.

---

## 2. SITL using simulated MoCap + estimator

With the same simulator and DDS agent running:

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

For each robot this creates a simulated raw MoCap stream with the same NED
world / FRD body convention used by the laboratory system:

```text
PX4 VehicleOdometry
 -> simulated MoCap pose + FLU gyro
 -> mocap_odom_ekf
 -> /mocap/<robot>/pose_core
 -> /mocap/<robot>/odom_ekf
 -> controller
```

The wrapper explicitly selects `gazebo` dynamics for all three vehicles.

Verify at least one robot from each state stage:

```bash
ros2 topic echo /mocap/splash/pose --once
ros2 topic echo /mocap/glub/imu --once
ros2 topic echo /mocap/bubble/pose_core --once
ros2 topic echo /mocap/glub/odom_ekf --once
```

Healthy operation produces no periodic MoCap status spam; only abnormal
conditions are logged.

Then repeat with `dry_run:=false`.

---

## 3. Real robots using PX4 state directly

```bash
ros2 launch formation_control_ros three_robot_experiment.launch.py \
  dry_run:=true \
  leader:=splash \
  follower_left:=glub \
  follower_right:=bubble \
  leader_reference_mode:=velocity \
  state_source:=px4 \
  workspace_barrier_enabled:=true \
  workspace_adaptive:=true
```

The physical-name `auto` dynamics mapping is:

```text
splash -> heavy_tube
glub   -> heavy_tube
bubble -> standard
```

After the dry-run checks, repeat with `dry_run:=false`.

---

## 4. Real robots using real MoCap + estimator

Required real MoCap inputs:

```text
/mocap/splash/pose
/mocap/glub/pose
/mocap/bubble/pose
```

Launch:

```bash
ros2 launch formation_control_ros \
  three_robot_experiment_with_ekf.launch.py \
  dry_run:=true \
  leader:=splash \
  follower_left:=glub \
  follower_right:=bubble \
  leader_reference_mode:=velocity \
  workspace_barrier_enabled:=true \
  workspace_adaptive:=true
```

The estimator output is always core-NWU / body-FLU.  The current hardware
wrapper defaults to `use_imu_gyro:=false` because the MAVROS gyro stream is not
available in the pool stack; angular velocity therefore falls back to MoCap
attitude finite differences.  SITL still enables its pseudo-gyro explicitly.

The laboratory raw MoCap convention is now built into the default hardware
path:

```text
raw MoCap = NED world / FRD body
EKF output = core NWU world / FLU body
```

No frame arguments are required for the normal pool experiment.  Generic
calibration arguments remain available for future recalibration.

Before actuation, verify all three `/pose_core` and `/odom_ekf` streams.

---

## Recorder

### PX4

```bash
scripts/record_formation_experiment.sh \
  --name three_robot_experiment_px4 \
  --robots splash,glub,bubble \
  --edge glub:splash \
  --edge bubble:splash \
  --state-source px4
```

### MoCap estimator

```bash
scripts/record_formation_experiment.sh \
  --name three_robot_experiment_ekf \
  --robots splash,glub,bubble \
  --edge glub:splash \
  --edge bubble:splash \
  --state-source nav_msgs \
  --state-topic-template '/mocap/{robot}/odom_ekf' \
  --mocap-world-frame core_nwu \
  --odom-twist-frame body \
  --imu-topic-template '/{robot}/mavros/imu/data'
```

For EKF SITL, use:

```text
--imu-topic-template '/mocap/{robot}/imu'
```

---

## Mission runner

First:

```bash
python scripts/run_three_robot_experiment.py \
  --leader splash \
  --profile cautious
```

After validation:

```bash
python scripts/run_three_robot_experiment.py \
  --leader splash \
  --profile full
```

The runner publishes:

```text
WAITING -> RUNNING -> COMPLETE
```

or `ABORTED` on interruption/error.

## Recommended real-world procedure

1. Reproduce the exact names/topology in SITL.
2. Test SITL with PX4 state.
3. Test SITL through the simulated-MoCap estimator path.
4. Start physical PX4, MoCap, and IMU streams.
5. Run the selected physical state source with `dry_run:=true`.
6. Verify `/pose_core`, `/odom_ekf`, gyro signs, workspace, and both sensing edges.
7. Re-launch with `dry_run:=false`.
8. Record initialization and mission.
9. Run `cautious`.
10. Inspect three-way estimator plots before running `full`.
