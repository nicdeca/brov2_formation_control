# Two-robot experiment

## Topology

Usual physical assignment:

```text
leader   = splash
follower = bubble

bubble -> splash
```

Known laboratory dynamics mapping:

```text
glub    -> heavy_tube
splash  -> heavy_tube
bubble  -> standard
```

The desired relative vector uses the controller convention

```text
p_parent - p_follower
```

For the current two-robot `follower:=bubble` / `leader:=splash` launch, the initialization geometry is

```text
splash   = [2.675, 0.050, -1.450]
follower = [4.475, 0.750, -1.450]

p_splash - p_follower = [-1.800, -0.700, 0.000]
```

The common depth is a launch argument:

```text
initialization_z:=-1.45
```

This is the wet-test default because MoCap tracking degrades near the surface.
The controller nodes receive this value through their `initialization_position`
parameter; no controller source edit is required.

## Four supported state-estimation test modes

The same controller can be tested in four combinations:

| environment | state source | controller launch |
| --- | --- | --- |
| SITL | PX4 `VehicleOdometry` | `two_robot_experiment.launch.py` |
| SITL | simulated MoCap + in-repository estimator | `two_robot_experiment_with_ekf_sitl.launch.py` |
| real robots | PX4 `VehicleOdometry` | `two_robot_experiment.launch.py` |
| real robots | real MoCap + in-repository estimator | `two_robot_experiment_with_ekf.launch.py` |

Use `dry_run:=true` first for state/frame checks. The robots do **not** execute
the initialization maneuver in dry-run mode. Repeat with `dry_run:=false` when
the state pipeline is verified.

---

## 1. SITL using PX4 state directly

### Start Gazebo + PX4

Use the same names as the physical robots:

```bash
ros2 launch formation_control_ros multi_bluerov2_sim.launch.py \
  robot_count:=2 \
  robot_1_name:=splash \
  robot_2_name:=bubble \
  px4_dir:=/home/nicola/Gits/KTH-PX4/PX4-Autopilot
```

Start the Micro XRCE-DDS agent:

```bash
micro-xrce-dds-agent udp4 -p 8888
```

### Dry-run controller

Because hardware-like names would make `auto` choose the physical dynamics
models, explicitly select `gazebo`:

```bash
ros2 launch formation_control_ros two_robot_experiment.launch.py \
  dry_run:=true \
  leader:=splash \
  follower:=bubble \
  leader_robot_configuration:=gazebo \
  follower_robot_configuration:=gazebo \
  leader_reference_mode:=velocity \
  state_source:=px4 \
  workspace_barrier_enabled:=true \
  workspace_adaptive:=true
```

For active initialization/control, change only:

```text
dry_run:=false
```

The controller consumes:

```text
/splash/fmu/out/vehicle_odometry
/bubble/fmu/out/vehicle_odometry
```

and the ROS boundary converts PX4 NED/FRD to core NWU/FLU.

---

## 2. SITL using the MoCap estimator

Keep the same Gazebo/PX4 and DDS processes running, then launch:

```bash
ros2 launch formation_control_ros \
  two_robot_experiment_with_ekf_sitl.launch.py \
  leader:=splash \
  follower:=bubble \
  dry_run:=true \
  leader_reference_mode:=velocity \
  workspace_barrier_enabled:=true \
  workspace_adaptive:=true
```

This starts the complete simulated sensor/estimator path.  The simulated raw
MoCap now deliberately matches the laboratory convention (NED world / FRD
body), so SITL exercises exactly the same frame conversion as hardware:

```text
PX4 VehicleOdometry
        |
        v
simulated_mocap
   |             |
   v             v
/mocap/.../pose  /mocap/.../imu
        \         /
         v       v
       mocap_odom_ekf
             |
             +--> /mocap/<robot>/pose_core
             |
             +--> /mocap/<robot>/odom_ekf
                         |
                         v
                    controller
```

The SITL wrapper forces both vehicle models to `gazebo`.

Verify once:

```bash
ros2 topic echo /mocap/bubble/pose --once
ros2 topic echo /mocap/bubble/imu --once
ros2 topic echo /mocap/bubble/pose_core --once
ros2 topic echo /mocap/bubble/odom_ekf --once
```

and repeat for `splash`.

Healthy MoCap/estimator operation is intentionally quiet. Warnings are printed
only if state/gyro data stop arriving, measurements are rejected, or gyro
fusion falls back.

Then repeat the launch with:

```text
dry_run:=false
```

---

## 3. Real robots using PX4 state directly

Start the physical PX4/network stack, then:

```bash
ros2 launch formation_control_ros two_robot_experiment.launch.py \
  dry_run:=true \
  leader:=splash \
  follower:=bubble \
  leader_reference_mode:=velocity \
  state_source:=px4 \
  workspace_barrier_enabled:=true \
  workspace_adaptive:=true
```

For physical robot names, the default `auto` dynamics selection gives:

```text
splash -> heavy_tube
glub   -> heavy_tube
```

After checking state signs, initialization targets, Offboard behavior, and
workspace margins, repeat with:

```text
dry_run:=false
```

---

## 4. Real robots using real MoCap + in-repository estimator

The real MoCap system must publish:

```text
/mocap/splash/pose
/mocap/bubble/pose
```

The hardware wrapper starts one estimator per robot and configures the
controllers and phase manager to consume `/mocap/<robot>/odom_ekf`:

```bash
ros2 launch formation_control_ros \
  two_robot_experiment_with_ekf.launch.py \
  dry_run:=true \
  leader:=splash \
  follower:=bubble \
  leader_reference_mode:=velocity \
  workspace_barrier_enabled:=true \
  workspace_adaptive:=true
```

The estimator output is always:

```text
world = core NWU
body  = FLU
twist = body FLU
```

The hardware wrapper currently defaults to `use_imu_gyro:=false` because the
MAVROS gyro topic is not available in the current pool stack.  The estimator
therefore uses MoCap attitude finite differences for angular velocity.  If a
body-FLU gyro becomes available later, enable it with `use_imu_gyro:=true` and
set `imu_topic_template` accordingly.

### Stabilized laboratory frame convention

The laboratory raw MoCap convention has been measured and is now the default:

```text
raw MoCap world = NED
raw rigid body  = FRD
```

The estimator converts this internally to

```text
controller world = core NWU
controller body  = FLU
```

Therefore the normal hardware launch no longer needs custom frame arguments.
The equivalent low-level settings are `input_world_frame:=ned` and
`input_body_frame:=frd`.

The generic transform arguments remain available only for future recalibration
or another MoCap system.

Before actuation, verify:

```bash
ros2 topic echo /mocap/bubble/pose_core --once
ros2 topic echo /mocap/bubble/odom_ekf --once
```

`pose_core` is the raw MoCap measurement after only frame/origin calibration;
`odom_ekf` is the filtered controller-facing state.

---

## Recorder

### PX4-controlled run

```bash
scripts/record_formation_experiment.sh \
  --name two_robot_experiment_px4 \
  --robots splash,bubble \
  --edge bubble:splash \
  --state-source px4
```

### MoCap-estimator-controlled run

```bash
scripts/record_formation_experiment.sh \
  --name two_robot_experiment_ekf \
  --robots splash,bubble \
  --edge bubble:splash \
  --state-source nav_msgs \
  --state-topic-template '/mocap/{robot}/odom_ekf' \
  --mocap-world-frame core_nwu \
  --odom-twist-frame body \
  --imu-topic-template '/{robot}/mavros/imu/data'
```

For EKF SITL, use instead:

```text
--imu-topic-template '/mocap/{robot}/imu'
```

The recorder retains PX4, estimator output, raw MoCap, transformed raw MoCap,
and gyro streams when available.

---

## Mission runner

```bash
python scripts/run_two_robot_experiment.py \
  --leader splash \
  --profile cautious
```

Use `--profile full` only after `cautious` is validated.

Expected experiment phase:

```text
INITIALIZE -> settle dwell -> FORMATION
```

The phase manager uses the same selected state source as the controllers.

---

## After the run

```bash
RUN=$(ls -dt outputs/experiments/*_two_robot_experiment* | head -n 1)

python scripts/export_formation_bag.py "$RUN"
uv run python scripts/plot_experiment.py "$RUN"
```

For an EKF recording, the estimator-comparison figures show:

```text
PX4
MoCap estimator
raw transformed MoCap
```

including raw-MoCap finite-difference velocity diagnostics.
