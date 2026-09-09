# Three-robot experiment

## Physical assignment and topology

```text
          splash
          /    \
       glub    bubble
```

Directed sensing edges:

```text
glub   -> splash
bubble -> splash
```

Default pool-centered core-NWU initialization:

```text
splash = [2.500,  0.000, -1.45]
glub   = [4.300, -0.700, -1.45]
bubble = [4.300,  0.700, -1.45]
```

The nominal triangle centroid is therefore approximately
`[3.70, 0.00]`, matching the center of the configured x/y workspace.

The leader initialization is exposed as:

```text
initialization_x:=2.500
initialization_y:=0.000
initialization_z:=-1.45
```

Both follower initialization targets are derived automatically from the
parent-minus-follower vectors, so translating the leader initialization moves
the whole triangle consistently.

## Pool-centered wet mission

The standard cautious and full missions stay at `z=-1.45 m` and deliberately
keep the formation near the center of the pool in x and y.

The maintained formation library uses:

```text
triangle_nominal:  [-1.80, +/-0.70, 0.00]
triangle_wide:     [-2.20, +/-0.95, 0.00]
triangle_compact:  [-1.40, +/-0.40, 0.00]
```

The full mission uses symmetric leader excursions:

```text
x: 0 -> +0.50 -> -0.50 -> 0 m relative to initialization
y: 0 -> -0.25 -> +0.25 -> 0 m relative to initialization
```

The wide triangle is commanded only while the leader is close to `y=0`.
Consequently the planned follower references remain far from both y walls.
`triangle_high` remains available for deliberate vertical tests but is not
used by the standard wet missions.

## Real MoCap + EKF

Dry run:

```bash
ros2 launch formation_control_ros \
  three_robot_experiment_with_ekf.launch.py \
  leader:=splash \
  follower_left:=glub \
  follower_right:=bubble \
  dry_run:=true \
  leader_reference_mode:=velocity \
  workspace_barrier_enabled:=true \
  workspace_adaptive:=true
```

Active run: change only `dry_run:=false`.

Normal defaults are raw NED/FRD MoCap, core-NWU/FLU EKF output,
opportunistic gyro use with automatic fallback, unlimited EKF coasting, and
`initialization_z=-1.45`.

## SITL + EKF: ideal MoCap

```bash
ros2 launch formation_control_ros \
  three_robot_experiment_with_ekf_sitl.launch.py \
  leader:=splash \
  follower_left:=glub \
  follower_right:=bubble \
  dry_run:=true \
  leader_reference_mode:=velocity \
  mocap_measurement_mode:=ideal \
  use_imu_gyro:=true
```

## SITL + EKF: intermittent MoCap, gyro enabled

```bash
ros2 launch formation_control_ros \
  three_robot_experiment_with_ekf_sitl.launch.py \
  leader:=splash \
  follower_left:=glub \
  follower_right:=bubble \
  dry_run:=true \
  leader_reference_mode:=velocity \
  mocap_measurement_mode:=intermittent \
  mocap_dropout_start_sec:=5.0 \
  mocap_dropout_period_sec:=10.0 \
  mocap_dropout_duration_sec:=2.0 \
  use_imu_gyro:=true
```

## SITL + EKF: intermittent MoCap, no gyro

Use the same command with:

```text
use_imu_gyro:=false
```

## Mission runner

```bash
python3 scripts/run_three_robot_experiment.py \
  --leader splash \
  --profile cautious
```

After validation:

```bash
python3 scripts/run_three_robot_experiment.py \
  --leader splash \
  --profile full
```

## Recording, export, and plotting

The normal workflow uses one recorder command for the run, followed by one
export command and one plot command. The latter two automatically process
**every phase that is present**.

### 1. Record

Start the recorder before arming:

```bash
scripts/record_formation_experiment.sh \
  --name three_robot_experiment_ekf \
  --robots splash,glub,bubble \
  --edge glub:splash \
  --edge bubble:splash
```

For the maintained MoCap/EKF experiments, the recorder defaults are already:

```text
state_source              = nav_msgs
state_topic_template      = /mocap/{robot}/odom_ekf
mocap_world_frame         = core_nwu
odom_twist_frame          = body
ekf_topic_template        = /mocap/{robot}/odom_ekf
mocap_pose_topic_template = /mocap/{robot}/pose
core_pose_topic_template  = /mocap/{robot}/pose_core
imu_topic_template        = /{robot}/mavros/imu/data
```

The recorder also always retains the PX4 state, raw/transformed MoCap, EKF,
MAVROS gyro, and simulated-MoCap gyro streams when they are present. Therefore
none of these MoCap-related arguments need to be repeated in a normal wet
experiment. For a deliberate direct-PX4 controller run, override only:

```text
--state-source px4
```


The recorder may create:

```text
<RUN>/initialization/bag
<RUN>/mission/bag
```

depending on how far the experiment progressed. For every listed robot it also
records the parallel state-estimator streams:

```text
/<robot>/fmu/out/vehicle_odometry
/mocap/<robot>/pose
/mocap/<robot>/pose_core
/mocap/<robot>/odom_ekf
```

### 2. Export everything that exists

Select the latest run:

```bash
RUN=$(ls -dt outputs/experiments/* | head -n 1)
```

Then use the run-level exporter:

```bash
python3 scripts/export_experiment.py "$RUN"
```

This single command behaves as follows:

```text
initialization only  -> exports initialization
mission only         -> exports mission
both                 -> exports both
neither              -> reports an error
```

When present, the exported histories are:

```text
$RUN/initialization/initialization_history.npz
$RUN/mission/formation_history.npz
```

There is no need to call the phase-specific exporters during the normal
workflow.

### 3. Plot everything that was exported

Use the run-level plotter:

```bash
python3 scripts/plot_experiment.py "$RUN" \
  --mission-preset paper \
  --format pdf
```

Again, this is a single command:

```text
initialization history only  -> plots initialization
mission history only         -> plots mission
both                         -> plots both
neither                      -> reports an error
```

For each available phase, the plotter also attempts the PX4 / EKF /
transformed-raw-MoCap comparison. Therefore, when both phases are present, the
estimator is compared separately during initialization and during the mission.

The output folders are:

```text
$RUN/initialization/plots/
$RUN/mission/plots/
```

whenever the corresponding phase exists.

The lower-level scripts

```text
export_initialization_bag.py
export_formation_bag.py
plot_initialization_experiment.py
plot_formation_experiment.py
plot_state_estimator_comparison.py
```

remain available for debugging, but are not needed for a normal run.
