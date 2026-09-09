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

Default core-NWU initialization:

```text
splash = [2.675,  0.050, -1.45]
glub   = [4.475, -0.650, -1.45]
bubble = [4.475,  0.750, -1.45]
```

The depth is exposed as `initialization_z` and defaults to `-1.45 m`.

## Wet mission stays at depth

The maintained `run_three_robot_experiment.py` does **not** command
`triangle_high`. All formation references used by the cautious and full wet
profiles have zero vertical relative offsets, and all leader velocity commands
use `v_z=0`.

`triangle_high` remains available in the formation library for deliberate
vertical tests, but is not part of the standard wet mission.

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
  --edge bubble:splash \
  --state-source nav_msgs \
  --state-topic-template '/mocap/{robot}/odom_ekf' \
  --mocap-world-frame core_nwu \
  --odom-twist-frame body \
  --imu-topic-template '/{robot}/mavros/imu/data'
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
