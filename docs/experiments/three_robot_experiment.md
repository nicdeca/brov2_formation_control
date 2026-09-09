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
