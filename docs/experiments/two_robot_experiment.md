# Two-robot experiment

## Physical assignment and topology

Current wet-test assignment:

```text
leader   = splash
follower = bubble

bubble -> splash
```

The controller relative vector is `p_parent - p_follower`.

The default initialization is:

```text
splash = [2.675, 0.050, -1.45]
bubble = [4.475, 0.750, -1.45]
```

so

```text
p_splash - p_bubble = [-1.80, -0.70, 0.00].
```

The depth is exposed through:

```text
initialization_z:=-1.45
```

The maintained wet mission keeps leader vertical velocity at zero and does not
command `pair_high`; routine two-robot wet tests therefore remain around the
initialization depth.

## Real MoCap + EKF

Dry run:

```bash
ros2 launch formation_control_ros \
  two_robot_experiment_with_ekf.launch.py \
  leader:=splash \
  follower:=bubble \
  dry_run:=true \
  leader_reference_mode:=velocity \
  workspace_barrier_enabled:=true \
  workspace_adaptive:=true
```

Active run: change only `dry_run:=false`.

The wrapper defaults to:

```text
raw MoCap       = NED / FRD
EKF output      = core NWU / FLU
use_imu_gyro    = true, with automatic fallback if unavailable
max_coast_sec   = 0.0
initialization_z = -1.45
```

## SITL + EKF: ideal MoCap

```bash
ros2 launch formation_control_ros \
  two_robot_experiment_with_ekf_sitl.launch.py \
  leader:=splash \
  follower:=bubble \
  dry_run:=true \
  leader_reference_mode:=velocity \
  mocap_measurement_mode:=ideal \
  use_imu_gyro:=true
```

## SITL + EKF: intermittent MoCap, gyro enabled

```bash
ros2 launch formation_control_ros \
  two_robot_experiment_with_ekf_sitl.launch.py \
  leader:=splash \
  follower:=bubble \
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

In both intermittent tests, `/mocap/<robot>/pose` should disappear during the
dropout windows while `/mocap/<robot>/odom_ekf` continues.

## Mission runner

```bash
python3 scripts/run_two_robot_experiment.py \
  --leader splash \
  --profile cautious
```

After validation:

```bash
python3 scripts/run_two_robot_experiment.py \
  --leader splash \
  --profile full
```

The `challenging` profile is intended only after the normal wet profiles have
been validated.
