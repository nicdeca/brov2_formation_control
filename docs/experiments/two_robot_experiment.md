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


## Recording, export, and plotting

Start the recorder **before arming**. For the current physical two-robot
experiment:

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

The recorder explicitly includes, for every robot,

```text
/<robot>/fmu/out/vehicle_odometry
/mocap/<robot>/pose
/mocap/<robot>/pose_core
/mocap/<robot>/odom_ekf
```

in addition to controller diagnostics and the configured IMU topic. Thus PX4,
the EKF, and transformed raw MoCap are available for offline comparison even
when the controller itself uses `/mocap/<robot>/odom_ekf`.

The recorder automatically exports and plots when recording closes. To repeat
post-processing manually, first identify the latest run:

```bash
RUN=$(ls -dt outputs/experiments/* | head -n 1)
```

### Export the complete split run

Recommended:

```bash
python3 scripts/export_experiment.py "$RUN"
```

This exports both initialization and mission when available.

To export only the mission:

```bash
python3 scripts/export_formation_bag.py "$RUN" --phase mission
```

The mission output is:

```text
$RUN/mission/formation_history.npz
```

and contains the controller state plus the parallel comparison arrays:

```text
px4_positions
px4_quaternions
px4_linear_velocity_body
px4_angular_velocity_body

ekf_positions
ekf_quaternions
ekf_linear_velocity_body
ekf_angular_velocity_body

mocap_positions
mocap_quaternions
mocap_linear_velocity_body_fd
mocap_angular_velocity_body_fd
```

### Plot everything

The recommended high-level command is:

```bash
python3 scripts/plot_experiment.py "$RUN" \
  --mission-preset paper \
  --format pdf
```

It produces the normal initialization/formation figures **and** invokes the
PX4/EKF/raw-MoCap estimator comparison automatically.

For mission plots only, the following also generates estimator-comparison
figures by default:

```bash
uv run python scripts/plot_formation_experiment.py \
  "$RUN/mission/formation_history.npz" \
  --paper \
  --paper-quality \
  --save
```

To suppress only the estimator comparison, add:

```text
--no-estimator-comparison
```

To generate only the state-estimator figures:

```bash
uv run python scripts/plot_state_estimator_comparison.py \
  "$RUN/mission/formation_history.npz" \
  --save
```

The estimator figures are saved in:

```text
$RUN/mission/plots/
```

with names such as:

```text
estimator_state_comparison_splash.pdf
estimator_disagreement_splash.pdf
estimator_state_comparison_bubble.pdf
estimator_disagreement_bubble.pdf
```

### Quick check that comparison data were exported

If no estimator plots are generated, first verify that the bag actually
contains the three state streams:

```bash
ros2 bag info "$RUN/mission/bag" | grep -E \
  'vehicle_odometry|odom_ekf|pose_core'
```

Then run the dedicated estimator plotter directly. It reports whether PX4 and
EKF samples overlap and whether transformed raw MoCap is available.
