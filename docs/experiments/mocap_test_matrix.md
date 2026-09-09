# MoCap/EKF test commands

These commands assume the simulator and Micro XRCE-DDS agent are already
running.

## Two robots: splash + bubble

### A. Ideal MoCap + IMU

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

### B. Intermittent MoCap + IMU

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

### C. Intermittent MoCap, no IMU

Repeat B with:

```text
use_imu_gyro:=false
```

### D. Active two-robot test

After A--C are satisfactory, repeat the selected launch with:

```text
dry_run:=false
```

Then run:

```bash
python3 scripts/run_two_robot_experiment.py \
  --leader splash \
  --profile cautious
```

## Three robots: splash + glub + bubble

### A. Ideal MoCap + IMU

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

### B. Intermittent MoCap + IMU

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

### C. Intermittent MoCap, no IMU

Repeat B with:

```text
use_imu_gyro:=false
```

### D. Active three-robot test

After A--C are satisfactory, repeat the selected launch with:

```text
dry_run:=false
```

Then run:

```bash
python3 scripts/run_three_robot_experiment.py \
  --leader splash \
  --profile cautious
```

## What should happen during a dropout

```text
/mocap/<robot>/pose       -> absent for the requested interval
/mocap/<robot>/imu        -> continuous in SITL
/mocap/<robot>/odom_ekf   -> continuous at the EKF publish rate
controller                -> continues receiving fresh odometry
```

With `use_imu_gyro:=false`, `/mocap/<robot>/imu` may still exist in SITL but
the EKF deliberately ignores it.
