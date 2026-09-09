# Simulated MoCap

The SITL MoCap adapter intentionally reproduces the **raw laboratory MoCap
interface**, rather than publishing controller-ready states directly.

```text
PX4 VehicleOdometry
        |
        v
trusted state adapter (core NWU / FLU)
        |
        v
simulated raw MoCap (NED / FRD)
        |
        v
mocap_odom_ekf
        |
        v
controller state (core NWU / FLU)
```

This makes SITL exercise the same frame-conversion boundary as the wet
experiment.

## Measurement modes

The simulated pose stream has two modes:

```text
measurement_mode:=ideal
measurement_mode:=intermittent
```

`ideal` publishes every valid pose.

`intermittent` periodically suppresses only `/mocap/<robot>/pose`. By default:

```text
dropout_start_sec    = 5.0
dropout_period_sec   = 10.0
dropout_duration_sec = 2.0
```

Thus, after a 5 s initialization period, the pose disappears for 2 s every
10 s. The pseudo-IMU continues throughout the dropout.

The two/three-robot SITL wrappers expose these as:

```text
mocap_measurement_mode
mocap_dropout_start_sec
mocap_dropout_period_sec
mocap_dropout_duration_sec
```

They also expose:

```text
use_imu_gyro:=true|false
```

so the same intermittent pose pattern can be tested with and without gyro
assistance.

## Standalone adapter example

```bash
ros2 launch formation_control_ros simulated_mocap.launch.py \
  robots:=splash,bubble \
  measurement_mode:=intermittent \
  dropout_start_sec:=5.0 \
  dropout_period_sec:=10.0 \
  dropout_duration_sec:=2.0
```

Inspect:

```bash
ros2 topic hz /mocap/splash/pose
ros2 topic hz /mocap/splash/imu
```

The IMU remains continuous while the pose exhibits the requested gaps.
