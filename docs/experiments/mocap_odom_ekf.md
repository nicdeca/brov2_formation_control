# MoCap EKF: frame contract, dropout behavior, and standalone debugging

## Controller-facing contract

The estimator publishes `/mocap/<robot>/odom_ekf` using one fixed convention:

```text
world frame     = core / pool-aligned NWU
body frame      = FLU
quaternion      = body FLU -> core NWU
Odometry.twist  = body FLU
parent frame    = core_nwu
child frame     = <robot>/base_link_ekf
```

The controllers therefore consume the EKF as:

```text
state_source:=nav_msgs
mocap_world_frame:=core_nwu
odom_twist_frame:=body
```

## Pool/laboratory raw MoCap convention

Wet testing established that the raw laboratory MoCap pose is:

```text
world = NED
body  = FRD
```

The maintained estimator defaults are consequently:

```text
input_world_frame:=ned
input_body_frame:=frd
```

No extra frame parameters are required in the normal pool launch.

With
`S = diag(1,-1,-1)`, the zero-origin-offset conversion is

```text
p_NWU = S p_NED
R_NWU<-FLU = S R_NED<-FRD S
```

`world_to_core_translation` remains available if the MoCap origin must later
be shifted relative to the controller pool origin.

The transformed raw measurement is republished on

```text
/mocap/<robot>/pose_core
```

before estimator gating, so frame conversion and filtering can be inspected
separately.

## IMU behavior

`use_imu_gyro:=true` is the maintained default. This is opportunistic rather
than mandatory:

1. if fresh gyro samples are available, the EKF uses them;
2. if the gyro topic is absent or stale, it automatically uses the
   finite-difference angular rate reconstructed from accepted MoCap attitudes;
3. if MoCap is also absent long enough for that finite-difference rate to
   expire, angular rate falls back to zero.

Only the **gyroscope** is fused by this estimator; accelerometer measurements
are not currently used.

For explicit no-IMU tests:

```text
use_imu_gyro:=false
```

## MoCap dropout behavior

Once initialized, the EKF continues predicting and publishing
`/mocap/<robot>/odom_ekf` during MoCap outages.

Maintained defaults:

```text
publish_rate_hz                       = 80.0
max_coast_sec                         = 0.0
mocap_angular_velocity_timeout_sec    = 0.35
reacquire_after_sec                   = 0.50
coast_warning_sec                     = 0.50
```

`max_coast_sec=0.0` means unlimited estimator coasting. The controller's own
freshness check is intentionally unchanged: it continues because the EKF keeps
publishing fresh predicted odometry.

When MoCap returns after a longer gap, the first valid measurement re-anchors
position and attitude immediately while preserving the predicted linear
velocity.

## Standalone pool launch

Two robots:

```bash
ros2 launch formation_control_ros mocap_odom_ekf.launch.py \
  robots:=splash,bubble
```

Three robots:

```bash
ros2 launch formation_control_ros mocap_odom_ekf.launch.py \
  robots:=splash,glub,bubble
```

No frame arguments are required: the standalone launch defaults to raw
NED/FRD input and core-NWU/FLU output.

To explicitly ignore IMU:

```bash
ros2 launch formation_control_ros mocap_odom_ekf.launch.py \
  robots:=splash,bubble \
  use_imu_gyro:=false
```

## Live checks

Raw laboratory pose:

```bash
ros2 topic echo /mocap/splash/pose --once
```

Same pose after only the input frame/origin transform:

```bash
ros2 topic echo /mocap/splash/pose_core --once
```

Filtered controller-facing state:

```bash
ros2 topic echo /mocap/splash/odom_ekf --once
```

Rates:

```bash
ros2 topic hz /mocap/splash/pose
ros2 topic hz /mocap/splash/odom_ekf
```

During a MoCap outage, `/pose` should stop while `/odom_ekf` continues near the
configured EKF publication rate.

Useful numerical checks for the current pool configuration:

- core-NWU `z` is negative below the surface;
- the initialization depth is approximately `z=-1.45 m`;
- moving in pool/core +x increases `pose_core.position.x`;
- moving in pool/core +y increases `pose_core.position.y`;
- a nearly level robot should not be rejected by the body-z tilt gate.
