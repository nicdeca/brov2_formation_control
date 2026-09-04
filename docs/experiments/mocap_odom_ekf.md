# MoCap state estimator: frame contract and tuning

The maintained MoCap state estimator lives in

```text
ros2/formation_control_ros/
├── formation_control_ros/mocap_odom_ekf.py
└── launch/mocap_odom_ekf.launch.py
```

The executable remains named `mocap_odom_ekf` for compatibility, but the
revised implementation is deliberately simpler and more suitable for
closed-loop control:

- constant-velocity Kalman filtering for translation;
- direct gated MoCap attitude updates;
- body angular velocity from IMU gyro when available;
- finite-difference MoCap attitude as a fallback angular-rate source.

The previous implementation inferred angular velocity through the covariance
coupling of a quaternion CV EKF. In aggressive rotations this introduced
substantial angular-rate and attitude lag. The revised estimator removes that
source of lag.

## Fixed controller-facing frame contract

Everything published by the estimator has one unambiguous convention:

| quantity | convention |
|---|---|
| world frame | core / pool-aligned NWU |
| body frame | FLU |
| quaternion | rotation body FLU -> core NWU |
| `Odometry.twist` | body FLU |
| output parent frame | `core_nwu` |
| output child frame | `<robot>/base_link_ekf` (FLU) |

The controller must therefore consume the estimator as

```text
state_source:=nav_msgs
mocap_world_frame:=core_nwu
odom_twist_frame:=body
```

No FRD->FLU transform is applied downstream of the estimator.

## Real MoCap input

The real MoCap bridge publishes a `geometry_msgs/PoseStamped` on

```text
/mocap/<robot>/pose
```

The estimator converts that incoming pose to the fixed core-NWU / FLU contract
at its input boundary.

This is intentionally configurable because the physical MoCap system may
differ from the controller in:

- world-axis convention;
- world origin;
- rigid-body axis convention;
- rigid-body reference point.

### World-frame modes

```text
input_world_frame:=core_nwu
input_world_frame:=ros_enu
input_world_frame:=custom
```

`core_nwu` means that the incoming position/quaternion already use the
pool-aligned core frame.

`ros_enu` applies

```text
[x, y, z]_core = [y, -x, z]_ENU
```

before the optional translation.

`custom` uses

```text
world_to_core_quaternion_xyzw
```

as the rotation from incoming-world coordinates to core-NWU coordinates.

For all three modes,

```text
world_to_core_translation
```

is added after the world-axis rotation. This is the parameter that handles a
MoCap origin that is far from the pool/controller origin. An axis conversion
alone cannot fix an origin offset.

### Incoming body-frame modes

```text
input_body_frame:=flu
input_body_frame:=frd
input_body_frame:=custom
```

`flu` means the rigid-body orientation in the incoming PoseStamped already
describes the robot FLU body.

`frd` applies the proper 180-degree rotation about body +x:

```text
[x, y, z]_FRD = [x, -y, -z]_FLU
```

`custom` uses

```text
body_flu_to_input_quaternion_xyzw
```

for an arbitrary fixed rigid-body-axis calibration.

If the MoCap rigid-body origin is not the desired controller body origin, use

```text
body_origin_offset_input_body
```

which is the vector from the MoCap pose origin to the controller body origin,
expressed in the incoming rigid-body frame.

### Transform equations

Let:

- `M` be the incoming MoCap world;
- `Bm` be the incoming MoCap rigid-body frame;
- `C` be core NWU;
- `B` be controller body FLU.

The estimator computes

```text
p_C = t_C_M + R_C_M ( p_M + R_M_Bm r_Bm_B )
R_C_B = R_C_M R_M_Bm R_Bm_B
```

where all configurable transforms above have exactly these meanings.

## Transformed raw MoCap output

Every finite incoming pose is also republished, before estimator gating, as

```text
/mocap/<robot>/pose_core
```

with

```text
frame_id = core_nwu
body convention = FLU
```

This topic is important for experimental diagnostics. It lets the offline
plots compare:

```text
PX4 estimate
MoCap estimator output
raw MoCap after only the frame transform
```

without mixing coordinate systems.

## Angular velocity

### Preferred path: IMU gyro

The estimator uses gyro by default:

```text
use_imu_gyro:=true
```

The hardware convenience launches default to

```text
/<robot>/mavros/imu/data
```

and assume

```text
imu_body_frame:=flu
```

which is the normal ROS `base_link` convention.

Other supported conventions are

```text
imu_body_frame:=frd
imu_body_frame:=custom
```

For `custom`, provide

```text
body_flu_to_imu_quaternion_xyzw
```

The estimator converts the gyro to FLU once at the ROS input boundary and does
not apply any later FRD/FLU sign changes.

### Gyro filtering

Important parameters:

```text
gyro_timeout_sec:=0.20
gyro_time_constant_sec:=0.03
gyro_std:=0.03
max_gyro_abs_rad_s:=5.0
```

If the gyro is missing or stale, the estimator automatically falls back to a
finite-difference angular velocity reconstructed from successive accepted
MoCap attitudes.

The status line reports the active source:

```text
angular_source=gyro
```

or

```text
angular_source=mocap_fd
```

## Attitude behavior

The estimator no longer strongly smooths the MoCap quaternion through a
constant-angular-velocity EKF.

By default:

```text
orientation_measurement_gain:=1.0
orientation_std:=0.015
```

so every accepted MoCap attitude update is applied directly. This is deliberate
for feedback control: the previous large attitude/angular-rate lag was more
damaging than the small high-frequency MoCap noise.

Set `orientation_measurement_gain < 1` only if physical MoCap data show a real
need for additional smoothing.

## Translation

The translational filter remains a standard constant-velocity Kalman filter.

Defaults:

```text
position_std:=0.01
linear_accel_std:=0.7
initial_velocity_std:=0.5
max_position_innovation_m:=0.50
```

Linear velocity is estimated in core-NWU and rotated to body FLU for the
published Odometry.

## Hardware launch

Two robots:

```bash
ros2 launch formation_control_ros \
  two_robot_experiment_with_ekf.launch.py \
  leader:=splash \
  follower:=glub \
  dry_run:=true \
  leader_reference_mode:=velocity
```

Three robots:

```bash
ros2 launch formation_control_ros \
  three_robot_experiment_with_ekf.launch.py \
  leader:=splash \
  follower_left:=glub \
  follower_right:=bubble \
  dry_run:=true \
  leader_reference_mode:=velocity
```

These commands assume that the real MoCap PoseStamped is already core-NWU /
FLU.

If the actual lab bridge is standard ENU, for example:

```bash
ros2 launch formation_control_ros \
  two_robot_experiment_with_ekf.launch.py \
  leader:=splash \
  follower:=glub \
  dry_run:=true \
  input_world_frame:=ros_enu
```

If a translation or arbitrary rigid transform is required, use the `custom`
parameters documented above.

## Mandatory real-experiment frame check

Before `dry_run:=false`, verify all three streams:

```bash
ros2 topic echo /mocap/glub/pose --once
ros2 topic echo /mocap/glub/pose_core --once
ros2 topic echo /mocap/glub/odom_ekf --once
```

Then physically check:

1. moving the robot in core +x increases `pose_core.position.x`;
2. moving in core +y increases `pose_core.position.y`;
3. moving upward increases core z;
4. positive FLU roll/pitch/yaw signs agree with the controller convention;
5. `/odom_ekf.twist.twist.angular` agrees in sign with the real gyro;
6. `pose_core` lies inside the configured pool workspace.

This check is especially important if the raw MoCap origin is not the
pool/controller origin.

## Logging behavior

The MoCap estimator performs an internal health check at
`status_period_sec` (default `2.0 s`), but **healthy operation is silent**.

It reports only abnormal conditions, including:

- no new MoCap pose samples;
- pose samples arriving while the estimator cannot initialize;
- rejected MoCap measurements;
- missing/stale gyro data when gyro fusion is enabled;
- fallback from `gyro` to `mocap_fd`;
- coast timeout or estimator re-anchoring.

The normal startup configuration message is still printed once.

This keeps experiment logs readable while preserving failure diagnostics.

## Quick verification

A healthy real MoCap-estimator pipeline should satisfy:

```bash
ros2 topic echo /mocap/glub/pose --once
ros2 topic echo /mocap/glub/pose_core --once
ros2 topic echo /mocap/glub/odom_ekf --once
```

and, when gyro fusion is enabled:

```bash
ros2 topic echo /glub/mavros/imu/data --once
```

For SITL, the gyro check is instead:

```bash
ros2 topic echo /mocap/glub/imu --once
```

The absence of periodic INFO status lines is expected when everything is
working normally.
