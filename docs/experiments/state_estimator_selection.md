# State-estimator selection

The formation-control ROS layer supports either PX4 `VehicleOdometry` or
generic `nav_msgs/Odometry` without changing the ROS-independent core.

## PX4

```text
state_source:=px4
```

uses

```text
/<robot>/fmu/out/vehicle_odometry
```

which is converted at the ROS boundary from PX4 NED/FRD to core NWU/FLU.

## Revised MoCap estimator

```text
state_source:=nav_msgs
state_topic_template:='/mocap/{robot}/odom_ekf'
mocap_world_frame:=core_nwu
odom_twist_frame:=body
```

The revised estimator guarantees that its output already satisfies

```text
world = core NWU
body = FLU
```

The generic odometry adapter therefore performs no hidden FRD correction.

## Hardware convenience launches

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

The hardware wrappers enable IMU gyro fusion by default.

The incoming MoCap pose convention is independently configurable through

```text
input_world_frame
input_body_frame
world_to_core_translation
world_to_core_quaternion_xyzw
body_flu_to_input_quaternion_xyzw
body_origin_offset_input_body
```

See `mocap_odom_ekf.md` for the exact transform definition.

## SITL EKF validation

Two robots:

```bash
ros2 launch formation_control_ros \
  two_robot_experiment_with_ekf_sitl.launch.py \
  leader:=splash \
  follower:=glub \
  dry_run:=true \
  leader_reference_mode:=velocity
```

Three robots:

```bash
ros2 launch formation_control_ros \
  three_robot_experiment_with_ekf_sitl.launch.py \
  leader:=splash \
  follower_left:=glub \
  follower_right:=bubble \
  dry_run:=true \
  leader_reference_mode:=velocity
```

These wrappers create simulated MoCap pose and gyro topics from PX4 SITL and
feed them through exactly the same estimator interface used by hardware.

## Before wet actuation

Keep `dry_run:=true` until:

```bash
ros2 topic echo /mocap/glub/pose_core --once
ros2 topic echo /mocap/glub/odom_ekf --once
```

show a physically correct core-NWU / FLU state.

The offline estimator-comparison plots should then be used to verify PX4,
MoCap-estimator, and raw-MoCap agreement before enabling demanding formation
maneuvers.
