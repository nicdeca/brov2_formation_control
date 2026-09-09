# SITL setup and state-estimator test matrix

## Common simulator processes

Two robots:

```bash
ros2 launch formation_control_ros multi_bluerov2_sim.launch.py \
  robot_count:=2 \
  robot_1_name:=splash \
  robot_2_name:=bubble \
  px4_dir:=/home/nicola/Gits/KTH-PX4/PX4-Autopilot
```

Three robots:

```bash
ros2 launch formation_control_ros multi_bluerov2_sim.launch.py \
  robot_count:=3 \
  robot_1_name:=splash \
  robot_2_name:=glub \
  robot_3_name:=bubble \
  px4_dir:=/home/nicola/Gits/KTH-PX4/PX4-Autopilot
```

DDS agent:

```bash
micro-xrce-dds-agent udp4 -p 8888
```

## State paths

Direct PX4:

```text
PX4 VehicleOdometry -> state_adapter -> controller core NWU/FLU
```

EKF path:

```text
PX4
 -> simulated raw MoCap NED/FRD
 -> mocap_odom_ekf
 -> core NWU/FLU
 -> controller
```

The simulated and real MoCap paths therefore exercise the same estimator input
frame convention.

## Recommended sequence

For each robot count:

1. direct PX4 state;
2. EKF + ideal MoCap + gyro;
3. EKF + intermittent MoCap + gyro;
4. EKF + intermittent MoCap + `use_imu_gyro:=false`;
5. repeat the validated estimator path with `dry_run:=false`;
6. run the cautious mission, then the full mission.

During intermittent tests compare:

```bash
ros2 topic hz /mocap/splash/pose
ros2 topic hz /mocap/splash/odom_ekf
```

and inspect:

```bash
ros2 topic echo /mocap/splash/pose_core --once
ros2 topic echo /mocap/splash/odom_ekf --once
```
