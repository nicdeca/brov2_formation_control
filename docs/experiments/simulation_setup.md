# SITL setup and state-estimator test matrix

## Common simulator processes

Two robots:

```bash
ros2 launch formation_control_ros multi_bluerov2_sim.launch.py \
  robot_count:=2 \
  robot_1_name:=splash \
  robot_2_name:=bubble \
  px4_dir:=$PX4_AUTOPILOT_DIR
```

Three robots:

```bash
ros2 launch formation_control_ros multi_bluerov2_sim.launch.py \
  robot_count:=3 \
  robot_1_name:=splash \
  robot_2_name:=glub \
  robot_3_name:=bubble \
  px4_dir:=$PX4_AUTOPILOT_DIR
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


## Pool-centered mission geometry

The maintained two- and three-robot mission launches use the same pool-aligned
workspace as the wet experiment:

```text
physical x:      [0.300, 7.100] m
physical y:      [-1.975, 1.975] m
conservative x:  [0.450, 6.950] m
conservative y:  [-1.825, 1.825] m
```

The x/y center is `[3.70, 0.00]`. Standard mission references and leader
translations are chosen around this region rather than using one-sided large
excursions. This both increases workspace reserve and corresponds to the
preferred MoCap visibility region in the pool.
