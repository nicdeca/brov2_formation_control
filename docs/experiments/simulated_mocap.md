# Simulated MoCap and gyro for EKF SITL validation

PX4/Gazebo SITL does not provide the physical experiment topics

```text
/mocap/<robot>/pose
/<robot>/mavros/imu/data
```

The maintained `simulated_mocap` node creates a software-equivalent sensor
interface from PX4 SITL `VehicleOdometry`.

## SITL signal path

```text
/<robot>/fmu/out/vehicle_odometry
                 |
                 | formation_control_ros PX4 state conversion
                 v
       simulated_mocap
          |             |
          |             +--> /mocap/<robot>/imu
          |                    body FLU gyro
          v
/mocap/<robot>/pose
core NWU / FLU
          |
          v
    mocap_odom_ekf
          |
          +--> /mocap/<robot>/pose_core
          |
          +--> /mocap/<robot>/odom_ekf
                       |
                       v
                 controller
```

The simulated pose and gyro are both derived from PX4 odometry. Therefore this
test validates software integration and frame handling; it is not an
independent estimator benchmark.

## Two robots

Start Gazebo/PX4:

```bash
ros2 launch formation_control_ros multi_bluerov2_sim.launch.py \
  robot_count:=2 \
  robot_1_name:=splash \
  robot_2_name:=glub \
  px4_dir:=/home/nicola/Gits/KTH-PX4/PX4-Autopilot
```

Start the DDS agent:

```bash
micro-xrce-dds-agent udp4 -p 8888
```

Then validate the complete estimator path:

```bash
ros2 launch formation_control_ros \
  two_robot_experiment_with_ekf_sitl.launch.py \
  leader:=splash \
  follower:=glub \
  dry_run:=true \
  leader_reference_mode:=velocity \
  workspace_barrier_enabled:=true \
  workspace_adaptive:=true
```

Check:

```bash
ros2 topic echo /mocap/glub/pose --once
ros2 topic echo /mocap/glub/imu --once
ros2 topic echo /mocap/glub/pose_core --once
ros2 topic echo /mocap/glub/odom_ekf --once
```

The estimator status should report

```text
angular_source=gyro
```

For active SITL initialization/control, repeat with

```text
dry_run:=false
```

## Three robots

Use the corresponding simulator command with `splash`, `glub`, and `bubble`,
then:

```bash
ros2 launch formation_control_ros \
  three_robot_experiment_with_ekf_sitl.launch.py \
  leader:=splash \
  follower_left:=glub \
  follower_right:=bubble \
  dry_run:=true \
  leader_reference_mode:=velocity
```

The wrapper explicitly forces the `gazebo` dynamics preset even though the
robot names match the physical vehicles.
