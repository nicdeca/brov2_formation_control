# SITL simulation setup

## Terminal 1 — Gazebo + PX4 SITL

Three robots:

```bash
cd ~/discower_ws

source /opt/ros/jazzy/setup.bash
source ~/discower_ws/install/setup.bash

ros2 launch formation_control_ros multi_bluerov2_sim.launch.py \
  robot_count:=3 \
  px4_dir:=/home/nicola/Gits/KTH-PX4/PX4-Autopilot
```

Two robots:

```bash
cd ~/discower_ws

source /opt/ros/jazzy/setup.bash
source ~/discower_ws/install/setup.bash

ros2 launch formation_control_ros multi_bluerov2_sim.launch.py \
  robot_count:=2 \
  px4_dir:=/home/nicola/Gits/KTH-PX4/PX4-Autopilot
```

For two-robot SITL, the expected vehicle names are normally:

```text
itrl_rov_1
itrl_rov_2
```

## Terminal 2 — Micro XRCE-DDS agent

```bash
micro-xrce-dds-agent udp4 -p 8888
```

## QGroundControl

Open one QGroundControl instance.

Before starting the experiment:

- confirm that every intended vehicle appears;
- confirm there are no unresolved PX4 health failures;
- confirm every intended vehicle can be armed;
- confirm every intended vehicle reaches Offboard;
- verify that the robot IDs shown by QGC match the ROS namespaces used by the
  launch file.

## Useful checks

PX4 processes:

```bash
pgrep -af px4
```

Odometry:

```bash
ros2 topic hz /itrl_rov_1/fmu/out/vehicle_odometry
ros2 topic hz /itrl_rov_2/fmu/out/vehicle_odometry
```

Controller snapshots:

```bash
ros2 topic list | grep diagnostic_snapshot
```

Expected control rate is approximately 50 Hz.
