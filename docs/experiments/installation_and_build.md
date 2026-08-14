# Installation and build

## Repository and workspace

Expected workspace layout:

```text
~/discower_ws/
└── src/
    └── brov2_formation_control/
```

The commands below assume ROS 2 Jazzy.

## Build the ROS package

```bash
cd ~/discower_ws

source /opt/ros/jazzy/setup.bash
source ~/discower_ws/src/brov2_formation_control/setup_ros2.sh

colcon build \
  --packages-select formation_control_ros \
  --symlink-install

source ~/discower_ws/install/setup.bash
```

If the generated ROS Python entry points contain an invalid interpreter line,
reapply the known shebang workaround:

```bash
sed -i '1c #!/usr/bin/env python3' \
  ~/discower_ws/install/formation_control_ros/lib/formation_control_ros/leader_controller \
  ~/discower_ws/install/formation_control_ros/lib/formation_control_ros/follower_controller \
  ~/discower_ws/install/formation_control_ros/lib/formation_control_ros/offboard_heartbeat_wrench \
  ~/discower_ws/install/formation_control_ros/lib/formation_control_ros/experiment_phase_manager
```

## Python environment for offline tools

From the repository:

```bash
cd ~/discower_ws/src/brov2_formation_control
source setup_ros2.sh
```

The main offline tools are:

```text
scripts/record_formation_experiment.sh
scripts/export_formation_bag.py
scripts/plot_formation_experiment.py
scripts/run_two_robot_experiment.py
scripts/run_three_robot_experiment.py
```

## Rebuild policy

A rebuild is required after changing:

- ROS nodes under `ros2/formation_control_ros/formation_control_ros/`;
- launch files under `ros2/formation_control_ros/launch/`;
- installed Python package/core files used through the ROS package.

A rebuild is not required after changing standalone scripts under `scripts/`,
although the environment still needs to be sourced.
