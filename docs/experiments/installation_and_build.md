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

After substantial ROS-package changes, a clean package rebuild can avoid stale
installed entry points:

```bash
cd ~/discower_ws
rm -rf build/formation_control_ros install/formation_control_ros
source src/brov2_formation_control/setup_ros2.sh
colcon build --packages-select formation_control_ros --symlink-install
source install/setup.bash
```

If the generated ROS Python entry points contain an invalid interpreter line,
use the repository workaround:

```bash
bash src/brov2_formation_control/scripts/fix_ros2_shebangs.sh
```

## Offline / experiment scripts

The currently maintained workflow uses:

```text
scripts/record_formation_experiment.sh
scripts/record_formation_experiment.py
scripts/export_experiment.py
scripts/export_initialization_bag.py
scripts/export_formation_bag.py
scripts/plot_experiment.py
scripts/plot_initialization_experiment.py
scripts/plot_formation_experiment.py
scripts/run_two_robot_experiment.py
scripts/run_two_robot_adaptive_mission.py
scripts/run_five_robot_tree_experiment.py
```

The old three-robot runners and old one-off relaxation scripts are retained only
as historical/debug material unless explicitly ported to the current pool
frame.

## Environments

For ROS-aware scripts such as bag exporters:

```bash
cd ~/discower_ws/src/brov2_formation_control
source setup_ros2.sh
source ~/discower_ws/install/setup.bash
```

The umbrella plotter and the standard mission plotter are ROS-independent once
the NPZ files have been exported and can normally be run through the core
environment:

```bash
uv run python scripts/plot_experiment.py <RUN>
uv run python scripts/plot_formation_experiment.py ...
```

The initialization exporter needs ROS because it reads rosbag/PX4 messages.

## Rebuild policy

A ROS rebuild is required after changing:

- nodes under `ros2/formation_control_ros/formation_control_ros/`;
- launch files under `ros2/formation_control_ros/launch/`;
- installed Python/core code imported by the ROS nodes.

A rebuild is not required after changing standalone scripts under `scripts/`,
although the correct environment still needs to be sourced.
