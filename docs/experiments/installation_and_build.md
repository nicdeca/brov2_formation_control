# Installation and build

## Repository and workspace

The ROS 2 package is intended to be cloned inside the `src/` directory of a
colcon workspace:

```text
ros2_ws/
└── src/
    └── brov2_formation_control/   # repository root
```

The commands below assume ROS 2 Jazzy and, unless stated otherwise, are run
from the repository root. Repository and workspace paths are kept relative so
the instructions are independent of the local checkout location.

## Core Python environment

The ROS-independent package uses Python 3.12+ and [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
```

For ROS-aware tools, `setup_ros2.sh` creates the local `.venv-ros` environment
on first use and installs the core package in editable mode. The environment is
local-only and must not be committed to Git.

```bash
source setup_ros2.sh
```

`setup_ros2.sh` also sources the standard ROS 2 Jazzy installation and, when
available, the surrounding workspace `install/setup.bash`.

## Build the ROS package

From the repository root, move to the colcon workspace root, build, then return
to the repository:

```bash
source setup_ros2.sh

cd ../..
colcon build \
  --packages-select formation_control_ros \
  --symlink-install

source install/setup.bash
cd src/brov2_formation_control
```

After substantial ROS-package changes, a clean package rebuild can avoid stale
installed entry points:

```bash
cd ../..
rm -rf build/formation_control_ros install/formation_control_ros
source src/brov2_formation_control/setup_ros2.sh
colcon build --packages-select formation_control_ros --symlink-install
source install/setup.bash
cd src/brov2_formation_control
```

If generated ROS Python entry points contain an invalid interpreter line, use
the repository workaround from the repository root:

```bash
bash scripts/fix_ros2_shebangs.sh
```

## PX4 checkout

SITL launch files accept the PX4 checkout through the `px4_dir` launch
argument. For portable commands, set the environment variable
`PX4_AUTOPILOT_DIR` to your PX4-Autopilot checkout and use

```text
px4_dir:="$PX4_AUTOPILOT_DIR"
```

in launch commands. No user-specific PX4 path is assumed by the documentation.

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

For ROS-aware scripts such as bag exporters, from the repository root:

```bash
source setup_ros2.sh
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
