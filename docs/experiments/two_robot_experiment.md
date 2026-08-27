# Two-robot experiment

## Topology

The launch defaults to `itrl_rov_1` / `itrl_rov_2`, but hardware names can
be passed directly. The known laboratory dynamics mapping is

```text
glub    -> heavy_tube
splash  -> heavy_tube
bubble  -> standard
```

Both per-robot dynamics arguments default to `auto`, so a command such as
`leader:=splash follower:=bubble` selects the appropriate characterized models
without additional arguments.

Formation edge:

```text
itrl_rov_2 -> itrl_rov_1
```

The follower tracks the desired parent-minus-follower vector.

## Terminal 1 — simulator or hardware state stack

For SITL, see `simulation_setup.md`.

For hardware, start the real state-estimation/PX4 stack instead.

## Terminal 2 — DDS bridge

For SITL:

```bash
micro-xrce-dds-agent udp4 -p 8888
```

Use the lab's actual DDS/network procedure for hardware if different.

## Terminal 3 — controller launch

```bash
cd ~/discower_ws

source /opt/ros/jazzy/setup.bash
source ~/discower_ws/src/brov2_formation_control/setup_ros2.sh
source ~/discower_ws/install/setup.bash

ros2 launch formation_control_ros two_robot_experiment.launch.py \
  dry_run:=false \
  leader_reference_mode:=velocity \
  workspace_barrier_enabled:=true \
  workspace_adaptive:=true
```

All launch arguments and defaults are listed in `launch_parameters.md`.

Optional explicit gains:

```bash
ros2 launch formation_control_ros two_robot_experiment.launch.py \
  dry_run:=false \
  leader_reference_mode:=velocity \
  position_gain:=3.0 \
  formation_gain:=3.0 \
  virtual_linear_gain:=0.55 \
  virtual_angular_gain:=0.80 \
  command_filter_linear_bandwidth:=3.0 \
  command_filter_angular_bandwidth:=4.0 \
  alpha_gain:=0.8 \
  workspace_barrier_enabled:=true \
  workspace_adaptive:=true
```

## Terminal 4 — recorder

Start before arming:

```bash
cd ~/discower_ws/src/brov2_formation_control
source setup_ros2.sh
source ~/discower_ws/install/setup.bash

scripts/record_formation_experiment.sh \
  --name two_robot_experiment \
  --robots itrl_rov_1,itrl_rov_2 \
  --edge itrl_rov_2:itrl_rov_1
```

## Terminal 5 — experiment runner

Start before arming:

```bash
cd ~/discower_ws/src/brov2_formation_control
source setup_ros2.sh
source ~/discower_ws/install/setup.bash

python scripts/run_two_robot_experiment.py --profile cautious
```

The runner waits for `FORMATION`, so it is safe to start before arming.

## Arm + Offboard

Use QGroundControl to arm both robots and enable Offboard.

The expected phase sequence is:

```text
INITIALIZE
    |
    v
All robots are inside initialization tolerances
    |
    v
settle dwell
    |
    v
FORMATION
```

## Profiles

### `cautious`

Use for the first real-water validation.

Contains:

- nominal formation;
- moderate far formation;
- nominal;
- small +y leader translation;
- small -y leader translation;
- close formation;
- nominal.

### `full`

Use after `cautious` is validated.

Contains larger formation changes and moderate leader translation.

### `challenging`

Stress-test profile. Primarily intended for simulation until real hardware has
been validated thoroughly.

Includes formations intentionally entering the relaxable sensing region:

```text
pair_range_far_edge
pair_range_close_edge
pair_fov_edge
```

and faster/larger leader maneuvers.

## After the run

Allow the robots to settle briefly, then stop the recorder with Ctrl-C.

Export:

```bash
RUN=$(ls -dt outputs/experiments/*_two_robot_experiment | head -n 1)

python scripts/export_formation_bag.py "$RUN"
```

Plot:

```bash
uv run python scripts/plot_formation_experiment.py \
  "$RUN/formation_history.npz" \
  --paper \
  --paper-quality \
  --save \
  --format pdf
```

Inspect at minimum:

- formation actual vs desired;
- formation error norm;
- leader actual vs desired position;
- thruster forces and signed limits for each robot;
- sensing margins and relaxation;
- workspace margin and relaxation;
- CLF slack / required slack;
- actuation margin;
- controller timing.
