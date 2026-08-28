# Two-robot experiment

This is the canonical normal two-robot experiment for SITL and real water.
The dedicated adaptive paper mission is documented separately in
`two_robot_adaptive_mission.md`.

## Topology and current initialization

Directed sensing edge:

```text
follower -> leader
```

The follower tracks the parent-minus-follower vector. In the current
pool-aligned core NWU frame:

```text
leader initialization   = [2.675, 0.050, -0.775]
follower initialization = [4.475, 0.750, -0.775]
nominal d_21             = [-1.800, -0.700, 0.000]
```

## Hardware dynamics names

```text
glub    -> heavy_tube
splash  -> heavy_tube
bubble  -> standard
```

On hardware, `leader_robot_configuration` and `follower_robot_configuration`
default to `auto` and use this mapping. In SITL, always explicitly select
`gazebo`, even if using the same robot names.

## Terminal 1 — simulator or hardware state stack

For SITL, follow `simulation_setup.md`.

For hardware, start the real PX4/state-estimation stack and verify the pool
frame as described in `hardware_setup.md`.

## Terminal 2 — DDS bridge

SITL example:

```bash
micro-xrce-dds-agent udp4 -p 8888
```

Use the actual lab networking procedure on hardware if different.

## Terminal 3 — controller launch

### Hardware example

```bash
cd ~/discower_ws
source /opt/ros/jazzy/setup.bash
source ~/discower_ws/src/brov2_formation_control/setup_ros2.sh
source ~/discower_ws/install/setup.bash

ros2 launch formation_control_ros two_robot_experiment.launch.py \
  leader:=splash \
  follower:=glub \
  dry_run:=false \
  leader_reference_mode:=velocity \
  workspace_barrier_enabled:=true \
  workspace_adaptive:=true
```

### SITL with hardware-like names

```bash
ros2 launch formation_control_ros two_robot_experiment.launch.py \
  leader:=splash \
  follower:=glub \
  leader_robot_configuration:=gazebo \
  follower_robot_configuration:=gazebo \
  dry_run:=false \
  leader_reference_mode:=velocity \
  workspace_barrier_enabled:=true \
  workspace_adaptive:=true
```

## Terminal 4 — split recorder

Start **before arming**:

```bash
cd ~/discower_ws/src/brov2_formation_control
source setup_ros2.sh
source ~/discower_ws/install/setup.bash

scripts/record_formation_experiment.sh \
  --name two_robot_experiment \
  --robots splash,glub \
  --edge glub:splash
```

Use the actual namespaces for the current run. The recorder immediately starts
`initialization/bag`, closes it at `FORMATION`, then records only the actual
mission after `mission_status = RUNNING`.

## Terminal 5 — mission runner

```bash
python scripts/run_two_robot_experiment.py \
  --leader splash \
  --profile cautious
```

The runner waits for `FORMATION` and required subscribers. It publishes
`WAITING`, `RUNNING`, `COMPLETE`, or `ABORTED` on
`/formation_control/mission_status`.

## Arm + Offboard

The phase manager requires all configured robots to remain within

```text
position tolerance = 0.65 m
speed tolerance    = 0.08 m/s
```

for `1.5 s` before switching to `FORMATION`.

## Profiles

### `cautious`

First hardware-validation profile. It uses nominal/far/close formation changes
and small leader translations.

### `full`

Larger formation transitions and moderate leader motion. Use after `cautious`
is reliable.

### `challenging`

Stress profile, primarily for simulation until hardware is validated. It uses

```text
pair_range_far_edge
pair_range_close_edge
pair_fov_edge
```

plus faster leader maneuvers. These targets are intended to excite adaptive
sensing-domain enlargement.

## Output and post-processing

A successful split run looks like

```text
outputs/experiments/<timestamp>_two_robot_experiment/
├── run_manifest.yaml
├── initialization/
│   ├── bag/
│   ├── initialization_history.npz
│   └── plots/
└── mission/
    ├── bag/
    ├── formation_history.npz
    └── plots/
```

The recorder post-processes by default. To locate the latest run:

```bash
RUN=$(ls -dt outputs/experiments/*_two_robot_experiment | head -n 1)
```

If manual processing is needed:

```bash
python scripts/export_initialization_bag.py "$RUN"
python scripts/plot_initialization_experiment.py \
  "$RUN/initialization/initialization_history.npz" --save

python scripts/export_formation_bag.py "$RUN" --phase mission
uv run python scripts/plot_formation_experiment.py \
  "$RUN/mission/formation_history.npz" \
  --paper --paper-quality --save --format pdf
```

Do not type the literal word `RUN` as a path; `<RUN>` in documentation means
the actual run directory.

## Minimum acceptance checks

Initialization:

- position histories and, when logged, initialization-reference overlays;
- speed convergence;
- PX4 armed/Offboard state;
- workspace physical margin and relaxation;
- required slack / thruster utilization.

Mission:

- actual vs desired formation;
- formation-error norm;
- leader tracking;
- all thruster forces and signed limits;
- sensing physical margins and relaxation;
- workspace margins and relaxation;
- CLF required slack / actuation margin;
- controller timing.
