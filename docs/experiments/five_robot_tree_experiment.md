# Five-robot directed-tree SITL experiment

This is the current multi-robot scaling experiment.

## Topology

```text
4 -> 2 -> 1
5 -> 3 -> 1
```

Robot 1 is the leader. Each follower has exactly one parent, so the existing
one-parent decentralized follower controller is reused without a new control
architecture.

## Current initialization in pool-aligned core NWU

```text
p1 = [2.675,  0.050, -0.775]
p2 = [4.325, -0.650, -0.775]
p3 = [4.325,  0.750, -0.775]
p4 = [5.775, -1.150, -0.775]
p5 = [5.775,  1.250, -0.775]
```

Nominal parent-minus-follower vectors:

```text
d21 = [-1.650,  0.700, 0.000]
d31 = [-1.650, -0.700, 0.000]
d42 = [-1.450,  0.500, 0.000]
d53 = [-1.450, -0.500, 0.000]
```

## Launch

Simulator:

```bash
ros2 launch formation_control_ros multi_bluerov2_sim.launch.py \
  robot_count:=5 \
  px4_dir:=$PX4_AUTOPILOT_DIR
```

Controller:

```bash
ros2 launch formation_control_ros five_robot_tree_experiment.launch.py \
  dry_run:=false \
  leader_reference_mode:=velocity \
  workspace_barrier_enabled:=true \
  workspace_adaptive:=true
```

All five controllers use the `gazebo` dynamics preset explicitly.

Recorder:

```bash
scripts/record_formation_experiment.sh \
  --name five_robot_tree \
  --robots itrl_rov_1,itrl_rov_2,itrl_rov_3,itrl_rov_4,itrl_rov_5 \
  --edge itrl_rov_2:itrl_rov_1 \
  --edge itrl_rov_3:itrl_rov_1 \
  --edge itrl_rov_4:itrl_rov_2 \
  --edge itrl_rov_5:itrl_rov_3
```

Runner:

```bash
python scripts/run_five_robot_tree_experiment.py \
  --leader itrl_rov_1 \
  --profile full
```

## Profiles

`cautious` validates nominal/wide/compact/staggered tree shapes and small leader
motions. `full` uses larger translations. `challenging` adds stronger moving-
leader and 3-D excitation.

The launch also contains additional 3-D formation references
`tree_depth_split`, `tree_crossed_3d`, `tree_opposed_3d`, and
`tree_parallel_3d` for demanding validation. Treat them as stress-test
references rather than default hardware targets.

## Timing note

`five_robot_tree_experiment.launch.py` has a launch-level `gazebo_timer`
argument, but the current five-robot mission runner does not have a matching
`--gazebo-timer` option. The normal scripted workflow therefore uses wall time.

## Acceptance checks

Inspect every directed edge, not only the first-level followers:

- formation error and actual/desired relative position;
- sensing margins and relaxation;
- physical margin for each parent-follower pair.

Inspect every robot for workspace margin/relaxation, thruster forces,
actuation feasibility and controller timing.


## Post-processing shortcut

For a completed split-recorded run:

```bash
python scripts/export_experiment.py "$RUN"
uv run python scripts/plot_experiment.py "$RUN"
```

These commands process every available phase and keep initialization and mission
outputs in their respective folders.
