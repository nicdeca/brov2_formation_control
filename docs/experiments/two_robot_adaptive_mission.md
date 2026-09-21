# Two-robot adaptive sensing-domain SITL mission

This is the dedicated SITL paper mission for demonstrating adaptive sensing
domain enlargement. It is intentionally separate from the normal two-robot
hardware launch.

## Why a separate launch exists

The mission uses tighter conservative sensing limits and different tuning:

```text
d_max_conservative   = 2.4 m
alpha_h_conservative = 0.45
alpha_v_conservative = 0.45

relaxation recovery       = 0.8
relaxation barrier gain   = 0.20
domain margin ratio       = 0.02
activation on ratio       = 0.001
activation off ratio      = 0.15
```

It fixes the robot dynamics to `gazebo` and keeps the common pool workspace
barriers active.

## Formation schedule

Parent-minus-follower vectors are expressed in pool-aligned core NWU:

```text
t =  0 s  adaptive_A  [-1.80, -0.70,  0.00]
t = 18 s  adaptive_B  [-1.65, -1.45, -0.40]
t = 38 s  adaptive_C  [-1.65,  1.45,  0.40]
t = 60 s  adaptive_D  [-2.30, -0.55, -0.30]
t = 80 s  adaptive_A
t =100 s  end
```

The moving-leader command is a tank-safe pool-length translation; the large
formation reconfigurations remain the main source of range/FoV adaptation.

## Launch

Simulator:

```bash
ros2 launch formation_control_ros multi_bluerov2_sim.launch.py \
  robot_count:=2 \
  px4_dir:=$PX4_AUTOPILOT_DIR
```

Controller:

```bash
ros2 launch formation_control_ros two_robot_adaptive_mission.launch.py \
  dry_run:=false
```

Recorder, before arming:

```bash
scripts/record_formation_experiment.sh \
  --name adaptive_mission \
  --robots itrl_rov_1,itrl_rov_2 \
  --edge itrl_rov_2:itrl_rov_1
```

Runner:

```bash
python scripts/run_two_robot_adaptive_mission.py \
  --leader itrl_rov_1
```

For a Gazebo simulation-time mission, launch with `gazebo_timer:=true` and add
`--gazebo-timer` to the runner.

## What to inspect

The important paper diagnostics are:

- distance and adaptive minimum/maximum range boundaries;
- horizontal/vertical normalized image coordinates and adaptive FoV limits;
- normalized relaxation states;
- minimum physical sensing margin;
- required CLF slack and actuation margin;
- thruster utilization;
- formation error;
- workspace margin/relaxation to confirm the event is sensing driven rather
  than caused by the tank boundary.

The normalized sensing enlargement state is not upper-clipped. `s > 1` should
therefore always be interpreted together with the physical-margin diagnostic.


## Post-processing shortcut

For a completed split-recorded run:

```bash
python scripts/export_experiment.py "$RUN"
uv run python scripts/plot_experiment.py "$RUN"
```

These commands process every available phase and keep initialization and mission
outputs in their respective folders.
