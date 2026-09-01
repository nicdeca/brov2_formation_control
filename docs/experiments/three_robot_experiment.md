# Three-Robot Formation Experiment

This document describes the current three-BlueROV formation experiment and
the associated mission runners.

## Topology

The experiment uses a leader with two direct followers:

```text
             robot 1
             /     \
        robot 2   robot 3
```

The directed follower-to-parent sensing/control graph is

```text
robot 2 -> robot 1
robot 3 -> robot 1
```

The launch file uses the generic robot names by default:

```text
robot 1 = itrl_rov_1
robot 2 = itrl_rov_2
robot 3 = itrl_rov_3
```

Robot names are launch arguments, so the same launch file is used for the
physical experiment without modifying the source.

For the current three physical BlueROVs, use

```text
robot 1 = splash
robot 2 = glub
robot 3 = bubble
```

that is,

```text
             splash
             /    \
          glub    bubble
```

with

```text
glub   -> splash
bubble -> splash
```

## Launch file

Install

```text
three_robot_experiment.launch.py
```

under

```text
ros2/formation_control_ros/launch/
```

The launch uses the current pool-aligned core-NWU geometry:

```text
leader:
    [2.675,  0.050, -0.775]

follower 2:
    [4.475, -0.650, -0.775]

follower 3:
    [4.475,  0.750, -0.775]
```

with nominal parent-minus-follower vectors

```text
robot 2 -> robot 1:
    [-1.800,  0.700, 0.000]

robot 3 -> robot 1:
    [-1.800, -0.700, 0.000]
```

The available formation commands are

```text
triangle_nominal
triangle_wide
triangle_compact
triangle_high
```

The launch also uses the current physical/conservative pool workspace and
adaptive workspace barrier settings.

## Robot dynamics configuration

Each robot has an independent configuration argument:

```text
leader_robot_configuration
follower_left_robot_configuration
follower_right_robot_configuration
```

All default to

```text
auto
```

so the current robot-name-to-model mapping is used when available. They can
also be set explicitly to

```text
gazebo
standard
heavy_tube
```

if required.

## Physical three-robot launch

For the physical experiment with `splash` as leader, `glub` as follower 2,
and `bubble` as follower 3:

```bash
ros2 launch formation_control_ros three_robot_experiment.launch.py \
  dry_run:=false \
  leader:=splash \
  follower_left:=glub \
  follower_right:=bubble \
  leader_reference_mode:=velocity \
  workspace_barrier_enabled:=true \
  workspace_adaptive:=true
```

If desired, the model presets can be specified explicitly:

```bash
ros2 launch formation_control_ros three_robot_experiment.launch.py \
  dry_run:=false \
  leader:=splash \
  follower_left:=glub \
  follower_right:=bubble \
  leader_robot_configuration:=heavy_tube \
  follower_left_robot_configuration:=heavy_tube \
  follower_right_robot_configuration:=standard \
  leader_reference_mode:=velocity \
  workspace_barrier_enabled:=true \
  workspace_adaptive:=true
```

The launch defaults to `dry_run:=true`; keep this default for the first topic,
state, and initialization checks.

## Generic/default launch

With the default robot names:

```bash
ros2 launch formation_control_ros three_robot_experiment.launch.py
```

is equivalent to

```bash
ros2 launch formation_control_ros three_robot_experiment.launch.py \
  leader:=itrl_rov_1 \
  follower_left:=itrl_rov_2 \
  follower_right:=itrl_rov_3
```

## Mission runners

Two three-robot mission runners are available:

```text
scripts/run_three_robot_experiment.py
scripts/run_three_robot_experiment_final.py
```

Both wait for the global `FORMATION` phase, verify the two follower
formation-command subscriptions and the leader `cmd_vel` subscriber, and then
execute a reproducible sequence.

The runners use the current pool-aligned core-NWU velocity directions.

### Moderate mission

Run

```bash
python scripts/run_three_robot_experiment.py --leader splash
```

The sequence is

```text
triangle_nominal
    |
+0.60 m in pool-frame x
    |
triangle_wide
    |
-0.40 m in pool-frame y
    |
triangle_high
    |
undo x translation
    |
triangle_nominal
    |
undo y translation
```

This mission is recommended for the first three-physical-robot test.

### Final / large-excursion mission

Run

```bash
python scripts/run_three_robot_experiment_final.py --leader splash
```

The sequence is

```text
triangle_nominal
    |
+1.40 m in pool-frame x
    |
triangle_wide
    |
triangle_compact
    |
-0.70 m in pool-frame y
    |
triangle_high
    |
undo x translation
    |
triangle_nominal
    |
undo y translation
```

The contraction to `triangle_compact` is performed before the larger lateral
leader maneuver to increase workspace clearance for both followers.

## Difference from the older mission files

The older mission runners used the pre-pool-alignment velocity directions.
The current frame mapping changes the horizontal commands as follows:

```text
old +y -> current +x
old +x -> current -y
```

Accordingly, the mission geometry is unchanged conceptually, but the
published `Twist` components have been rotated into the current pool-aligned
core frame.

The final mission differs from the moderate mission in two main ways:

1. larger leader translations (`1.4 m` and `0.7 m` instead of `0.6 m` and
   `0.4 m`);
2. an additional `triangle_compact` phase before the large lateral motion.

## Recommended procedure

1. Start the three robot state/PX4 interfaces.
2. Launch the controller with `dry_run:=true`.
3. Verify that all three robot states are received and that the experiment
   reaches the expected initialization phase.
4. Verify the graph:
   `glub -> splash` and `bubble -> splash`.
5. Check that the initial physical and adaptive sensing constraints are
   strictly admissible.
6. Re-launch with `dry_run:=false`.
7. Arm/enter Offboard according to the standard experiment procedure.
8. Run the moderate mission.
9. Only after validating the moderate mission, run the final large-excursion
   mission.
