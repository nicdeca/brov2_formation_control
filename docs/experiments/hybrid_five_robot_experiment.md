# Hybrid Five-Robot Experiment

This document describes the mixed-reality five-robot experiment combining
two physical BlueROV2 vehicles with three vehicles simulated in Gazebo.

## Robot mapping

The five-robot controller uses the directed sensing/control graph

```text
2 -> 1
3 -> 1
4 -> 2
5 -> 3
```

with the following assignment:

| Controller index | Robot | Platform |
| --- | --- | --- |
| 1 | `rv1` | Gazebo |
| 2 | `splash` | Physical BlueROV2 |
| 3 | `rv2` | Gazebo |
| 4 | `glub` | Physical BlueROV2 |
| 5 | `rv3` | Gazebo |

Hence, the hybrid formation is

```text
                    rv1
                  (simulated)
                   /      \
                  /        \
             splash         rv2
            (physical)   (simulated)
                |             |
                |             |
              glub           rv3
           (physical)     (simulated)
```

Equivalently, the directed follower-to-parent edges are

```text
splash -> rv1
rv2    -> rv1
glub   -> splash
rv3    -> rv2
```

This preserves the same balanced-tree topology used by the five-robot
simulation. The `glub -> splash` branch is entirely physical, while the other
branch is simulated. Since `splash` follows the simulated leader `rv1`, its
relative sensing relation to the leader is virtual and is provided through the
common ROS 2 state interface.

## Timing

The hybrid experiment must use wall time for the formation-control nodes:

```text
gazebo_timer:=false
```

The physical vehicles cannot follow Gazebo simulation time. Gazebo should
therefore run approximately in real time while the controller nodes use the
wall-clock timers.

## Vehicle models

The simulated vehicles must use the Gazebo BlueROV2 model, whereas the
physical vehicles use the physical `heavy_tube` model:

```text
rv1     -> gazebo
splash  -> heavy_tube
rv2     -> gazebo
glub    -> heavy_tube
rv3     -> gazebo
```

The current `five_robot_tree_experiment.launch.py` uses a single common
`robot_configuration="gazebo"` parameter. Before running the hybrid
experiment, expose one configuration argument per robot, for example

```text
robot_1_configuration
robot_2_configuration
robot_3_configuration
robot_4_configuration
robot_5_configuration
```

and pass each value to the corresponding leader/follower controller.

## 1. Launch the simulated robots

Only the three virtual robots should be spawned in Gazebo. Do **not** create
simulated copies of `splash` or `glub`.

From `~/discower_ws`:

```bash
ros2 launch formation_control_ros multi_bluerov2_sim.launch.py \
  robot_count:=3 \
  robot_1_name:=rv1 \
  robot_2_name:=rv2 \
  robot_3_name:=rv3 \
  rov_1_pose:="-1.050,2.675,-1.275,0,0,-1.57079632679" \
  rov_2_pose:="-1.700,4.775,-1.275,0,0,-1.57079632679" \
  rov_3_pose:="-1.550,5.775,-1.275,0,0,-1.57079632679" \
  px4_dir:=/home/nicola/Gits/KTH-PX4/PX4-Autopilot
```

The three Gazebo poses correspond to robots 1, 3, and 5 of the original
five-robot experiment.

## 2. Start the physical robots

Bring up `splash` and `glub` using the standard physical-robot procedure so
that their PX4/state topics are available on the same ROS 2 graph as the
Gazebo vehicles.

Before starting the formation controller, verify that the following five robot
names are simultaneously visible:

```text
rv1
splash
rv2
glub
rv3
```

## 3. Launch the five-robot formation controller

After adding the per-robot configuration arguments described above, launch

```bash
ros2 launch formation_control_ros five_robot_tree_experiment.launch.py \
  dry_run:=false \
  robot_1:=rv1 \
  robot_2:=splash \
  robot_3:=rv2 \
  robot_4:=glub \
  robot_5:=rv3 \
  robot_1_configuration:=gazebo \
  robot_2_configuration:=heavy_tube \
  robot_3_configuration:=gazebo \
  robot_4_configuration:=heavy_tube \
  robot_5_configuration:=gazebo \
  leader_reference_mode:=velocity \
  gazebo_timer:=false \
  workspace_barrier_enabled:=true \
  workspace_adaptive:=true
```

The resulting controller graph is

```text
splash -> rv1
rv2    -> rv1
glub   -> splash
rv3    -> rv2
```

with `rv1` acting as the leader.

## Practical checks before enabling actuation

Before running with `dry_run:=false`, verify:

1. All five state streams use the same world-frame convention expected by the
   controller.
2. `rv1`, `rv2`, and `rv3` are advancing in Gazebo close to real time.
3. `splash` and `glub` publish their physical state normally.
4. Every follower resolves the intended parent:
   - `splash` -> `rv1`
   - `rv2` -> `rv1`
   - `glub` -> `splash`
   - `rv3` -> `rv2`
5. The physical and simulated robots use the appropriate vehicle model
   (`heavy_tube` and `gazebo`, respectively).
6. `gazebo_timer` remains disabled for the hybrid experiment.
7. The initial physical and adaptive sensing constraints are admissible for all
   four follower-parent edges.

A first hybrid run should preferably be performed with conservative leader
commands and no formation transition, followed by the existing five-robot
formation sequence once the state routing and timing have been verified.
