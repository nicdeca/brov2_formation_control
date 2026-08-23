# ROS 2 launch-file parameters

This page is the reference for the current user-facing ROS 2 launch arguments.
It documents the canonical simulator and experiment launch files.

## Dynamics configuration

The controller dynamics model provides three physical/model presets:

| Value | Use |
|---|---|
| `gazebo` | Current PX4/Gazebo SITL vehicle model |
| `standard` | Characterized standard laboratory BlueROV2 configuration |
| `heavy_tube` | Characterized heavy-tube laboratory configuration |

At the ROS layer, `robot_configuration=auto` is also supported. It resolves
known physical robot names as

| Robot | Configuration |
|---|---|
| `glub` | `heavy_tube` |
| `splash` | `heavy_tube` |
| `bubble` | `standard` |

`auto` is not a fourth dynamics model. Unknown names in `auto` mode raise an
error. Use an explicit preset for any other robot.

## `multi_bluerov2_sim.launch.py`

Launches multiple PX4 SITL BlueROV2 instances in one Gazebo world.

| Argument | Default | Meaning |
|---|---|---|
| `px4_dir` | auto-detected PX4 checkout | Path to `PX4-Autopilot` |
| `world` | `kth_marinarium_docking` | Gazebo world name without `.sdf` |
| `robot_count` | `3` | Number of robots, from 1 to 6 |
| `spawn_delay` | `5.0` | Delay [s] between successive PX4 instances |
| `rov_1_pose` ... `rov_6_pose` | launch-file defaults | Gazebo ENU pose `x,y,z,roll,pitch,yaw` for each robot |

Five-robot SITL:

```bash
ros2 launch formation_control_ros multi_bluerov2_sim.launch.py \
  robot_count:=5 \
  px4_dir:=/home/nicola/Gits/KTH-PX4/PX4-Autopilot
```

## `two_robot_experiment.launch.py`

Used for the two-robot hardware experiment and two-robot controller validation.

| Argument | Default | Meaning |
|---|---|---|
| `leader` | `itrl_rov_1` | Leader namespace/name |
| `follower` | `itrl_rov_2` | Follower namespace/name |
| `dt` | `0.02` | Controller period [s] |
| `dry_run` | `true` | Compute diagnostics without sending the commanded wrench |
| `leader_reference_mode` | `stationary` | `stationary`, `velocity`, or `trajectory` |
| `leader_robot_configuration` | `auto` | Leader dynamics selection: `auto`, `gazebo`, `standard`, `heavy_tube` |
| `follower_robot_configuration` | `auto` | Follower dynamics selection |
| `position_gain` | `2.0` | Leader position-potential gain |
| `formation_gain` | `2.0` | Follower relative-position-potential gain |
| `virtual_linear_gain` | `0.55` | Translational entries of the backstepping gain |
| `virtual_angular_gain` | `0.80` | Rotational entries of the backstepping gain |
| `command_filter_linear_bandwidth` | `3.0` | Translational command-filter bandwidth |
| `command_filter_angular_bandwidth` | `4.0` | Rotational command-filter bandwidth |
| `alpha_gain` | `1.8` | CLF-QP dissipation gain |
| `workspace_barrier_enabled` | `true` | Enable workspace barriers |
| `workspace_adaptive` | `true` | Enable adaptive workspace-domain enlargement |

Example with automatic real-robot dynamics selection:

```bash
ros2 launch formation_control_ros two_robot_experiment.launch.py \
  leader:=splash \
  follower:=bubble \
  dry_run:=false \
  leader_reference_mode:=velocity \
  workspace_barrier_enabled:=true \
  workspace_adaptive:=true
```

This selects `heavy_tube` for `splash` and `standard` for `bubble`.

Manual override remains possible:

```bash
ros2 launch formation_control_ros two_robot_experiment.launch.py \
  leader:=splash \
  follower:=bubble \
  leader_robot_configuration:=heavy_tube \
  follower_robot_configuration:=standard
```

## `five_robot_tree_experiment.launch.py`

Current five-robot SITL controller launch. All controllers explicitly use the
`gazebo` dynamics preset.

The directed tree is

```text
itrl_rov_2 -> itrl_rov_1
itrl_rov_3 -> itrl_rov_1
itrl_rov_4 -> itrl_rov_2
itrl_rov_5 -> itrl_rov_3
```

| Argument | Default | Meaning |
|---|---|---|
| `robot_1` ... `robot_5` | `itrl_rov_1` ... `itrl_rov_5` | Robot namespaces |
| `dt` | `0.02` | Controller period [s] |
| `dry_run` | `true` | Disable commanded actuation when true |
| `gazebo_timer` | `false` | Run controller timers from Gazebo `/clock` |
| `leader_reference_mode` | `stationary` | Leader reference mode |
| `position_gain` | `2.0` | Leader position-potential gain |
| `formation_gain` | `1.0` | Follower formation-potential gain |
| `virtual_linear_gain` | `1.0` | Translational backstepping gain |
| `virtual_angular_gain` | `1.2` | Rotational backstepping gain |
| `command_filter_linear_bandwidth` | `10.0` | Translational command-filter bandwidth |
| `command_filter_angular_bandwidth` | `10.0` | Rotational command-filter bandwidth |
| `alpha_gain` | `3.0` | CLF-QP dissipation gain |
| `workspace_barrier_enabled` | `true` | Enable workspace barriers |
| `workspace_adaptive` | `true` | Enable adaptive workspace-domain enlargement |

Canonical active-control command:

```bash
ros2 launch formation_control_ros five_robot_tree_experiment.launch.py \
  dry_run:=false \
  leader_reference_mode:=velocity \
  workspace_barrier_enabled:=true \
  workspace_adaptive:=true
```

For a Gazebo-time mission, add `gazebo_timer:=true` and run the mission script
with the corresponding `--gazebo-timer` option.

## Node-level parameters

The launch tables above list parameters intentionally exposed as launch
arguments. Additional controller constants are currently set inside the launch
files or declared by the nodes. Their meaning and tuning are documented in
`controller_parameters.md`. If a parameter must be changed routinely between
runs, expose it as a launch argument rather than editing the controller core.
