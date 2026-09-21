# ROS 2 launch-file parameters

This page covers the current canonical simulator and experiment launch files.

## Dynamics configuration

| Value | Use |
|---|---|
| `gazebo` | PX4/Gazebo SITL vehicle model |
| `standard` | characterized standard laboratory BlueROV2 |
| `heavy_tube` | characterized heavy-tube laboratory BlueROV2 |
| `auto` | ROS-side name lookup for known physical robots only |

`auto` maps `glub` and `splash` to `heavy_tube`, and `bubble` to `standard`.
Unknown names in `auto` mode raise an error.

## `multi_bluerov2_sim.launch.py`

Launches 1--6 PX4 SITL BlueROV2 instances in the pool-aligned Gazebo world.

| Argument | Default | Meaning |
|---|---|---|
| `px4_dir` | auto-detected checkout | path to `PX4-Autopilot` |
| `world` | `kth_marinarium_docking` | Gazebo world name without `.sdf` |
| `robot_count` | `3` | number of robots, 1--6 |
| `spawn_delay` | `5.0` | delay [s] between PX4 instances |
| `robot_1_name` ... `robot_6_name` | `itrl_rov_1` ... `itrl_rov_6` | PX4/ROS namespaces |
| `rov_1_pose` ... `rov_6_pose` | launch defaults | internal Gazebo ENU pose `x,y,z,r,p,yaw` |

Example with hardware-like names:

```bash
ros2 launch formation_control_ros multi_bluerov2_sim.launch.py \
  robot_count:=2 \
  robot_1_name:=splash \
  robot_2_name:=glub \
  px4_dir:=$PX4_AUTOPILOT_DIR
```

## `two_robot_experiment.launch.py`

Normal two-robot SITL/hardware controller launch.

| Argument | Default | Meaning |
|---|---|---|
| `leader` | `itrl_rov_1` | leader namespace |
| `follower` | `itrl_rov_2` | follower namespace |
| `dt` | `0.02` | controller period [s] |
| `dry_run` | `true` | compute without commanding actuation |
| `leader_reference_mode` | `stationary` | `stationary`, `velocity`, or `trajectory` |
| `leader_robot_configuration` | `auto` | dynamics selection |
| `follower_robot_configuration` | `auto` | dynamics selection |
| `position_gain` | `2.0` | leader position-potential gain |
| `formation_gain` | `2.0` | follower formation-potential gain |
| `virtual_linear_gain` | `0.55` | translational virtual/backstepping gain |
| `virtual_angular_gain` | `0.80` | rotational virtual/backstepping gain |
| `command_filter_linear_bandwidth` | `3.0` | translational command-filter bandwidth |
| `command_filter_angular_bandwidth` | `4.0` | rotational command-filter bandwidth |
| `alpha_gain` | `1.8` | CLF dissipation gain |
| `workspace_barrier_enabled` | `true` | enable workspace barriers |
| `workspace_adaptive` | `true` | enable workspace-domain enlargement |

Hardware example:

```bash
ros2 launch formation_control_ros two_robot_experiment.launch.py \
  leader:=splash follower:=glub \
  dry_run:=false \
  leader_reference_mode:=velocity
```

SITL with those same names must override both configurations:

```bash
ros2 launch formation_control_ros two_robot_experiment.launch.py \
  leader:=splash follower:=glub \
  leader_robot_configuration:=gazebo \
  follower_robot_configuration:=gazebo \
  dry_run:=false \
  leader_reference_mode:=velocity
```

## `two_robot_adaptive_mission.launch.py`

Dedicated SITL-only paper mission. It fixes `robot_configuration = gazebo` and
uses its own tighter sensing-domain/adaptation tuning.

| Argument | Default |
|---|---:|
| `leader` | `itrl_rov_1` |
| `follower` | `itrl_rov_2` |
| `dt` | `0.02` |
| `dry_run` | `true` |
| `gazebo_timer` | `false` |
| `position_gain` | `2.0` |
| `formation_gain` | `1.4` |
| `virtual_linear_gain` | `0.55` |
| `virtual_angular_gain` | `0.80` |
| `command_filter_linear_bandwidth` | `3.0` |
| `command_filter_angular_bandwidth` | `4.0` |
| `alpha_gain` | `0.8` |
| `d_max_conservative` | `2.4` |
| `alpha_h_conservative` | `0.45` |
| `alpha_v_conservative` | `0.45` |
| `relaxation_recovery_gain` | `0.8` |
| `relaxation_barrier_gain` | `0.20` |
| `relaxation_domain_margin_ratio` | `0.02` |
| `relaxation_activation_on_ratio` | `0.001` |
| `relaxation_activation_off_ratio` | `0.15` |
| `workspace_barrier_enabled` | `true` |
| `workspace_adaptive` | `true` |

If `gazebo_timer:=true`, run `run_two_robot_adaptive_mission.py` with
`--gazebo-timer` as well.

## `five_robot_tree_experiment.launch.py`

Current five-robot depth-two directed-tree SITL launch:

```text
4 -> 2 -> 1
5 -> 3 -> 1
```

All controllers explicitly use the `gazebo` dynamics preset.

| Argument | Default |
|---|---:|
| `robot_1` ... `robot_5` | `itrl_rov_1` ... `itrl_rov_5` |
| `dt` | `0.02` |
| `dry_run` | `true` |
| `gazebo_timer` | `false` |
| `leader_reference_mode` | `stationary` |
| `position_gain` | `2.0` |
| `formation_gain` | `1.0` |
| `virtual_linear_gain` | `1.0` |
| `virtual_angular_gain` | `1.2` |
| `command_filter_linear_bandwidth` | `10.0` |
| `command_filter_angular_bandwidth` | `10.0` |
| `alpha_gain` | `3.0` |
| `workspace_barrier_enabled` | `true` |
| `workspace_adaptive` | `true` |

Canonical active-control launch:

```bash
ros2 launch formation_control_ros five_robot_tree_experiment.launch.py \
  dry_run:=false \
  leader_reference_mode:=velocity \
  workspace_barrier_enabled:=true \
  workspace_adaptive:=true
```

The current five-robot mission runner has no `--gazebo-timer` CLI option, so
leave launch timing on its normal wall-time path for the standard scripted run.
