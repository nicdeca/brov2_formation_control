# formation_control_ros

ROS 2 (`rclpy` / `ament_python`) integration for the ROS-independent
`formation_control` package.

## Where this package should live

Keep it **in the same Git repository** as the controller core, but **not**
inside `src/formation_control`.

Recommended repository layout:

```text
brov2_formation_control/
├── pyproject.toml
├── src/
│   └── formation_control/          # ROS-independent core
├── examples/
├── tests/
└── ros2/
    └── formation_control_ros/      # this ROS 2 ament package
```

This keeps the mathematical/control package importable without ROS while
versioning the ROS adapter together with it.

If the whole repository is checked out below a ROS 2 workspace `src/`
directory, `colcon` can discover the nested `package.xml`.  Alternatively,
symlink only `ros2/formation_control_ros` into your ROS 2 workspace.

## Architecture

```text
MoCap / future onboard sensing
             |
             v
      ROS 2 state adapter
             |
             v
     frame_conventions.py
             |
             v
   formation_control core
   - potentials
   - command filter
   - adaptive domain
   - CLF-QP
             |
             v
       body wrench FLU
             |
             v
       PX4 frame adapter
             |
             v
VehicleThrustSetpoint / VehicleTorqueSetpoint
```

The controller core has no `rclpy`, ROS message, or PX4 dependency.

## Frame contract

This is intentionally explicit because the previous workspace contained
implicit sign changes.

| Quantity | Core | Initial MoCap assumption | PX4 |
|---|---|---|---|
| inertial frame | NWU project frame | already project NWU by default | NED-related autopilot internals |
| body frame | FLU | ROS child FLU | FRD |
| quaternion storage | `[qw,qx,qy,qz]` | ROS message `[qx,qy,qz,qw]` | handled at ROS boundary |
| generalized velocity | body FLU | configurable `body`/`world` | not passed directly |
| wrench | body FLU | n/a | body FRD normalized setpoint |

`frame_conventions.py` is the only place where axis transforms should be
introduced.

### MoCap world frame

Parameter:

```text
mocap_world_frame = core_nwu | ros_enu
```

The default `core_nwu` matches the interpretation in the student's ROS 2
workspace.  If the MoCap bridge is standard ENU, set `ros_enu`.

### Odometry twist

Parameter:

```text
odom_twist_frame = body | world
```

The default is `body`, again matching the previous workspace.  Verify this
against the actual MoCap bridge before experiments.

### Core -> PX4 body wrench

The core uses FLU and PX4 uses FRD:

```text
[x, y, z]_FRD = [x, -y, -z]_FLU
```

The same proper rotation is applied to body torque.

## Why wrench-space control is the ROS default

The validated core supports both thruster-space and wrench-space CLF-QPs.
The existing PX4 offboard interface accepts body thrust/torque rather than the
eight individual BlueROV thruster forces.  Therefore the ROS 2 package defaults
to

```text
control_space: wrench
```

using the exact achievable BlueROV2 wrench polytope.  PX4 then performs the
downstream allocation.

Direct eight-thruster offboard actuation should be implemented later as a
separate interface if exact reproduction of the thruster-space QP decision is
required.  It should not be mixed silently with the body-wrench interface.

The dynamics model accepts the presets `gazebo`, `standard`, and
`heavy_tube`. The ROS parameter `robot_configuration` additionally accepts
`auto`. In `auto` mode the current laboratory mapping is

```text
glub    -> heavy_tube
splash  -> heavy_tube
bubble  -> standard
```

Unknown robot names raise an error in `auto` mode rather than silently selecting
a model. The real two-robot launch defaults both robots to `auto` and still
allows explicit per-robot overrides. The five-robot SITL launch explicitly
selects `gazebo` for every controller.

### Optional Gazebo-synchronized timing

The five-robot tree launch uses wall-clock timers by default. For simulations
running below real time, the complete experiment can instead follow Gazebo
simulation time:

```bash
ros2 launch formation_control_ros five_robot_tree_experiment.launch.py \
    gazebo_timer:=true \
    leader_reference_mode:=velocity
```

This option starts a Gazebo-to-ROS `/clock` bridge and enables `use_sim_time`
for the leader and follower controllers. The PX4 offboard heartbeats and the
initialization phase manager deliberately remain on wall time so clock startup
or a paused simulation cannot suppress heartbeat or phase messages. Run the
mission profile with the matching option so its command and settling durations
use Gazebo time:

```bash
python scripts/run_five_robot_tree_experiment.py \
    --leader itrl_rov_1 \
    --profile full \
    --gazebo-timer
```

If Gazebo is paused, controller and mission time pause; heartbeat and phase
publication continue. Omitting the options preserves the existing wall-clock
behavior.


## Current launch files and arguments

The canonical simulator launch is

```text
multi_bluerov2_sim.launch.py
```

and the current controller launches are

```text
two_robot_experiment.launch.py
five_robot_tree_experiment.launch.py
```

A complete table of their user-facing launch arguments, defaults, dynamics
preset behavior, and canonical commands is maintained in
`docs/experiments/launch_parameters.md`. Keep that page synchronized with the
launch files whenever an argument is added or renamed.

## ROS 2 nodes

### `leader_controller`

Uses the same second-order CLF-QP as the followers.

Reference modes:

```text
stationary
velocity
trajectory
```

`stationary` holds the first received MoCap position and attitude.

`velocity` subscribes to a standard ROS 2 `geometry_msgs/TwistStamped`.
A first-order velocity-reference filter generates consistent `p_r`, `v_r`,
and `a_r`.

`trajectory` subscribes to
`trajectory_msgs/MultiDOFJointTrajectoryPoint` and uses the first transform,
velocity and acceleration.  The current leader controller uses translational
`p_r`, `v_r`, `a_r` while holding the initial attitude.

### `follower_controller`

Subscribes to:

```text
/mocap/<robot>/odom
/mocap/<parent>/odom
```

and runs one directed parent edge.  The desired relative-position parameter
uses the **core convention**

```text
desired_relative_position = p_parent - p_follower
```

not the opposite convention used in the old student controller.

The sensing-domain adaptation is derivative-free with respect to the parent
motion: the auxiliary barrier potential is activated near the adaptive-domain
margin and does not require the parent velocity or `h_c_dot`. This preserves
the intended relative-position-only sensing interface of the follower
controller.

### `offboard_heartbeat_wrench`

ROS 2 `rclpy` heartbeat setting

```text
OffboardControlMode.thrust_and_torque = True
```

at 20 Hz, matching the previous PX4 interface.

## Diagnostics

Each controller namespace publishes:

```text
formation_control/fallback
formation_control/slack
formation_control/required_slack
formation_control/actuation_margin
formation_control/thruster_utilization
formation_control/minimum_physical_margin
formation_control/domain_relaxation
formation_control/conservative_constraint_values
```

`required_slack` and `actuation_margin` can be `NaN` in wrench-space control
because the current closed-form feasibility diagnostic is specific to a
box-constrained control variable.

## Build

The ROS interpreter used by `colcon` must also be able to import the
ROS-independent `formation_control` package.

For example, in the Python environment used for the ROS 2 workspace:

```bash
pip install -e /path/to/brov2_formation_control
```

Then:

```bash
cd ~/ros2_ws
colcon build --symlink-install --packages-select formation_control_ros
source install/setup.bash
```

If you use `uv`, make sure the ROS 2 Python packages (`rclpy`, `px4_msgs`, ...)
and the `formation_control` editable install are visible to the same Python
interpreter used by `colcon`.

## First smoke test

The first milestone is deliberately only one leader and one follower:

```bash
ros2 launch formation_control_ros two_robot_mocap.launch.py \
    leader:=Splash \
    follower:=itrl_rov_1
```

The leader holds its initial pose.  The follower uses MoCap relative state and
converges to the configured one-parent formation.

Before moving to the seven-robot graph, verify:

1. the MoCap frame assumptions;
2. body wrench signs using small commands;
3. the ROS node output against one stored pure-Python controller sample;
4. controller/filter state evolution at the configured `dt`;
5. PX4 offboard gating and zero-command behavior.

## Safety note

The current integration fallback publishes zero wrench, matching the previous
simulation-oriented ROS 2 workspace.  Do **not** use this as the final hardware
failsafe.  Before Marinarium experiments, connect controller failure/lost
sensing to the vehicle's tested hold/disarm/failsafe strategy.


## Current BlueROV SITL odometry

The current multi-BlueROV simulator publishes:

```text
/itrl_rov_1/fmu/out/vehicle_odometry
/itrl_rov_2/fmu/out/vehicle_odometry
/itrl_rov_3/fmu/out/vehicle_odometry
```

as `px4_msgs/msg/VehicleOdometry`.

Observed values are:

```text
pose_frame:     1  -> PX4 NED
velocity_frame: 1  -> PX4 NED
```

PX4's quaternion is `[w,x,y,z]` and maps body FRD to the pose frame.
`angular_velocity` is body FRD.  The adapter converts this to the core contract:

```text
position:         NWU
orientation:      body FLU -> NWU, [w,x,y,z]
linear velocity:  body FLU
angular velocity: body FLU
```

The conversion is rejected if PX4 reports a different pose or velocity frame.

Current SITL smoke test:

```bash
ros2 launch formation_control_ros two_robot_px4_sitl.launch.py \
    leader:=itrl_rov_1 \
    follower:=itrl_rov_2
```


## Dry-run smoke test

The SITL smoke-test launch defaults to `dry_run:=true`. In this mode the
controller evaluates normally and publishes diagnostics, but does not publish
the computed wrench to PX4.

```bash
ros2 launch formation_control_ros two_robot_px4_sitl.launch.py \
    leader:=itrl_rov_1 \
    follower:=itrl_rov_2 \
    dry_run:=true
```

After verifying frames, margins, and controller outputs, enable actuation with
`dry_run:=false`.
