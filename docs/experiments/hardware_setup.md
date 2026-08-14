# Real-water hardware setup

This page is a pre-experiment procedure. Do not copy SITL geometry blindly to
the real tank.

## 1. Verify robot identities

Confirm which physical vehicles are available and map them to namespaces:

```text
leader:   itrl_rov_?
follower: itrl_rov_?
```

The two-robot launch defaults to ordered names:

```text
leader   = itrl_rov_1
follower = itrl_rov_2
```

Override them explicitly at launch if the hardware pair differs.

## 2. Verify state-estimation streams

Before arming:

```bash
ros2 topic hz /itrl_rov_1/fmu/out/vehicle_odometry
ros2 topic hz /itrl_rov_2/fmu/out/vehicle_odometry
```

Both should be stable and fast enough for the controller.

Inspect one sample:

```bash
ros2 topic echo /itrl_rov_1/fmu/out/vehicle_odometry --once
ros2 topic echo /itrl_rov_2/fmu/out/vehicle_odometry --once
```

## 3. Verify frame conventions

The controller expects core NWU positions and body FLU velocities after the
state adapter.

Before enabling formation control, put the robots in a known geometry and
verify:

```text
p_parent - p_follower
```

has the expected sign and magnitude.

This is a mandatory hardware sanity check.

## 4. Verify camera geometry

For the follower:

- verify the camera mounting transform;
- verify horizontal/vertical FoV parameters;
- verify that the parent appears at positive camera depth;
- verify normalized image coordinates have the expected sign.

Do not proceed if the controller reports:

```text
normalized image coordinates require strictly positive camera depth
```

## 5. Measure the real tank workspace

Update the launch parameters for:

```text
workspace_physical_lower
workspace_physical_upper
workspace_conservative_lower
workspace_conservative_upper
```

The physical controller boundary must already include a safe robot-center
clearance from the actual pool wall. The conservative boundary must lie
strictly inside it.

## 6. Conservative first launch

For the first wet test:

```bash
ros2 launch formation_control_ros two_robot_experiment.launch.py \
  dry_run:=false \
  leader_reference_mode:=velocity \
  position_gain:=2.0 \
  formation_gain:=2.0 \
  workspace_barrier_enabled:=true \
  workspace_adaptive:=true
```

Do not start immediately with the most aggressive gains/profile simply
because they worked in SITL.

## 7. Safety

Before arming, agree on:

- who controls QGroundControl;
- who watches the physical robots;
- how to immediately leave Offboard;
- how to disarm;
- when the run must be aborted.

The experiment script is not an emergency-stop mechanism. QGroundControl/manual
intervention must remain available throughout the run.
