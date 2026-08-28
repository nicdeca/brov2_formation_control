# Real-water hardware setup

This is the pre-experiment procedure for the current pool-aligned controller.
Do not copy SITL spawn poses to hardware; use the shared PX4 pool frame and the
actual physical robot positions.

## 1. Confirm the pool frame

The intended raw PX4 local NED frame is:

```text
origin: kitchen-side edge, lateral centerline, water surface
+X:     away from the kitchen
+Y:     across the pool
+Z:     downward
```

The reported approximate real-pool extent is `x in [0,9] m`,
`y in [-2.5,2.5] m`, `z in [0,3] m` in NED.

Before the first run, physically verify the sign of the lateral `+Y` direction.
See `pool_coordinate_frame.md`.

## 2. Verify robot identities and dynamics presets

Currently characterized vehicles:

```text
glub    -> heavy_tube
splash  -> heavy_tube
bubble  -> standard
```

The two-robot launch accepts physical names directly and defaults the two
dynamics selections to `auto`.

Example:

```bash
ros2 launch formation_control_ros two_robot_experiment.launch.py \
  leader:=splash \
  follower:=glub \
  dry_run:=false \
  leader_reference_mode:=velocity \
  workspace_barrier_enabled:=true \
  workspace_adaptive:=true
```

Unknown names fail in `auto` mode; choose an explicit dynamics preset for an
unmapped robot.

## 3. Verify state-estimation streams

Before arming:

```bash
ros2 topic hz /splash/fmu/out/vehicle_odometry
ros2 topic hz /glub/fmu/out/vehicle_odometry
```

Inspect raw samples:

```bash
ros2 topic echo /splash/fmu/out/vehicle_odometry --once
ros2 topic echo /glub/fmu/out/vehicle_odometry --once
```

Check absolute pool coordinates and relative geometry, not only that messages
exist.

## 4. Verify controller/core frame conversion

The controller converts PX4 NED/FRD to core NWU/FLU. In the core:

```text
x_core =  x_NED
y_core = -y_NED
z_core = -z_NED
```

Put the robots in a known geometry and verify the parent-minus-follower vector
has the expected sign and magnitude.

## 5. Verify camera geometry

For every follower:

- verify camera mounting/extrinsics;
- verify horizontal and vertical FoV parameters;
- verify the parent has strictly positive camera depth;
- verify normalized image-coordinate signs.

Do not tune gains to hide a frame/camera error.

## 6. Verify workspace bounds

The current maintained launches use the common safe subset

```text
physical:      [0.300, -1.975, -2.155] ... [7.100, 1.975, 0.225]
conservative:  [0.450, -1.825, -1.955] ... [6.950, 1.825, -0.325]
```

in core NWU. These are transformed bounds inherited from the validated SITL
workspace and are not the full real-pool dimensions.

Before treating them as final hardware safety limits, verify the actual
robot-center clearance from walls, floor and surface. In particular, decide
explicitly whether the physical `z` upper bound should permit the vehicle
center above the nominal water-surface plane.

## 7. Start the split recorder before arming

For a `splash` leader and `glub` follower:

```bash
scripts/record_formation_experiment.sh \
  --name two_robot_water_01 \
  --robots splash,glub \
  --edge glub:splash
```

Initialization is recorded separately from the mission. If initialization
fails, interrupt the recorder and use the initialization export/plots rather
than repeatedly attempting to start the mission blind.

## 8. Conservative first profile

Start with the `cautious` profile:

```bash
python scripts/run_two_robot_experiment.py \
  --leader splash \
  --profile cautious
```

Only move to `full` or `challenging` after the actual hardware setup has been
validated.

## 9. Safety

Before arming, agree on:

- who controls QGroundControl;
- who watches the physical robots;
- how to immediately leave Offboard;
- how to disarm;
- when the run must be aborted.

The experiment runner is not an emergency-stop mechanism. QGroundControl/manual
intervention must remain available throughout the run.
