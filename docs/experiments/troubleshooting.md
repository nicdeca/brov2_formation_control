# Troubleshooting

## Phase remains `INITIALIZE`

Check:

```bash
ros2 topic echo /formation_control/experiment_phase --once
```

Inspect actual positions/speeds against the initialization targets.

The manager requires **all** configured robots to satisfy the position and
speed conditions continuously for the dwell time.

Model mismatch/buoyancy can create absolute-position offsets. The initialization
tolerance is intentionally loose.

## Formation commands appear in logs but robots do not react

Confirm the phase is actually `FORMATION`.

During `INITIALIZE`, followers may receive and store named formation commands,
but the initialization controller remains active.

## Named formation is received by only one follower

Check:

```bash
ros2 topic info /formation_control/desired_formation --verbose
```

For three robots there should be two follower subscriptions.

The experiment runners repeatedly publish the formation command to avoid
short-lived publisher discovery issues.

## Leader jumps back toward the spawn point at `FORMATION`

This was caused by a stale velocity-reference state initialized before the
absolute initialization maneuver.

The corrected leader node resets the stationary/velocity reference from the
actual current leader state on `INITIALIZE -> FORMATION`.

If this symptom returns, verify that the corrected leader node is the one
installed/built.

## `normalized image coordinates require strictly positive camera depth`

The parent is behind the camera according to the configured camera/frame
geometry.

Do not tune gains to hide this.

Verify:

- robot geometry;
- body/world conversions;
- camera mounting/extrinsics;
- parent-minus-follower sign.

## Robot oscillates next to a wall

Compare the desired formation equilibrium with the logged workspace bounds.

If the formation target itself lies beyond or too close to the wall, increasing
the formation gain will usually make the conflict worse.

Inspect:

```text
workspace_<robot>.pdf
workspace_relaxation_<robot>.pdf
formation_tracking_<edge>.pdf
```

## Reverse thruster limit appears with the wrong sign in plots

The logged reverse force limit is a positive magnitude. The plotted signed
lower bound must be:

```text
-reverse_limit
```

The current plotting script includes this correction.

## One SITL vehicle refuses to arm

PX4 may report:

```text
Arming denied: Resolve system health failures first
```

First verify the ROS/PX4 namespaces match the actual spawned vehicle IDs.

For two-robot SITL the default ordered pair is:

```text
itrl_rov_1
itrl_rov_2
```

Then inspect vehicle status, control mode, odometry, and QGroundControl health
messages.

## QGroundControl becomes very slow

Check for stale/repeated PX4/Gazebo processes:

```bash
pgrep -af px4
pgrep -af 'gz sim|gzserver|gzclient'
```

Check CPU load:

```bash
ps -eo pid,ppid,%cpu,%mem,etime,cmd \
  --sort=-%cpu | head -20
```

Stop leftover simulator instances before restarting a clean experiment.

## Plotter warns about more than 20 open figures

The paper plotter can create many figures for three-robot experiments.

This warning does not invalidate already-created figures, but the plotting
script should eventually close figures after saving if memory becomes an issue.
