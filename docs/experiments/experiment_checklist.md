# Poolside / experiment checklist

Use this as the short operational checklist.

## Before launching controllers

- [ ] Correct robots are powered or spawned.
- [ ] Simulator/hardware namespaces match the controller launch exactly.
- [ ] QGroundControl sees every intended vehicle.
- [ ] No unresolved PX4 health failures.
- [ ] Odometry streams are present and stable.
- [ ] Raw PX4 pool-frame origin/sign check completed.
- [ ] Core NWU relative geometry is physically plausible.
- [ ] Camera sees each parent with positive depth.
- [ ] Workspace bounds are appropriate for this run.
- [ ] Correct dynamics preset is selected (`gazebo` for SITL).
- [ ] Someone is assigned to QGC/manual safety intervention for hardware.

## Start software

- [ ] PX4/state-estimation stack running.
- [ ] DDS/XRCE bridge running.
- [ ] Controller launch running.
- [ ] Controller logs show intended robot names/configurations.
- [ ] Workspace barrier/adaptation status is correct.
- [ ] **Split recorder started before arming.**
- [ ] Experiment runner started and waiting for `FORMATION`.

## Arm / initialize

- [ ] Arm leader and follower(s).
- [ ] Put every robot in Offboard.
- [ ] Verify controller outputs are active.
- [ ] Verify phase is `INITIALIZE`.
- [ ] Robots move toward the intended initialization geometry.
- [ ] No unexpected workspace/sensing physical-margin violation.
- [ ] Phase transitions to `FORMATION`.
- [ ] Initialization bag closes.
- [ ] Leader reference does not jump at the phase transition.

## Mission start

- [ ] Runner discovers expected formation/cmd_vel subscribers.
- [ ] `mission_status` changes to `RUNNING`.
- [ ] Mission bag starts before the first command.

## During experiment

- [ ] Physical robots remain in safe workspace.
- [ ] Parent remains visible to each follower camera.
- [ ] No persistent fallback warnings.
- [ ] No unexpected oscillation against a wall.
- [ ] QGC/manual intervention remains available.

## End

- [ ] Runner reports `COMPLETE`, or `ABORTED` if interrupted.
- [ ] Leader command is zero.
- [ ] Mission bag closes cleanly.
- [ ] Robots are disarmed when appropriate.
- [ ] `initialization/` and, for a started mission, `mission/` exist.
- [ ] Run `python scripts/export_experiment.py "$RUN"` if export was not automatic.
- [ ] Run `uv run python scripts/plot_experiment.py "$RUN"` if plots were not automatic.
- [ ] Initialization NPZ/plots exist.
- [ ] Mission NPZ/plots exist if the mission started.
- [ ] Physical sensing/workspace margins are inspected before accepting the run.
