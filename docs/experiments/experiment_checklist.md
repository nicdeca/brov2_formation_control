# Poolside / experiment checklist

Use this as the short operational checklist.

## Before launching controllers

- [ ] Correct robots are powered / spawned.
- [ ] Robot namespaces match the launch arguments.
- [ ] QGroundControl sees every intended vehicle.
- [ ] No unresolved PX4 health failures.
- [ ] Odometry streams are present and stable.
- [ ] Frame/sign sanity check completed.
- [ ] Relative geometry is physically plausible.
- [ ] Camera sees the parent with positive depth.
- [ ] Real workspace bounds have been verified for hardware.
- [ ] Someone is assigned to QGC/manual safety intervention.

## Start the software

- [ ] State/PX4 stack running.
- [ ] DDS/XRCE bridge running.
- [ ] Controller launch running.
- [ ] Controller logs show the intended robot names.
- [ ] Workspace barrier status is correct.
- [ ] Recorder started.
- [ ] Experiment runner started and waiting for `FORMATION`.

## Arm / initialize

- [ ] Arm leader.
- [ ] Arm follower(s).
- [ ] Put every robot in Offboard.
- [ ] Verify controller outputs are active.
- [ ] Verify phase is `INITIALIZE`.
- [ ] Robots move toward the intended initial geometry.
- [ ] Phase transitions to `FORMATION`.
- [ ] Leader reference does not jump at the phase transition.

## During experiment

- [ ] Physical robots remain in safe workspace.
- [ ] Parent remains visible to follower camera.
- [ ] No persistent fallback warnings.
- [ ] No unexpected oscillations against a wall.
- [ ] QGC/manual intervention remains available.

## End

- [ ] Experiment runner reports completion or is interrupted safely.
- [ ] Leader command is zero.
- [ ] Robots are disarmed.
- [ ] Recorder stopped cleanly with Ctrl-C.
- [ ] Bag exists and is readable.
- [ ] Export NPZ.
- [ ] Generate plots.
- [ ] Inspect physical sensing/workspace margins before accepting the run.
