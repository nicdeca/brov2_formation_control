# Formation-control experiments

This directory documents the current reproducible experiment workflow for the
BlueROV2 formation-control project. It complements, but does not replace, the
repository-level README.

## Current canonical experiments

The actively maintained experiment paths are:

1. `two_robot_experiment.md`: normal two-robot SITL and real-water validation;
2. `two_robot_adaptive_mission.md`: dedicated two-robot SITL mission for the
   adaptive sensing-domain paper demonstration;
3. `five_robot_tree_experiment.md`: five-robot depth-two directed-tree SITL
   experiment.

`three_robot_experiment.md` is retained only as a legacy reference. Its old
launches used the pre-pool-alignment coordinate system and should not be used
without explicitly porting them.

## Recommended reading order

For a new machine or lab user:

1. `installation_and_build.md`;
2. `pool_coordinate_frame.md`;
3. either `simulation_setup.md` or `hardware_setup.md`;
4. `controller_parameters.md` and `launch_parameters.md`;
5. the experiment-specific page;
6. `experiment_checklist.md` during the run;
7. `troubleshooting.md` when something does not behave as expected;
8. `../experiment_logging.md` for recording/export/plotting details.

## Current experiment lifecycle

The controller phase and the recording phase are related, but deliberately not
identical:

```text
start split recorder before arming
        |
        +--> record initialization/bag
        |
launch controllers
        |
        v
   INITIALIZE
        |
        | all configured robots are close enough to their initialization
        | positions and sufficiently slow for the dwell time
        v
   FORMATION  ------------------> close initialization/bag
        |
        | experiment runner has discovered its subscribers
        v
mission_status = RUNNING -------> open mission/bag
        |
        | named formation commands + leader cmd_vel
        v
mission_status = COMPLETE ------> close mission/bag
        |
        v
export + plot initialization and mission separately
```

On an interrupted mission, the runner publishes `ABORTED` and the mission bag
is closed. If initialization never reaches `FORMATION`, interrupting the
recorder still leaves a clean initialization bag for diagnosis.

## Frames used by the current experiments

The SITL world has been re-anchored so that **PX4 local NED uses the same
intended pool origin/orientation as the real Marinarium frame**:

- origin: kitchen-side edge, lateral pool centerline, water surface;
- `+X_NED`: along the pool away from the kitchen;
- `+Y_NED`: across the pool;
- `+Z_NED`: downward.

The controller core still uses NWU/FLU, so the standard PX4-to-core conversion
remains

```text
p_core = diag(1, -1, -1) p_PX4_NED.
```

See `pool_coordinate_frame.md` for the exact simulated dimensions, current
workspace bounds and verification procedure.

## Current topology assumptions

Every follower has at most one parent. The directed edge notation used in the
documentation is

```text
follower -> parent
```

and the desired vector is always the **parent-minus-follower** position.

The phase manager is supervisory and centralized. After the handoff to
`FORMATION`, the follower control law remains decentralized over the configured
directed sensing tree.
