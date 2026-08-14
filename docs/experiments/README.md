# Formation-control experiments

This directory documents the reproducible experiment workflow for the
BlueROV2 formation-control project.

It intentionally does **not** replace the repository-level README.

## Recommended order

For a new machine or lab user:

1. Read `installation_and_build.md`.
2. Read either `simulation_setup.md` or `hardware_setup.md`.
3. Read `controller_parameters.md`.
4. Follow `two_robot_experiment.md` or `three_robot_experiment.md`.
5. Keep `experiment_checklist.md` open during the run.
6. Use `troubleshooting.md` if something does not behave as expected.

## Current experiment architecture

The experiment lifecycle is

```text
launch controllers
      |
      v
INITIALIZE
      |
      | all robots sufficiently close to their initialization targets
      | and sufficiently slow for the configured dwell time
      v
FORMATION
      |
      +-- online named formation changes
      +-- leader velocity commands / cmd_vel
      +-- sensing-domain adaptation
      +-- workspace-wall adaptation
```

The phase transition is supervisory and centralized: the phase manager checks
all robots before publishing `FORMATION`.

After the transition, the formation controller is decentralized over the
configured sensing graph. In the two-robot case there is one leader-follower
edge. In the three-robot case both followers independently use the leader as
their parent.

## Important status

The three-robot SITL configuration has been validated extensively.

The two-robot SITL `cautious` and `full` profiles have also been validated.

Before the first real-water experiment, the following values must be verified
against the actual Marinarium setup:

- absolute initialization positions;
- real tank bounds in the controller's `core_nwu` frame;
- camera extrinsics and FoV parameters;
- state-estimation frames and sign conventions;
- PX4 force/torque normalization and command limits;
- robot namespaces / IDs;
- available physical robots and their health.
