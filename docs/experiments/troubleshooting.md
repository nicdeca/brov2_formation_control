# Troubleshooting

## Phase remains `INITIALIZE`

Check:

```bash
ros2 topic echo /formation_control/experiment_phase --once
```

Inspect the separate initialization dataset rather than guessing from the
mission plots:

```bash
RUN=outputs/experiments/<run>
python scripts/export_initialization_bag.py "$RUN"
python scripts/plot_initialization_experiment.py \
  "$RUN/initialization/initialization_history.npz" --save
```

The phase manager requires **all** configured robots to satisfy the position
and speed conditions continuously for the dwell time.

## `FileNotFoundError: .../RUN/initialization/bag`

`RUN` in examples is a shell variable/placeholder, not a literal directory.
Use an actual run path:

```bash
ls -dt outputs/experiments/* | head
RUN=$(ls -dt outputs/experiments/* | head -n 1)
python scripts/export_initialization_bag.py "$RUN"
```

## Mission bag never starts

Check:

```bash
ros2 topic echo /formation_control/mission_status --once
```

The maintained runners publish `WAITING`, then `RUNNING` only after
`FORMATION` and subscriber discovery. The split recorder starts
`mission/bag` on `RUNNING`.

If using an old runner that does not publish `mission_status`, the mission
recorder will wait forever. Port the runner or use a maintained one.

## Runner waits forever for leader `cmd_vel`

The runner's `--leader` must match the controller namespace. For example, if
the controller leader is `splash`:

```bash
python scripts/run_two_robot_experiment.py \
  --leader splash \
  --profile cautious
```

Without `--leader splash`, the default runner waits on
`/itrl_rov_1/formation_control/cmd_vel`.

## QGroundControl says no Offboard signal / one SITL robot is not ready

First compare simulator and controller names. A common mismatch is spawning
`splash,bubble` and launching the second controller as `glub`.

Check:

```bash
ros2 topic list | grep '/fmu/in/offboard_control_mode'
ros2 topic hz /splash/fmu/in/offboard_control_mode
ros2 topic hz /glub/fmu/in/offboard_control_mode
```

## SITL uses a real-hardware dynamics model

`two_robot_experiment.launch.py` defaults dynamics to `auto`. If the SITL
names are `splash`, `glub`, or `bubble`, `auto` selects the corresponding real
model. Override explicitly:

```text
leader_robot_configuration:=gazebo
follower_robot_configuration:=gazebo
```

## PX4 positions still look like `z ~ -95`

The pool-aligned world is not the one being launched, or stale Gazebo resources
are being used. In the current setup raw PX4 NED depth should be a small
positive number in metres, not an absolute Gazebo altitude near `-95 m`.

Verify the updated `kth_marinarium*.sdf` resources and restart all PX4/Gazebo
processes.

## Formation commands appear in logs but robots do not react

Confirm the phase is `FORMATION`. During `INITIALIZE`, the initialization
controller remains active even if a named formation command has been received.

## Named formation is received by fewer followers than expected

```bash
ros2 topic info /formation_control/desired_formation --verbose
```

The runner checks the subscription count before starting the mission.

## Leader jumps back toward an old spawn/reference at `FORMATION`

The leader reference must be reset from the actual current state at the
`INITIALIZE -> FORMATION` handoff. If the symptom returns, verify the current
leader node is built/installed rather than a stale workspace copy.

## `normalized image coordinates require strictly positive camera depth`

The parent is behind the camera according to the configured geometry. Verify:

- parent-minus-follower sign;
- body/world conversion;
- camera extrinsics;
- actual physical camera orientation.

Do not increase gains to hide this error.

## Robot oscillates next to a wall

Compare actual position, desired formation equilibrium and workspace bounds.
If the desired equilibrium conflicts with the wall, increasing formation gain
usually makes the conflict worse.

Inspect initialization and mission workspace plots separately.

## Sensing relaxation exceeds one

The current normalized sensing enlargement state is not clipped at one.
`s = 1` corresponds to the adaptive zero set reaching the physical limit.
Interpret `s > 1` together with `minimum_physical_margin`; do not judge safety
from the relaxation state alone.

## Reverse thruster limit appears with the wrong sign

The logged reverse force limit is a positive magnitude. The plotted signed
lower limit is `-reverse_limit`. The current mission plotter applies this sign.

## QGroundControl becomes very slow

Check for stale/repeated simulator processes:

```bash
pgrep -af px4
pgrep -af 'gz sim|gzserver|gzclient'
ps -eo pid,ppid,%cpu,%mem,etime,cmd --sort=-%cpu | head -20
```

Stop leftover simulator instances before restarting.

## Automatic post-processing fails after a valid recording

The bags are still useful. Export manually from the correct environment:

```bash
source setup_ros2.sh
python scripts/export_initialization_bag.py "$RUN"
python scripts/export_formation_bag.py "$RUN" --phase mission
```

Then run the mission plotter through `uv` if needed:

```bash
uv run python scripts/plot_formation_experiment.py \
  "$RUN/mission/formation_history.npz" --paper --paper-quality --save
```
