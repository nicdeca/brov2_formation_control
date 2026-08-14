# Three-robot SITL experiment

## Topology

```text
itrl_rov_1 leader

itrl_rov_2 -> itrl_rov_1
itrl_rov_3 -> itrl_rov_1
```

The phase manager releases `FORMATION` only after all three robots satisfy the
initialization handoff condition.

After release, the two follower controllers operate independently.

## Controller launch

```bash
ros2 launch formation_control_ros three_robot_experiment.launch.py \
  dry_run:=false \
  leader_reference_mode:=velocity \
  workspace_barrier_enabled:=true \
  workspace_adaptive:=true
```

The validated SITL tuning used geometric gains around:

```text
position_gain = 3.0
formation_gain = 3.0
```

## Recorder

```bash
scripts/record_formation_experiment.sh \
  --name three_robot_experiment \
  --robots itrl_rov_1,itrl_rov_2,itrl_rov_3 \
  --edge itrl_rov_2:itrl_rov_1 \
  --edge itrl_rov_3:itrl_rov_1
```

## Runner

```bash
python scripts/run_three_robot_experiment.py
```

The final validated scripted experiment uses clearly distinct formation
references and a large leader excursion while remaining inside the configured
workspace.

## What to inspect

For both follower edges:

- actual vs desired relative position components;
- formation-error norm;
- sensing-domain relaxation.

For all three robots:

- workspace position vs conservative/adaptive/physical bounds;
- workspace relaxation;
- thruster forces and signed limits;
- controller timing;
- actuation feasibility.

For the leader:

- actual vs reference position;
- position/velocity tracking-error norms.

## Known limitation

With the star topology there is no explicit follower-follower sensing/collision
edge between robots 2 and 3. Prescribed experiment formations must therefore
remain safely separated.
