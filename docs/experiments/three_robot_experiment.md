# Three-robot experiment — legacy reference

> **Status:** not part of the current canonical experiment workflow.

The historical three-robot launches used a star topology

```text
2 -> 1
3 -> 1
```

and were developed before the pool-frame re-anchoring. The archived launch
files still contain old absolute coordinates around `z = -95 m` and old
workspace bounds. Running them unchanged with the current pool-aligned Gazebo
world is unsafe/inconsistent.

The current maintained scaling experiment is the five-robot depth-two tree in
`five_robot_tree_experiment.md`. For normal two-robot hardware/SITL work, use
`two_robot_experiment.md`.

If a three-robot experiment is needed again, port it explicitly by:

1. expressing all absolute initialization positions in the current pool-aligned
   core NWU frame;
2. rotating all parent-minus-follower formation vectors consistently;
3. using the current shared workspace bounds;
4. selecting `gazebo` explicitly for SITL dynamics;
5. adding `mission_status` publication to the runner so split recording works;
6. validating all formations against the current tank geometry before arming.

Do not use the old `three_robot_px4_sitl.launch.py`,
`three_robot_sim_experiment.launch.py`, or old three-robot mission runner as a
shortcut around this port.
