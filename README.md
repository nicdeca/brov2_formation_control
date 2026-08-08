# Underwater Formation Control

Distributed formation control for multiple BlueROV2 Heavy underwater vehicles under sensing and actuation constraints.

The project combines:

- directed follower-to-parent sensing;
- distance and camera field-of-view constraints;
- recentered logarithmic barrier potentials;
- command-filtered backstepping;
- smooth norm saturation of the virtual twist;
- actuator-constrained CLF-QP control directly in thruster space;
- adaptive enlargement of conservative sensing domains toward the physical limits;
- long-horizon multi-robot validation with moving leaders;
- simulation-only robustness tests with uncertainty, current, noise, and delay.

<p align="center">
  <img src="docs/media/formation_animation.gif"
       alt="Seven BlueROV2 vehicles maintaining a directed formation while the leader follows a three-dimensional trajectory"
       width="850">
</p>

The animation above is generated from the seven-robot trajectory validation example. See [Generating the README animation](#generating-the-readme-animation).

## Overview

Each follower observes a unique parent through an onboard forward-looking camera. The sensing graph is therefore directed, with the convention

```text
observer -> parent
```

and is assumed to form a rooted tree.

For each follower, the controller combines a formation objective with distance and field-of-view constraints. The same second-order control architecture is used for the leader and the followers:

```text
configuration potential
        |
        v
generalized gradient
        |
        v
smooth norm-saturated virtual twist
        |
        v
command filter
        |
        v
backstepping CLF
        |
        v
actuator-constrained CLF-QP
        |
        v
BlueROV2 Heavy thruster forces
```

The leader differs only in the upper-level objective. It can track either:

- a full position/velocity/acceleration trajectory; or
- a velocity command, converted to a smooth position/velocity/acceleration reference by a command filter.

## Control architecture

### BlueROV2 dynamics

Each vehicle is modeled with the standard six-degree-of-freedom marine dynamics

```text
eta_dot = J(eta) nu
M nu_dot = w + d - C(nu) nu - D(nu) nu - g(eta)
```

with the BlueROV2 Heavy thruster allocation

```text
w = B f
```

and physical thruster bounds enforced directly by the CLF-QP.

### Formation and sensing constraints

For every directed sensing edge, the follower regulates the desired relative position while maintaining:

- minimum inter-vehicle distance;
- maximum sensing range;
- positive camera depth;
- horizontal field of view;
- vertical field of view.

The conservative sensing domain is strictly contained in the physical sensing domain.

### Recentered barriers

The constraint potentials use recentered logarithmic barriers of the form

```text
B_bar(h; h_d) = -log(h / h_d) + h / h_d - 1
```

so that both the barrier value and its derivative with respect to the scalar constraint function vanish at the desired value.

### Smooth virtual-twist saturation

The virtual backstepping correction is smoothly saturated by norm, separately for translational and rotational components:

```text
ssat(x, x_bar)
    = x_bar tanh(||x|| / x_bar) x / ||x||
```

for nonzero `x`.

This preserves the direction of each correction block while preventing unrealistically large intermediate velocity commands.

Current BlueROV2 defaults are:

```text
translational virtual-speed limit: 1.5 m/s
rotational virtual-speed limit:    2.0 rad/s
```

### Adaptive sensing domain

When the conservative sensing domain becomes too restrictive under limited actuation, the controller can temporarily enlarge it toward the physical domain.

For each constraint,

```text
h_a = h_c + rho_bar s,
s in [0, 1].
```

The auxiliary state `s` is governed by a CBF-like condition that maintains a positive physical margin. When the enlargement is no longer needed, the nominal dynamics recover `s -> 0`.

## Installation

The project uses Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).

From the repository root:

```bash
uv sync
```

Run the complete test/lint/format check with:

```bash
make check
```

Individual development commands can also be run through `uv`, for example:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## Examples

### 04 — Adaptive formation stress test

This example exercises the camera-constrained controller under limited actuation:

```bash
uv run python examples/04_bluerov2_fov_clf_qp.py \
    --control-space thruster \
    --stress-test \
    --adaptive \
    --save
```

Useful diagnostics include:

- conservative and physical sensing margins;
- normalized domain enlargement;
- thruster utilization;
- required CLF slack;
- zero-slack actuation margin;
- sampled-data funnel safeguard activation.

### 05 — Seven robots with a leader velocity command

The leader receives an inertial velocity command. A reference command filter generates a smooth position, velocity, and acceleration reference for the same CLF-QP architecture used by the followers.

```bash
uv run python examples/05_bluerov2_constant_velocity_formation.py \
    --control-space thruster \
    --adaptive \
    --save
```

The default leader command is

```text
[0.35, 0.08, 0.0] m/s
```

and the default simulation duration is 150 s.

### 06 — Seven robots following a 3-D leader trajectory

The leader tracks a smooth full translational trajectory with supplied position, velocity, and acceleration:

```bash
uv run python examples/06_bluerov2_trajectory_formation.py \
    --control-space thruster \
    --adaptive \
    --save
```

The default simulation duration is 200 s.

The example reports performance separately by tree depth, which makes it possible to inspect propagation of tracking error through the directed cascade.

### 07 — Realistic trajectory validation

This example adds simulation-only realism without modifying the controller:

```bash
uv run python examples/07_bluerov2_realistic_trajectory_validation.py \
    --control-space thruster \
    --adaptive \
    --save \
    --no-show
```

The default perturbations include:

- model-parameter uncertainty;
- unmodeled constant water current;
- relative-position measurement noise;
- body-twist measurement noise;
- fixed relative-measurement delay.

### 08 — Robustness Monte Carlo

A short Monte Carlo campaign can be run before moving to ROS/Gazebo:

```bash
uv run python examples/08_bluerov2_robustness_monte_carlo.py \
    --runs 10 \
    --duration 60
```

The summary focuses on:

- minimum physical sensing margin;
- maximum formation error;
- maximum adaptive enlargement;
- maximum thruster utilization;
- CLF feasibility;
- fallback occurrence.

## Seven-robot validation topology

The long-horizon examples use the balanced rooted tree

```text
            0
          /   \
         1     2
        / \   / \
       3   4 5   6
```

represented with the repository sensing convention as

```text
1 -> 0
2 -> 0
3 -> 1
4 -> 1
5 -> 2
6 -> 2
```

This gives a two-level cascade and makes it possible to compare formation performance at different graph depths.

## Generating the README animation

The README uses a GIF generated directly from the realistic seven-robot trajectory simulation.

Run:

```bash
bash scripts/generate_readme_animation.sh
```

which executes the equivalent of

```bash
uv run python examples/07_bluerov2_realistic_trajectory_validation.py \
    --control-space thruster \
    --adaptive \
    --save-animation \
    --animation-format gif \
    --frame-stride 20 \
    --output-dir docs/media \
    --no-show
```

and writes

```text
docs/media/formation_animation.gif
```

The image link at the top of this README will then render automatically on GitHub.

For a faster development preview, you can shorten the run:

```bash
uv run python examples/07_bluerov2_realistic_trajectory_validation.py \
    --duration 60 \
    --save-animation \
    --animation-format gif \
    --frame-stride 20 \
    --output-dir docs/media \
    --no-show
```

## Repository structure

```text
src/formation_control/
├── actuation/       # BlueROV2 Heavy thruster allocation and limits
├── constraints/     # distance and camera sensing constraints
├── control/         # command filters, CLF-QP, adaptive-domain logic
├── geometry/        # rigid-body and camera geometry
├── graphs/          # directed sensing graphs
├── models/          # marine vehicle models
├── potentials/      # formation, centering, and barrier potentials
├── simulation/      # simulation infrastructure and robustness models
└── visualization/   # plots and 3-D animations

examples/
tests/
docs/media/
```

## Current development status

The pure-Python controller contains the intended control architecture and is currently being validated before ROS integration.

Current milestones:

- [x] BlueROV2 6-DoF dynamics
- [x] directed tree formation model
- [x] camera field-of-view constraints
- [x] recentered logarithmic barriers
- [x] command-filtered backstepping
- [x] smooth norm saturation of the virtual twist
- [x] thruster-space CLF-QP
- [x] adaptive conservative-to-physical sensing domains
- [x] long-horizon seven-robot moving-leader validation
- [x] common CLF-QP architecture for leader and followers
- [x] simulation-only uncertainty/noise/delay layer
- [ ] ROS 2 wrapper
- [ ] Gazebo/SITL validation
- [ ] BlueROV2 experimental validation

## ROS integration direction

The ROS implementation should remain a thin wrapper around the Python control core.

The wrapper will be responsible for:

1. receiving the vehicle state and relative sensing measurements;
2. maintaining the command-filter state;
3. maintaining the adaptive-domain state;
4. converting ROS messages to the core controller inputs;
5. evaluating the existing controller;
6. publishing desired body wrench or thruster commands;
7. exposing diagnostics such as sensing margins, adaptive enlargement, CLF slack, actuation margin, thruster utilization, and fallback state.

The controller mathematics should remain ROS-independent.

## Development

Before committing changes, run:

```bash
make check
```

For the longer simulations, it is usually convenient to save outputs without opening figures:

```bash
--save --no-show
```

Animations can be generated separately once the numerical run has been validated.
