# Examples

These examples cover the progression from simple formation-control models to the
BlueROV2 controller used in the paper. Run them from the repository root with
`uv run python examples/<script>.py ...`.

## Current paper conventions

The BlueROV2 adaptive examples use the current paper formulation:

- direct-distance constraints are the default,
  `h_delta = d - d_min^c` and `h_Delta = d_max^c - d`;
- the recentered logarithmic barriers shift both the current and desired
  constraint values by the same adaptive enlargement;
- the adaptive state `s_ell in [0,1]` is activated by the smooth margin gate
  `sigma_ell(h_ell^a)`;
- the adaptive gate uses only the exact actuator-required CLF slack
  `delta_req`, as in the paper; the optimized QP slack is logged only for
  diagnostics;
- the continuous adaptive law is integrated with an implicit sampled-data step,
  which avoids the explicit-Euler overshoot observed near `h_a = 0`.

The old `relaxation_domain_margin_ratio` controller parameter is no longer part
of the adaptive law. The squared-distance constraint implementation is retained
in the core library as an optional legacy mode, but the examples use the paper's
direct-distance form by default.

## Recommended stress test

The short pure-Python stress test is `04_bluerov2_fov_clf_qp.py`. It is the
recommended first check before moving to Gazebo/PX4 because it directly exposes
CLF slack, actuator feasibility, adaptive-domain motion, and sensing margins.

Run the adaptive stress test with:

```bash
uv run python examples/04_bluerov2_fov_clf_qp.py \
  --control-space thruster \
  --stress-test \
  --thrust-derating 0.35 \
  --adaptive \
  --no-animation \
  --no-show
```

To compare against the fixed conservative domain, rerun the same case with
`--no-adaptive`:

```bash
uv run python examples/04_bluerov2_fov_clf_qp.py \
  --control-space thruster \
  --stress-test \
  --thrust-derating 0.35 \
  --no-adaptive \
  --no-animation \
  --no-show
```

Omit `--no-show` if you want the figures to open interactively. For a harder
initial disturbance, add e.g. `--stress-scale 1.5`.

The most useful stress-test diagnostics are the required CLF slack, adaptive
state/domain enlargement, adaptive-state rates, physical/conservative sensing
margins, and thruster utilization.

## Example overview

| Script | Purpose |
|---|---|
| `01_single_integrator_nominal.py` | Minimal directed formation regulation with single-integrator agents. |
| `02_single_integrator_constraints.py` | Adds collision/connectivity distance barriers to the single-integrator model. |
| `03_double_integrator_clf_qp.py` | Introduces the command-filtered CLF-QP on a second-order model. |
| `04_bluerov2_fov_clf_qp.py` | Main compact BlueROV2 simulation: 6-DoF dynamics, camera/FoV constraints, recentered barriers, command filter, thruster-level CLF-QP, adaptive domains, and the short stress test. |
| `05_recentered_barriers.py` | Generates/inspects the scalar recentered barrier curves used in the paper. |
| `05_bluerov2_constant_velocity_formation.py` | Multi-BlueROV formation under a leader velocity command. |
| `06_bluerov2_wrench_polytope.py` | Inspects the BlueROV2 Heavy achievable wrench polytope. |
| `06_bluerov2_trajectory_formation.py` | Multi-BlueROV formation following a full leader trajectory. |
| `07_compare_thruster_and_wrench_clf_qp.py` | Compares direct thruster-space and exact wrench-space CLF-QP formulations. |
| `07_bluerov2_realistic_trajectory_validation.py` | Adds realistic perturbations/model mismatch to the moving formation validation. |
| `08_bluerov2_adaptive_domain_relaxation.py` | Runs the short limited-authority adaptive stress case and exports the representative adaptive-domain figure. |
| `08_bluerov2_robustness_monte_carlo.py` | Monte Carlo campaign for the moving formation with model/sensing perturbations. |
| `09_bluerov2_readme_formation_demo.py` | Visual formation-acquisition demonstration intended for repository documentation. |
| `10_bluerov2_five_robot_tree_full_mission.py` | Pure-Python counterpart of the five-robot tree mission used before ROS/Gazebo deployment. |
| `11_bluerov2_funnel_relaxation_comparison.py` | Detailed fixed-vs-adaptive stress comparison. The filename is retained for compatibility; its controller now uses the current adaptive-domain law. |
| `12_bluerov2_adaptive_mission.py` | Two-robot moving mission with online formation switches and adaptive sensing domains. |
| `moving_formation_validation.py` | Long-horizon moving-formation validation used by several higher-level examples. |

## Useful commands

Main nominal BlueROV2 example:

```bash
uv run python examples/04_bluerov2_fov_clf_qp.py --control-space thruster
```

Representative adaptive-domain paper figure:

```bash
uv run python examples/08_bluerov2_adaptive_domain_relaxation.py
```

Detailed adaptive stress comparison:

```bash
uv run python examples/11_bluerov2_funnel_relaxation_comparison.py --save --no-show
```

Two-robot adaptive mission:

```bash
uv run python examples/12_bluerov2_adaptive_mission.py --save --no-show
```

Five-robot pure-Python mission:

```bash
uv run python examples/10_bluerov2_five_robot_tree_full_mission.py --save --no-show
```

Monte Carlo robustness check:

```bash
uv run python examples/08_bluerov2_robustness_monte_carlo.py --runs 10
```

## Barrier-weight tuning

`05_recentered_barriers.py` is the barrier-tuning utility. The existing
`--tuning` option plots the recentered barriers and their physical-coordinate
gradients. For Gazebo tuning, `--weight-sweep` directly compares candidate
barrier weights `mu` for one constraint channel. For example, for the vertical
FoV barrier:

```bash
uv run python examples/05_recentered_barriers.py \
  --weight-sweep 0.10 0.25 0.50 1.0 \
  --weight-sweep-channel vertical_fov \
  --save \
  --no-show
```

This saves the usual barrier figures plus
`barrier_weight_sweep_vertical_fov.png` under `outputs/recentered_barriers/`.
The second panel shows the absolute gradient with respect to the physical
coordinate, which is the useful view for deciding how early and strongly the
barrier acts.

## Notes

The two-digit prefixes are historical and are not a strict dependency order;
there are parallel examples with the same number. The scripts import the
`formation_control` package from the repository, so run them from an environment
where the project itself is installed (normally through `uv`).


## Stress-test outputs

`04_bluerov2_fov_clf_qp.py` automatically saves stress-test and thrust-authority-sweep runs.
If `--output-dir` is omitted, a timestamped directory is created under
`outputs/bluerov2_fov_clf_qp/`, and the absolute path is printed in the terminal.
A single stress-test run saves all diagnostic figures, `clf_diagnostics.csv`, and
`simulation_diagnostics.npz`, which contains the raw histories needed to reproduce or
inspect the plots.  Curves belonging to the same sensing edge use the same color in the
distance/FoV plots; adaptive-state/rate plots use a fixed color per constraint channel.

Example:

```bash
uv run python examples/04_bluerov2_fov_clf_qp.py \
  --control-space thruster \
  --stress-test \
  --thrust-derating 0.40 \
  --adaptive \
  --no-animation \
  --no-show
```

## Robust adaptive-domain implementation

The continuous adaptive law matches the paper: the actuation gate is based
only on the exact required zero-relaxation CLF slack `delta_req`. The optimized
QP slack is logged for diagnostics but does not enter the gate.

For sampled simulation/ROS execution, the continuous law is left unchanged and
three numerical safeguards are used:

1. `FunnelRelaxationPolicy.advance(...)` integrates the adaptive state
   implicitly with the sampled conservative constraint held fixed.
2. A one-step predictive guard estimates `h_c_dot` by backward finite
   differences and minimally increases the post-update `s` whenever
   `h_c + dt min(h_c_dot,0) + rho_max s` would fall below the implementation
   guard at the next sample.
3. `project_to_current_domain(...)` remains as an emergency projection if an
   unexpected inter-sample crossing still occurs.

The guard margin is 2% of the available enlargement reserve by default. These
operations are sampled-data implementation safeguards; they do not add terms to
the continuous-time adaptive law analyzed in the paper.

For the stress test, use the normal virtual-command limits first:

```bash
uv run python examples/04_bluerov2_fov_clf_qp.py \
  --control-space thruster \
  --stress-test \
  --thrust-derating 0.35 \
  --adaptive \
  --no-animation \
  --no-show
```

The example automatically writes plots and raw diagnostics to a timestamped
folder under `outputs/bluerov2_fov_clf_qp/` and prints that absolute path in the
terminal.
