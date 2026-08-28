# ROS experiment logging and plot pipeline

The current workflow records **initialization and the actual mission as two
separate bags** while preserving the same controller diagnostic snapshot used
for the rich BlueROV2 plots.

## 1. Data flow

```text
controllers + PX4
      |
      +--> /formation_control/experiment_phase
      +--> /formation_control/mission_status
      +--> /<robot>/formation_control/diagnostic_snapshot
      +--> PX4 odometry / setpoints / control mode
      |
      v
split recorder
      |
      +--> initialization/bag
      |       |
      |       +--> export_experiment.py
      |       |       +--> export_initialization_bag.py
      |       +--> initialization_history.npz
      |       +--> plot_experiment.py
      |               +--> plot_initialization_experiment.py
      |
      +--> mission/bag
              |
              +--> export_experiment.py
              |       +--> export_formation_bag.py
              +--> formation_history.npz
              +--> plot_experiment.py
                      +--> plot_formation_experiment.py
```

## 2. Split semantics

Start the recorder **before arming**.

- `initialization/bag` starts immediately;
- it closes when `/formation_control/experiment_phase` becomes `FORMATION`;
- the runner publishes `mission_status = RUNNING` after phase/subscriber checks;
- the runner waits one second so the mission recorder can open;
- `mission/bag` contains the actual scripted mission;
- it closes on `COMPLETE` or `ABORTED`.

If `FORMATION` is never reached, interrupting the recorder still closes and
preserves the initialization bag.

## 3. Run-directory layout

```text
outputs/experiments/<timestamp>_<name>/
├── run_manifest.yaml
├── initialization/
│   ├── bag/
│   ├── initialization_history.npz
│   └── plots/
└── mission/
    ├── bag/
    ├── formation_history.npz
    └── plots/
```

The recorder and exporters print the absolute output path after saving.

## 4. What is recorded

For every robot the recorder includes PX4 odometry, control mode, thrust/torque
setpoints, `cmd_vel`, the atomic controller diagnostic snapshot and the
human-readable sensing/workspace diagnostic topics.

The diagnostic snapshot contains, as available:

- controller time and fallback state;
- CLF slack, required zero-slack actuation slack and actuation margin;
- all eight thruster forces, limits, utilization and body wrench;
- sensing conservative/physical margins;
- sensing relaxation state/rate;
- normalized image coordinates;
- desired/actual relative quantities;
- CLF/backstepping decomposition;
- workspace relaxation and physical margins;
- leader reference quantities.

The mission exporter converts PX4 NED/FRD odometry to the controller's core
NWU/FLU convention. The pool re-anchoring does not change this conversion.

## 5. Record

Two-robot hardware-like example:

```bash
scripts/record_formation_experiment.sh \
  --name two_robot_01 \
  --robots splash,glub \
  --edge glub:splash
```

Five-robot tree:

```bash
scripts/record_formation_experiment.sh \
  --name five_tree_01 \
  --robots itrl_rov_1,itrl_rov_2,itrl_rov_3,itrl_rov_4,itrl_rov_5 \
  --edge itrl_rov_2:itrl_rov_1 \
  --edge itrl_rov_3:itrl_rov_1 \
  --edge itrl_rov_4:itrl_rov_2 \
  --edge itrl_rov_5:itrl_rov_3
```

Use `--no-postprocess` to record only.

Without `--no-postprocess`, the recorder automatically calls the same
`export_experiment.py` and `plot_experiment.py` umbrella commands after
the run closes.

## 6. Export both phases with one command

For normal post-processing, use the umbrella exporter:

```bash
RUN=$(ls -dt outputs/experiments/* | head -n 1)

python scripts/export_experiment.py "$RUN"
```

It exports every phase whose bag exists:

```text
initialization/bag -> initialization/initialization_history.npz
mission/bag        -> mission/formation_history.npz
```

If initialization failed and no mission bag exists, the command still exports
the initialization phase and simply reports that the mission is unavailable.

To export only one phase:

```bash
python scripts/export_experiment.py "$RUN" --phase initialization
python scripts/export_experiment.py "$RUN" --phase mission
```

The lower-level exporters remain available for debugging or custom workflows:

```bash
python scripts/export_initialization_bag.py "$RUN"
python scripts/export_formation_bag.py "$RUN" --phase mission
```

The initialization exporter is deliberately tolerant of startup failures. It
requires synchronized PX4 odometry; controller snapshots are decoded when
available but are not required for the export to exist.

## 7. Plot both phases with one command

After exporting, generate both initialization and mission plots with:

```bash
uv run python scripts/plot_experiment.py "$RUN"
```

The outputs are kept separate:

```text
initialization/plots/
mission/plots/
```

The default mission preset is the paper-oriented plot set with paper-quality
formatting. To generate all mission diagnostics instead:

```bash
uv run python scripts/plot_experiment.py "$RUN" --mission-preset all
```

Other useful options are:

```bash
uv run python scripts/plot_experiment.py "$RUN" --show-legends
uv run python scripts/plot_experiment.py "$RUN" --format png
uv run python scripts/plot_experiment.py "$RUN" --phase initialization
uv run python scripts/plot_experiment.py "$RUN" --phase mission
```

The initialization plotter produces:

- per-robot position histories and reference overlays when the corresponding
  reference is present in the logged snapshot;
- position-error/speed convergence where the reference is available;
- workspace physical margins and relaxation states;
- thruster utilization / required slack / armed / Offboard status.

The default plotted phase-manager thresholds are `0.65 m` position error and
`0.08 m/s` speed; override the lower-level initialization plotter arguments if
a launch uses different values.

The mission exporter still supports legacy runs containing `<run>/bag`. The
umbrella exporter is intended for the current split-recording layout.

Useful lower-level mission plot selectors include:

```text
--trajectory
--leader-tracking
--leader-position
--formation-error
--formation-tracking
--workspace
--workspace-relaxation
--distance
--fov
--adaptive-fov
--slack
--actuation
--domain
--relaxation-rates
--thrusters
--controller-time
--clf-value
--clf-balance
--clf-drift
--backstepping
--peak-debug
--paper
--all
```

Legends on compact formation/sensing paper plots are hidden by default; use
`--show-legends` when needed.

## 8. Animations

Mission histories can also generate the 3-D formation animation and diagnostic
animations:

```bash
uv run python scripts/plot_formation_experiment.py \
  "$RUN/mission/formation_history.npz" \
  --animation --diagnostic-animations \
  --animation-format mp4 \
  --save
```

## 9. Important operational rule

Do not manually restart recording at the `INITIALIZE -> FORMATION` transition.
The split recorder is designed to make the boundary deterministic while the
operator concentrates on QGC, arming and the physical robots.
