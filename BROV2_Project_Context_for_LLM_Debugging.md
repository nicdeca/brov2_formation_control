# BlueROV2 Formation-Control Project — Context for LLM-Assisted Debugging

Handover document for a new developer/student working on SITL, ROS 2, experiment execution, recording, and plotting

**Last updated: 28 August 2026**

## Purpose of this document

This document is meant to be given together with logs and relevant source files to an LLM when debugging the project. It summarizes the intended architecture, the maintained experiment workflow, important repository conventions, and known failure modes. The LLM should use it as context, but it must still inspect the current checkout before proposing code changes.

Important rule for debugging: do not infer that a file still has the exact contents described here. The project has been evolving quickly. Before modifying a file, inspect the current file and preserve the current architecture unless a change is explicitly justified.

## 1. Project overview

The repository implements distributed formation control for BlueROV2 Heavy underwater robots. The current experimental focus is a leader–follower formation controller with collision avoidance, sensing constraints, actuator saturation, adaptive barrier domains, PX4 SITL/Gazebo simulation, and later deployment on the physical Marinarium robots.

The maintained implementation uses ROS 2 around a control core. PX4 provides the low-level vehicle interface. In simulation, each BlueROV runs its own PX4 SITL instance and namespace. The controller sends body wrench commands through the existing PX4 interface, with the online CLF-QP solved directly in thruster coordinates.

## 2. Repository/workspace conventions

Typical repository root on the main developer machine is `~/discower_ws/src/brov2_formation_control`, with the ROS 2 workspace rooted at `~/discower_ws`. A student may instead have cloned the repository directly elsewhere; therefore scripts must not assume the main developer's home path.

The ROS 2 package is `formation_control_ros`. The maintained experiment launch and runner scripts live under the repository's `ros2/formation_control_ros/...` package and `scripts/` directory. The exact paths should be confirmed from the current checkout before editing.

## 3. Build environment

The project uses ROS 2, PX4 SITL, Gazebo, and Python. The repository provides `setup_ros2.sh`; use it when building/running the ROS package. The Python environment may use the repository's virtual environment, while ROS-facing commands must run in a shell with ROS 2 and the built package sourced.

Typical build sequence from the workspace root:

```bash
cd ~/discower_ws
source src/brov2_formation_control/setup_ros2.sh
rm -rf build/formation_control_ros install/formation_control_ros   # only if stale
colcon build --packages-select formation_control_ros --symlink-install
source install/setup.bash
```

If the repository is not inside `~/discower_ws`, adapt only the workspace path; do not hard-code another person's home directory into project files.

## 4. PX4 fork required by the SITL simulation

The maintained simulator does not use a completely generic upstream PX4 checkout. It expects the KTH-DHSG PX4 fork/branch that provides the custom `px4_sitl_uuv` configuration. The launcher expects the binary `<PX4_DIR>/build/px4_sitl_uuv/bin/px4`.

When the launch fails with `PX4 SITL UUV binary not found` and `make px4_sitl_uuv` reports that the target does not exist, first verify the PX4 repository and branch:

```bash
cd /path/to/PX4-Autopilot
git remote -v
git branch --show-current
git log -1 --oneline
make list_config_targets | grep -E 'sitl.*uuv|uuv'
```

The expected source is the KTH-DHSG fork, on the `kth-model-changes` branch. After switching/cloning, initialize submodules and build:

```bash
git submodule sync --recursive
git submodule update --init --recursive
make px4_sitl_uuv
ls -l build/px4_sitl_uuv/bin/px4
```

Do not try to fix this error by rebuilding the ROS package or changing the ROS launch code if the PX4 configuration itself is absent.

## 5. Maintained SITL launch and robot naming

The maintained multi-robot SITL launch is `multi_bluerov2_sim.launch.py`. It accepts `robot_count` and per-robot names such as `robot_1_name`, `robot_2_name`, etc. Typical two-robot names are `splash` (leader) and `glub` (follower). Robot names must be consistent across the simulator, controller launch, and runner.

Typical SITL command:

```bash
ros2 launch formation_control_ros multi_bluerov2_sim.launch.py   robot_count:=2   robot_1_name:=splash   robot_2_name:=glub   px4_dir:=/absolute/path/to/PX4-Autopilot
```

A namespace mismatch can cause QGroundControl/PX4 symptoms such as `Not Ready` or `No offboard signal`, even when the simulator itself appears alive. Always compare the robot names used by the simulator with the names used by the controller and runner.

## 6. Dynamics presets and SITL

The control model has several BlueROV2 parameter presets. Physical names map to physical parameter sets: `glub -> heavy_tube`, `splash -> heavy_tube`, `bubble -> standard`. However, SITL should use the explicit `gazebo` model rather than silently selecting a physical preset from the robot name.

Therefore, when debugging SITL, check that the experiment launch/controller configuration explicitly selects `gazebo` dynamics for simulated robots. Do not change the online controller back to wrench-space optimization; the maintained design solves the CLF-QP in thruster coordinates.

## 7. Coordinate frames and pool-aligned simulation

PX4 uses NED/FRD conventions. The control core uses NWU/FLU. The conversion uses `S = diag(1,-1,-1)` for the relevant vector transformations. Export/post-processing code must retain the PX4-to-core conversion even though the Gazebo world has been re-anchored.

The Gazebo tank/world was deliberately re-anchored so that PX4 local NED coordinates numerically resemble the intended real-pool frame. The simulated pool is not artificially enlarged. The maintained two-robot initialization, expressed in the control-core NWU frame, is approximately:

```text
splash (leader): [2.675, 0.050, -0.775]
glub   (follower): [4.475, 0.750, -0.775]
parent-minus-follower vector: [-1.800, -0.700, 0.000]
```

The shared maintained workspace in the core frame is:

```text
physical lower      = [0.300, -1.975, -2.155]
physical upper      = [7.100,  1.975,  0.225]
conservative lower  = [0.450, -1.825, -1.955]
conservative upper  = [6.950,  1.825, -0.325]
```

If a launch still contains positions around `z ≈ -95`, it is probably an older/legacy experiment file and should not be assumed to be part of the maintained pool-aligned workflow.

## 8. Current two-robot experiment lifecycle

The experiment is split into two phases: `INITIALIZE` and `FORMATION`.

During `INITIALIZE`, each robot tracks its own absolute initialization position. The follower is not mathematically controlled relative to the leader in this phase. Its initialization controller requires only the follower's own state. The leader does not have to be visible throughout initialization.

At the end of initialization, the chosen absolute poses are consistent with the nominal leader–follower geometry, so the leader should be visible and the sensing edge should be valid when the system switches to `FORMATION`. During `FORMATION`, the follower uses its parent/relative information and the follower control runtime.

The phase manager checks initialization convergence using position and speed tolerances before switching to `FORMATION`. A future improvement may explicitly require valid sensing edges before that transition.

## 9. Maintained experiment launch/runner files

The canonical maintained experiment set is currently:

```text
multi_bluerov2_sim.launch.py
two_robot_experiment.launch.py
two_robot_adaptive_mission.launch.py
five_robot_tree_experiment.launch.py

run_two_robot_experiment.py
run_two_robot_adaptive_mission.py
run_five_robot_tree_experiment.py

record_formation_experiment.sh
record_formation_experiment.py
export_experiment.py
export_initialization_bag.py
export_formation_bag.py
plot_experiment.py
plot_initialization_experiment.py
plot_formation_experiment.py
```

Older three-robot launches, smoke-test launches, old five-robot star-topology runners, and one-off relaxation test runners may still exist. Treat them as legacy/debug material unless the task explicitly concerns them.

## 10. Running the controller/mission

A common workflow is: start SITL, launch the relevant controller experiment, then run the mission runner. The runner's leader argument must match the actual leader namespace. For example, if the leader is `splash`, use `--leader splash` where supported.

The maintained runner publishes a transient-local mission status topic:

```text
/formation_control/mission_status
```

with the lifecycle `WAITING -> RUNNING -> COMPLETE` (or `ABORTED` on interruption/error). The phase manager publishes:

```text
/formation_control/experiment_phase
```

with `INITIALIZE -> FORMATION`.

## 11. Recording architecture

The recorder intentionally separates initialization and mission data. The current run directory has the following structure:

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

Typical recorder command from the repository root:

```bash
scripts/record_formation_experiment.sh   --name two_robot_experiment   --robots splash,glub   --edge glub:splash
```

If the shell returns `Permission denied`, set the executable bit once:

```bash
chmod +x scripts/record_formation_experiment.sh
```

The shell wrapper locates its Python recorder relative to the script itself; it should not contain a developer-specific absolute path. The Python recorder likewise locates its post-processing scripts relative to `__file__`.

One path detail matters: the default recording root is relative to the current working directory. If the recorder is launched from `repo/ros2`, data may appear under `repo/ros2/outputs/experiments` rather than `repo/outputs/experiments`. Prefer launching the recorder from the repository root, or pass an explicit `--output-root`.

## 12. How split recording is triggered

The initialization bag starts immediately when the recorder starts. When the phase becomes `FORMATION`, the initialization bag is closed. The recorder then waits for `/formation_control/mission_status = RUNNING` before opening the mission bag. It closes the mission bag on `COMPLETE` or `ABORTED`.

This means a recorder that appears to be 'not recording the mission' may actually be waiting for a missing status transition. Check:

```bash
ros2 topic echo /formation_control/experiment_phase
ros2 topic echo /formation_control/mission_status
ps aux | grep "[r]os2 bag record"
```

Expected lifecycle:

```text
experiment_phase: INITIALIZE -> FORMATION
mission_status:   WAITING -> RUNNING -> COMPLETE
```

If the student has an older runner that does not publish `mission_status`, initialization may record correctly while the mission bag never starts.

## 13. Exporting and plotting

The preferred manual post-processing uses the umbrella scripts:

```bash
RUN=$(ls -dt outputs/experiments/* | head -n 1)

python scripts/export_experiment.py "$RUN"
uv run python scripts/plot_experiment.py "$RUN"
```

These process all available phases and keep their outputs separate. They skip an unavailable phase instead of failing the entire run.

For phase-specific debugging:

```bash
python scripts/export_experiment.py "$RUN" --phase initialization
uv run python scripts/plot_experiment.py "$RUN" --phase initialization

python scripts/export_experiment.py "$RUN" --phase mission
uv run python scripts/plot_experiment.py "$RUN" --phase mission
```

Do not type the literal word `RUN` unless a shell variable named `RUN` has actually been assigned. The scripts print absolute output paths; use those paths when checking whether data were produced.

## 14. Initialization diagnostics and recent fixes

The initialization exporter/plotter is intended to show both leader and follower absolute initialization references, convergence, workspace margins, controller/PX4 state, required slack, and thruster utilization.

The follower must publish its initialization diagnostic snapshot so that its actual absolute initialization reference is recorded. During `INITIALIZE`, follower freshness should require only self odometry; parent odometry is required in `FORMATION`.

Workspace diagnostics are published on dedicated topics and should be recorded/exported. If workspace plots are empty or contain only a zero boundary, verify that the current `DiagnosticsPublisher`, recorder topic list, and exporter topic names agree exactly. Do not hide missing data with plotting tricks.

For an old bag recorded before workspace diagnostics were published, static workspace margins can be reconstructed from odometry and known bounds, but historical adaptive relaxation cannot be recovered if it was never recorded.

## 15. Control architecture constraints relevant when fixing code

The current online controller uses a CLF-QP in thruster coordinates. The optimization variable is the 8-dimensional thruster command and the actuator box constraints are enforced directly. Do not 'simplify' this back into a wrench-space QP without discussing the design change.

The sensing/collision constraints use recentered logarithmic Barrier Lyapunov Functions with adaptive domains. The relaxation is currently state/barrier driven: the adaptive domain enlarges as the adaptive barrier margin becomes small and recovers when margin is restored. It is not currently triggered by QP infeasibility or required CLF slack.

The relaxation state is not simply clipped at 1. The implementation uses a domain margin and an activation window. Any change to the adaptive-domain equations should be checked against the existing proof/design, not treated as a generic saturation variable.

A possible future extension is integral/adaptive disturbance estimation in the velocity dynamics to reject steady bias. It is not part of the current maintained controller and should not be added casually while debugging unrelated issues.

## 16. Two-robot adaptive mission defaults (useful sanity checks)

The dedicated adaptive SITL mission uses explicit `gazebo` dynamics and the same pool-aligned initialization/workspace. Representative sensing/adaptation defaults are:

```text
d_min physical              = 0.5
d_max physical              = 3.6
d_min conservative          = 0.8
d_max conservative          = 2.4
conservative FoV            = 0.45 / 0.45
recovery gain               = 0.8
barrier gain                = 0.20
domain margin               = 0.02
activation on               = 0.001
activation off              = 0.15
```

If observed behavior is radically inconsistent with these values, inspect the actual launch arguments and parameter overrides before changing controller code.

## 17. Five-robot maintained topology

The maintained five-robot experiment uses a balanced directed tree rather than the older star topology:

```text
4 -> 2 -> 1
5 -> 3 -> 1
```

The current five-robot runner should be inspected before assuming it exposes exactly the same CLI options as the two-robot runner. In particular, historical versions differed in `--gazebo-timer` support and in leader velocity commands.

## 18. First-line troubleshooting checklist

When something does not run, collect evidence before changing code:

```bash
# Repository / branch
git status
git branch --show-current
git log -1 --oneline

# ROS package
ros2 pkg prefix formation_control_ros
ros2 node list
ros2 topic list | sort

# Key experiment state
ros2 topic echo /formation_control/experiment_phase
ros2 topic echo /formation_control/mission_status

# Per-robot namespaces
ros2 topic list | grep -E '/(splash|glub)/'

# Recorder
ps aux | grep "[r]os2 bag record"
find . -type d -path "*/outputs/experiments/*" | tail -30

# PX4
cd /path/to/PX4-Autopilot
git remote -v
git branch --show-current
make list_config_targets | grep uuv
ls -l build/px4_sitl_uuv/bin/px4
```

Also copy the complete terminal error, not only the last line. For ROS launch errors, rerun with debug output if necessary. For namespace/recording issues, include `ros2 topic list` and the exact commands used to start simulator, controller, runner, and recorder.

## 19. Guidance to the LLM

When using this document as context, the LLM should follow these rules:

1. Inspect the current source file before proposing a modification; do not assume a historical version.
2. Prefer a minimal fix that preserves the maintained architecture.
3. Distinguish simulator setup problems, ROS namespace/topic problems, controller logic problems, and recording/post-processing problems.
4. Never introduce developer-specific absolute paths into repository code.
5. When a path must be supplied by the user (e.g. PX4), keep it as a launch/CLI argument.
6. Do not replace the thruster-coordinate CLF-QP with a different controller merely to make a runtime error disappear.
7. Do not change coordinate-frame conversions unless the current data flow has been traced end-to-end.
8. If a plot is empty, verify whether data were ever published/recorded/exported before modifying the plotting code.
9. For SITL, ensure robot names and dynamics presets are consistent across simulator, controller, and runner.
10. When suggesting file replacements, provide complete files or a clearly reviewable diff and state exactly which repository-relative path each file belongs to.

## 20. Suggested prompt to accompany this document

The student can paste the following together with this document, the relevant logs, and the source files:

```text
I am debugging the BlueROV2 formation-control project described in the attached
context document. Please use that document as architectural context, but treat
the source files I provide as authoritative for the current implementation.

My goal/problem is:
<describe the issue>

Commands I ran:
<paste exact commands>

Complete terminal output:
<paste logs>

Relevant current files:
<attach/paste files>

Please:
1. identify the most likely layer causing the problem;
2. explain the evidence from the logs/code;
3. avoid changing unrelated control architecture;
4. tell me which diagnostics/commands to run next if the cause is not yet
   certain;
5. if code changes are necessary, give complete replacement files with their
   repository-relative paths, or a minimal clearly scoped diff.
```
