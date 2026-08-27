# ROS experiment logging and paper-plot pipeline

This patch makes ROS/SITL/hardware runs feed the **same BlueROV2 diagnostic
plotters** already used by `examples/04_bluerov2_fov_clf_qp.py`.

```text
already-computed core controller evaluation
             |
             v
versioned diagnostic snapshot (ROS-independent schema)
             |
             v
/<robot>/formation_control/diagnostic_snapshot
             |
             +------------------------+
             |                        |
        PX4 odometry             PX4 setpoints
             |                        |
             +-----------+------------+
                         v
                     ros2 bag
                         |
                         v
             export_formation_bag.py
                         |
                         v
                formation_history.npz
                         |
                         v
             plot_formation_experiment.py
                         |
                         v
        existing example-04 plot functions
```

The core package stays ROS-independent. The ROS package only serializes and
publishes the snapshot.

## 1. What is logged

The snapshot contains everything required to reconstruct the histories used by
the rich BlueROV2 simulation plots:

- controller evaluation time and fallback state;
- CLF-QP slack, actuator-required slack and zero-slack actuation margin;
- maximum T200 utilization, all eight thruster forces, force limits and body
  wrench;
- conservative and physical sensing-constraint values;
- normalized adaptive state `s` and `s_dot` for collision/range/horizontal-FoV/
  vertical-FoV channels;
- normalized image coordinates `(alpha_h, alpha_v)`;
- current and desired parent-minus-follower relative position;
- distance/FoV domain parameters and enabled-constraint flags;
- camera half angles and camera-to-body extrinsics;
- full physical-CLF decomposition used by `CLFDiagnosticHistory`;
- generalized velocity, desired virtual velocity, filtered command and command
  derivative;
- leader position/velocity/acceleration/quaternion reference when provided.

PX4 odometry, thrust/torque setpoints and vehicle-control mode are also kept in
the rosbag. The exporter converts PX4 NED/FRD odometry back to the core
NWU/FLU convention before writing the portable NPZ.

## 2. One integration hook is required in the current ROS runtime

The archive can add the schema, publisher and scripts directly, but the exact
current `FollowerController` / `LeaderController` source was not part of the
attached repository snapshot. Add the publisher where each node already owns
its successful core evaluation. **Do not evaluate the controller a second time
for logging.**

Create the publisher once in each node:

```python
from formation_control_ros.snapshot_publisher import DiagnosticSnapshotPublisher

self._diagnostic_snapshot = DiagnosticSnapshotPublisher(self)
```

For a follower, call the ROS-independent extractor at the point where the core
runtime already has `evaluation`, sensing kinematics and the relaxation result:

```python
from formation_control.experiment import follower_snapshot_values

values = follower_snapshot_values(
    model=model,
    allocation=allocation,
    camera=camera,
    follower_state=follower_state,
    parent_position=parent_position,
    parent_velocity_inertial=parent_velocity_inertial,
    desired_relative_position=desired_relative_position,
    evaluation=evaluation,
    conservative_values=kinematics.values,
    relaxation_state=relaxation_state,        # normalized s, not rho
    relaxation_rate=relaxation_rate,          # normalized s_dot
    controller_time_s=controller_time_s,
    fallback=False,
    distance_domain=distance_domain,
    fov_domain=fov_domain,
    constraints_enabled=np.array(
        [distance_constraints, distance_constraints,
         fov_constraints, fov_constraints],
        dtype=bool,
    ),
    adaptive_enabled=adaptive,
    domain_margin_ratio=relaxation_policy.domain_margin_ratio,
)
self._diagnostic_snapshot.publish(values)
```

On fallback, still publish one snapshot with at least:

```python
self._diagnostic_snapshot.publish(
    {
        "role": 0.0,
        "fallback": 1.0,
        "controller_time_s": controller_time_s,
    }
)
```

For the leader, publish its actual optimized eight-thruster command and the
reference used by the leader controller:

```python
from formation_control.experiment import leader_snapshot_values

values = leader_snapshot_values(
    thruster_forces=thruster_forces,
    body_wrench=body_wrench,
    controller_time_s=controller_time_s,
    fallback=fallback,
    allocation=allocation,
    reference_position=p_r,
    reference_velocity=v_r,
    reference_acceleration=a_r,
    reference_quaternion=q_r,
)
self._diagnostic_snapshot.publish(values)
```

If the leader runtime already exposes its CLF internals, pass them through
`extra_values` using the schema field names. This is optional for the current
paper plots; position/velocity reference logging is enough for the leader RMS
tracking figure.

The publisher uses `std_msgs/msg/Float64MultiArray`. Ensure `std_msgs` is a
runtime dependency in `ros2/formation_control_ros/package.xml` if it is not
already present.

## 3. Record an experiment

Source the ROS environment first, then run:

```bash
scripts/record_formation_experiment.sh \
  --name two_robot_1p8 \
  --robots itrl_rov_1,itrl_rov_3 \
  --edge itrl_rov_3:itrl_rov_1
```

The run is written under:

```text
outputs/experiments/YYYYMMDD_HHMMSS_two_robot_1p8/
├── run_manifest.yaml
└── bag/
```

`Ctrl-C` stops rosbag cleanly. For a larger rooted tree, repeat `--edge` once
per follower-parent edge and list all robot namespaces in `--robots`.

## 4. Export the bag to a portable NPZ

The bag reader needs the ROS Python environment:

```bash
source setup_ros2.sh
python scripts/export_formation_bag.py \
  outputs/experiments/<run>
```

This creates:

```text
outputs/experiments/<run>/formation_history.npz
```

The exporter synchronizes all robot odometry and follower snapshots to a common
controller grid (30 ms nearest-neighbor tolerance by default). Use
`--max-sync-ms` to change it.

## 5. Generate the paper plots

The plotting stage is ROS-independent and should normally be run through the
core `uv` environment.

Paper-oriented preset:

```bash
uv run python scripts/plot_formation_experiment.py \
  outputs/experiments/<run>/formation_history.npz \
  --paper --paper-quality --save
```

All diagnostics that existed in the rich BlueROV2 simulation:

```bash
uv run python scripts/plot_formation_experiment.py \
  outputs/experiments/<run>/formation_history.npz \
  --all --save
```

A selected subset:

```bash
uv run python scripts/plot_formation_experiment.py \
  outputs/experiments/<run>/formation_history.npz \
  --trajectory --leader-tracking --distance --fov \
  --adaptive-fov --thrusters --controller-time \
  --paper-quality --save --show
```

Available plot selectors:

- `--trajectory`: 3-D formation trajectory, BlueROV geometry, sensing graph and
  edge-quality coloring;
- `--leader-tracking`: leader position/velocity tracking errors;
- `--distance`: old inter-agent distance/adaptive-boundary plot;
- `--fov`: old horizontal + vertical FoV plots;
- `--adaptive-fov`: publication-style representative FoV/domain-relaxation
  two-panel plot;
- `--slack`: optimal CLF slack and actuator-required slack;
- `--actuation`: zero-slack CLF actuation margin;
- `--domain`: old four-channel domain enlargement plot;
- `--relaxation-rates`: old auxiliary funnel-rate plot;
- `--thrusters`: old eight-T200 force plot;
- `--controller-time`: old controller-time plot;
- `--clf-value`: old `W` and `alpha(W)` plot;
- `--clf-balance`: old hard-CLF feasibility balance;
- `--clf-drift`: old CLF drift decomposition;
- `--backstepping`: old velocity-backstepping decomposition;
- `--peak-debug`: all four old peak-detail diagnostic figures;
- `--paper`: paper-oriented preset;
- `--all`: every plot above.

Use `--format pdf|png|svg`, `--output-dir PATH`, `--save`, and `--show` as
needed. With multiple followers, `--observer ROBOT_NAMESPACE` chooses the edge
used for the representative `--adaptive-fov` figure.

## 6. Why the plotting script imports example 04

This first version deliberately loads `examples/04_bluerov2_fov_clf_qp.py` and
calls its existing plotting functions. That guarantees that simulation and ROS
runs do not silently diverge into two plotting implementations.

After this pipeline has been validated on a few bags, the clean follow-up
refactor is to move `CLFDiagnosticHistory` and the reusable diagnostic plot
functions from example 04 into
`src/formation_control/visualization/bluerov2_diagnostics.py`. Then both the
simulation example and ROS experiment plotter can import that module directly;
the snapshot and NPZ formats do not need to change.


## Current paper and animation commands

Paper figures:

```bash
uv run python scripts/plot_formation_experiment.py \
  "$RUN/formation_history.npz" \
  --paper \
  --paper-quality \
  --save \
  --format pdf
```

The distance and horizontal/vertical FoV paper plots use one color per sensing
edge. Measured quantities are solid, adaptive bounds for the same edge are
dashed, and physical/conservative limits use distinct line styles.

Formation animation:

```bash
uv run python scripts/plot_formation_experiment.py \
  "$RUN/formation_history.npz" \
  --animation \
  --save \
  --animation-format mp4 \
  --frame-stride 2
```

The animation reconstructs the time-varying desired absolute positions along
the directed tree and renders them as subdued dashed/wireframe reference
vehicles alongside the measured formation.
