# Recording and comparing PX4, EKF, and raw MoCap

The comparison workflow records:

```text
/<robot>/fmu/out/vehicle_odometry
/mocap/<robot>/pose
/mocap/<robot>/pose_core
/mocap/<robot>/odom_ekf
```

`pose_core` is the raw MoCap pose after only the NED/FRD -> core-NWU/FLU
boundary conversion. It is therefore the correct raw reference to compare
against PX4 and the filtered EKF state.

## Two robots

Hardware:

```bash
scripts/record_formation_experiment.sh \
  --name two_robot_ekf \
  --robots splash,bubble \
  --edge bubble:splash \
  --state-source nav_msgs \
  --state-topic-template '/mocap/{robot}/odom_ekf' \
  --mocap-world-frame core_nwu \
  --odom-twist-frame body \
  --imu-topic-template '/{robot}/mavros/imu/data'
```

SITL:

```bash
scripts/record_formation_experiment.sh \
  --name two_robot_ekf_sitl \
  --robots splash,bubble \
  --edge bubble:splash \
  --state-source nav_msgs \
  --state-topic-template '/mocap/{robot}/odom_ekf' \
  --mocap-world-frame core_nwu \
  --odom-twist-frame body \
  --imu-topic-template '/mocap/{robot}/imu'
```

## Three robots

Hardware:

```bash
scripts/record_formation_experiment.sh \
  --name three_robot_ekf \
  --robots splash,glub,bubble \
  --edge glub:splash \
  --edge bubble:splash \
  --state-source nav_msgs \
  --state-topic-template '/mocap/{robot}/odom_ekf' \
  --mocap-world-frame core_nwu \
  --odom-twist-frame body \
  --imu-topic-template '/{robot}/mavros/imu/data'
```

SITL:

```bash
scripts/record_formation_experiment.sh \
  --name three_robot_ekf_sitl \
  --robots splash,glub,bubble \
  --edge glub:splash \
  --edge bubble:splash \
  --state-source nav_msgs \
  --state-topic-template '/mocap/{robot}/odom_ekf' \
  --mocap-world-frame core_nwu \
  --odom-twist-frame body \
  --imu-topic-template '/mocap/{robot}/imu'
```

The recorder defaults already include `/mocap/{robot}/pose_core`.

## Plotting

The normal run plotter invokes estimator comparison automatically when the
parallel streams are available:

```bash
python3 scripts/plot_experiment.py \
  outputs/experiments/<run>
```

To run only the estimator comparison on an exported phase history:

```bash
python3 scripts/plot_state_estimator_comparison.py \
  outputs/experiments/<run>/mission/formation_history.npz \
  --save
```

The comparison includes:

- position: PX4 / EKF / transformed raw MoCap;
- body linear velocity: PX4 / EKF / raw-MoCap finite difference;
- pairwise attitude disagreement;
- body angular velocity: PX4 / EKF / raw-MoCap finite difference.

Raw-MoCap finite-difference velocities are diagnostics only and are never fed
back to the controller.


## Which plotting command includes the estimator comparison?

`plot_state_estimator_comparison.py` is the dedicated estimator plotter.

`plot_experiment.py` is the run-level orchestrator and calls it automatically.

The maintained `plot_formation_experiment.py` also calls it automatically by
default when plotting a mission NPZ. Use `--no-estimator-comparison` only when
formation/control figures are desired in isolation.
