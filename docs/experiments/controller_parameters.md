# Controller and experiment parameters

This page summarizes the parameters most relevant to the maintained experiment
launches. Launch-specific defaults take precedence over generic/core defaults.

## Dynamics preset

The physical/model presets are:

```text
gazebo      PX4/Gazebo SITL model
standard    characterized standard laboratory configuration
heavy_tube  characterized heavy-tube configuration
```

At the ROS layer, `auto` maps:

```text
glub    -> heavy_tube
splash  -> heavy_tube
bubble  -> standard
```

Unknown names fail explicitly in `auto` mode. For SITL with physical-looking
names such as `splash` and `glub`, explicitly select `gazebo` so the controller
does not use the real-hardware dynamics model.

## Main gain defaults by maintained launch

### Normal two-robot experiment

```text
position_gain                     = 2.0
formation_gain                    = 2.0
virtual_linear_gain               = 0.55
virtual_angular_gain              = 0.80
command_filter_linear_bandwidth   = 3.0
command_filter_angular_bandwidth  = 4.0
alpha_gain                        = 1.8
```

### Dedicated adaptive two-robot mission

```text
position_gain                     = 2.0
formation_gain                    = 1.4
virtual_linear_gain               = 0.55
virtual_angular_gain              = 0.80
command_filter_linear_bandwidth   = 3.0
command_filter_angular_bandwidth  = 4.0
alpha_gain                        = 0.8
```

### Five-robot tree SITL

```text
position_gain                     = 2.0
formation_gain                    = 1.0
virtual_linear_gain               = 1.0
virtual_angular_gain              = 1.2
command_filter_linear_bandwidth   = 10.0
command_filter_angular_bandwidth  = 10.0
alpha_gain                        = 3.0
```

Do not mix these values casually: the dedicated adaptive mission intentionally
uses a different sensing domain and controller tuning from the normal hardware
experiment.

## Command-filtered backstepping

The follower constructs a virtual generalized-velocity command from the
configuration/barrier gradient and tracks a first-order filtered version.
This avoids requiring the unavailable parent velocity to differentiate the
virtual command directly.

Larger virtual gains produce stronger configuration correction. Larger command
filter bandwidth follows the virtual command faster but may create larger
acceleration/wrench transients after formation switches.

## CLF-QP and actuation

The online optimization variable is the eight-thruster force vector, not a free
six-dimensional wrench. Thruster box constraints are enforced directly.

`alpha_gain` controls the requested CLF dissipation. Increasing it can increase
required actuation and CLF slack.

Always inspect:

- all eight thruster forces and signed limits;
- required zero-slack CLF slack;
- actuation margin;
- fallback state;
- controller execution time.

## Normal two-robot sensing domain

The normal two-robot formation library is documented against approximately:

```text
d_min               = 0.5 m
d_max               = 3.6 m
d_min_conservative  = 0.8 m
d_max_conservative  = 3.0 m
alpha_h_conservative = 0.72
alpha_v_conservative = 0.72
```

The `challenging` profile includes targets that intentionally enter the
relaxable conservative region while remaining inside the physical sensing
limits.

## Dedicated adaptive-mission sensing domain

`two_robot_adaptive_mission.launch.py` intentionally tightens the conservative
domain:

```text
d_min                  = 0.5 m
d_max                  = 3.6 m
d_min_conservative     = 0.8 m
d_max_conservative     = 2.4 m
alpha_h_conservative   = 0.45
alpha_v_conservative   = 0.45
```

Final smooth-adaptation tuning:

```text
relaxation_recovery_gain          = 0.8
relaxation_barrier_gain           = 0.20
relaxation_domain_margin_ratio    = 0.02
relaxation_activation_on_ratio    = 0.001
relaxation_activation_off_ratio   = 0.15
```

The adaptation is margin driven and does not require `h_c_dot` or parent
velocity. The normalized enlargement state is nonnegative and is **not**
clipped at one. `s = 1` corresponds to the zero set reaching the physical
limit; values above one therefore require explicit inspection of the physical
constraint margin.

## Workspace adaptation

The maintained launches currently use:

```text
workspace_physical_lower      = [0.300, -1.975, -2.155]
workspace_physical_upper      = [7.100,  1.975,  0.225]
workspace_conservative_lower  = [0.450, -1.825, -1.955]
workspace_conservative_upper  = [6.950,  1.825, -0.325]
workspace_barrier_weight      = 0.10
workspace_reference_margin    = 0.05
workspace_relaxation_recovery_gain       = 0.8
workspace_relaxation_domain_margin_ratio = 0.10
workspace_minimum_constraint_margin      = 1e-3
```

Modes:

```text
workspace_barrier_enabled=false
    no workspace barrier

workspace_barrier_enabled=true, workspace_adaptive=false
    fixed conservative workspace

workspace_barrier_enabled=true, workspace_adaptive=true
    conservative workspace can enlarge toward the physical bounds
```

Sensing-domain adaptation and workspace-domain adaptation are separate
mechanisms and should be diagnosed separately.

## Initialization handoff

The maintained phase-manager settings are

```text
position_tolerance = 0.65 m
speed_tolerance    = 0.08 m/s
settle_time        = 1.5 s
```

The loose position tolerance is intentional: initialization establishes a
safe, slow starting geometry rather than precision absolute station keeping.

## Recommended tuning order

1. verify pool/PX4/core frames and camera geometry;
2. verify initialization and workspace references;
3. tune position/formation gains;
4. inspect thruster saturation and actuation margin;
5. tune virtual gains and command-filter bandwidth;
6. tune CLF decay;
7. only then modify sensing/workspace adaptation parameters.
