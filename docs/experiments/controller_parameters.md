# Controller and experiment parameters

This page summarizes the main tuning parameters exposed by the current ROS
controller.

## Geometric/task gains

### `position_gain`

Leader position-potential gain.

Typical validated SITL value:

```text
3.0
```

Higher values make leader position errors produce stronger virtual motion.

### `formation_gain`

Follower relative-position-potential gain.

Typical validated SITL value:

```text
3.0
```

Higher values make formation error produce a stronger virtual command.

## Backstepping virtual-velocity gain

The controller constructs a virtual generalized velocity from the configuration
potential gradient. The exposed isotropic translational/angular parameters are:

```text
virtual_linear_gain
virtual_angular_gain
```

Defaults preserving the original controller behavior:

```text
virtual_linear_gain  = 0.55
virtual_angular_gain = 0.80
```

These correspond to the translational/angular entries of `K_eta`.

## Command-filter bandwidth

The first-order generalized-velocity command filter uses:

```text
command_filter_linear_bandwidth
command_filter_angular_bandwidth
```

Defaults:

```text
command_filter_linear_bandwidth  = 3.0
command_filter_angular_bandwidth = 4.0
```

Larger values make the filtered generalized-velocity command follow the virtual
velocity more rapidly, but can also create larger required accelerations/wrench
transients after abrupt formation switches.

## CLF decay

```text
alpha_gain
```

Default:

```text
0.8
```

This controls the requested CLF decay rate in the QP. Increasing it makes the
dynamic CLF constraint more aggressive and can increase actuator demand.

## Virtual speed limits

```text
virtual_linear_speed_limit  = 1.5
virtual_angular_speed_limit = 2.0
```

These limit the virtual generalized-velocity command.

## Workspace barrier

Main parameters:

```text
workspace_barrier_enabled
workspace_adaptive
workspace_physical_lower
workspace_physical_upper
workspace_conservative_lower
workspace_conservative_upper
workspace_barrier_weight
workspace_reference_margin
workspace_relaxation_recovery_gain
workspace_relaxation_domain_margin_ratio
workspace_minimum_constraint_margin
```

Modes:

```text
workspace_barrier_enabled=false
    no workspace barrier

workspace_barrier_enabled=true
workspace_adaptive=false
    fixed conservative workspace

workspace_barrier_enabled=true
workspace_adaptive=true
    conservative workspace may relax toward physical safe bounds
```

## Sensing-domain parameters

Range:

```text
d_min                  = 0.5
d_max                  = 3.6
d_min_conservative     = 0.8
d_max_conservative     = 3.0
```

Conservative normalized FoV:

```text
alpha_h_conservative = 0.72
alpha_v_conservative = 0.72
```

The adaptive sensing state has four channels:

```text
[collision, range, horizontal_fov, vertical_fov]
```

## Initialization handoff

Typical SITL values:

```text
position_tolerance = 0.65 m
speed_tolerance    = 0.08 m/s
settle_time        = 1.5 s
```

The loose absolute-position tolerance is intentional: initialization establishes
a safe/slow starting geometry rather than precision absolute station keeping.

## Recommended tuning order

Do not change everything at once.

Suggested order:

1. verify frames and references;
2. tune `position_gain` / `formation_gain`;
3. inspect actuator saturation;
4. tune `virtual_linear_gain`;
5. tune `command_filter_linear_bandwidth`;
6. tune `alpha_gain`;
7. only then consider angular-side tuning if needed.

Always compare:

- formation / leader tracking;
- thruster forces and force limits;
- CLF slack / actuation margin;
- sensing relaxation;
- workspace relaxation and physical margin.
