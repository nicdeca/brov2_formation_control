# Pool coordinate frame

This page is the authoritative coordinate reference for the current SITL and
real-water experiment workflow.

## Intended real-pool frame

The working Marinarium convention is PX4 local NED with:

```text
origin: kitchen-side edge, center of the pool in the lateral direction,
        at the water surface
+X:     along the pool away from the kitchen
+Y:     across the pool
+Z:     downward
```

The reported approximate real-pool extent is

```text
X_NED:  0 ... 9.0 m
Y_NED: -2.5 ... 2.5 m
Z_NED:  0 ... 3.0 m
```

The exact lateral sign (`+Y` wall) should still be checked against the real
localization frame before relying on absolute lateral coordinates.

## SITL alignment

The Gazebo tank has **not been enlarged**. Instead, the world/tank was rigidly
re-anchored so that PX4 local NED has the same intended origin and axis
orientation as the real pool.

The current simulated water volume is approximately

```text
X_NED:  0 ... 7.40 m
Y_NED: -2.275 ... 2.275 m
Z_NED:  0 ... 2.55 m
```

The tank pose in the pool-aligned Gazebo world is

```text
1.15 1.525 -2.575 0 0 -1.57079632679
```

and the water surface is at Gazebo world `z = 0`. Gazebo's ENU convention is an
internal simulator detail; controllers interface only with PX4.

## Controller core convention

PX4 provides NED/FRD. The controller converts this to NWU/FLU with

```text
S = diag(1, -1, -1)
p_NWU = S p_NED
```

Therefore the simulated pool appears in the controller core approximately as

```text
x_core:  0 ... 7.40 m
y_core: -2.275 ... 2.275 m
z_core: -2.55 ... 0 m
```

Depth is negative in the controller core because `+z_core` points upward.

## Current shared workspace subset

The maintained two-robot, adaptive-mission and five-robot launches use the
same transformed workspace:

```text
physical lower      = [0.300, -1.975, -2.155]
physical upper      = [7.100,  1.975,  0.225]

conservative lower  = [0.450, -1.825, -1.955]
conservative upper  = [6.950,  1.825, -0.325]
```

These values are the rigid transform of the previously validated safe subset.
They are **not** a claim about the full physical dimensions of the larger real
pool.

The positive physical `z` upper bound (`+0.225 m` in core NWU) is also inherited
from that exact transform and permits a limited surface-crossing margin. If the
vehicle center must remain strictly submerged, change this as a separate
safety-design decision rather than as part of frame alignment.

## Current two-robot initialization

In pool-aligned controller NWU:

```text
leader   = [2.675, 0.050, -0.775]
follower = [4.475, 0.750, -0.775]

d_21 = p_1 - p_2 = [-1.800, -0.700, 0.000]
```

The default SITL spawn is intentionally not identical to the initialization
target, so `INITIALIZE` still performs a real maneuver.

For the first default SITL robot, raw PX4 NED should be approximately

```text
[2.675, -1.05, 1.275] m
```

before initialization motion.

## Verification before hardware experiments

Before arming, check at least one raw PX4 sample and physically confirm:

- near the kitchen-side edge, `x_NED` is close to zero;
- moving away from the kitchen increases `x_NED`;
- the lateral centerline is close to `y_NED = 0`;
- moving deeper increases `z_NED`;
- the sign of `+y_NED` agrees with the real localization frame.

Do not add another coordinate correction inside the controller merely to make
plots look familiar. If the PX4 local frame is wrong, fix the localization/SITL
origin-orientation at the source.
