# Frozen-s CLF + independent funnel relaxation

Physical CLF:

    W_x = V(x, s) + 1/2 e_nu^T M e_nu,

with `s` treated as a frozen scheduling parameter.

The CLF-QP optimizes only physical inputs. In thruster space the decision
dimension is again 8. Neither `1/2 s^T Q_s s` nor `(partial V / partial s) v`
is included in the CLF inequality.

Auxiliary funnel dynamics:

    v_ref = -K_s s,

with

    0 <= s + dt v <= 1,

    h_a + dt (h_c_dot + rho_max v) >= epsilon.

The selected rate is the closed-form projection

    v* = clip(v_ref, v_lower, v_upper).

Thus positive expansion occurs only when required to keep the logarithmic
barrier domain well defined.

Removed CLI parameters:

    --relaxation-weight
    --relaxation-rate-scale
    --relaxation-rate-limit
    --relaxation-state-lyapunov-weight

Remaining funnel parameter:

    --relaxation-recovery-gain
