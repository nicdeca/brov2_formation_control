# Practical adaptive-domain margin

The auxiliary funnel controller now distinguishes between:

1. a tiny absolute logarithmic-domain floor

       epsilon_abs = 1e-5,

   used only by the emergency sampled-data safeguard; and

2. a practical CBF target margin

       h_margin,l = max(
           epsilon_abs,
           margin_ratio * rho_max,l
       ),

   with default

       margin_ratio = 0.01.

The sampled auxiliary condition is therefore

    h_a,k
    + dt (h_c_dot,k + rho_max v_k)
    >= h_margin.

The emergency safeguard no longer fires merely because the practical margin
is undershot.  It activates only when the realized `h_a` reaches the tiny
absolute floor.  If that happens, it projects `s` back to the practical
channel-wise margin so the following logarithmic evaluation is numerically
well conditioned.

The BlueROV example exposes

    --relaxation-domain-margin-ratio

with default 0.01.
