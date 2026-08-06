# Finer velocity/backstepping CLF diagnostics

This patch is diagnostic only. It does not change the controller.

It splits

    -e_nu^T (h + M nu_c_dot)

into

    -e_nu^T h

and

    -e_nu^T M nu_c_dot,

with translational and rotational contributions recorded separately.

It also logs:

- `||e_nu||`, `||e_v||`, `||e_omega||`;
- `||nu_c_dot||`, `||v_c_dot||`, `||omega_c_dot||`;
- the six components of `nu`;
- the six components of `nu_c`;
- the six components of `nu_d`;
- the six components of `nu_c_dot`;
- the six components of the dynamics bias `h`.

New figures:

- `clf_velocity_backstepping_split`;
- `clf_velocity_backstepping_peak_detail`;
- `command_velocity_peak_detail`;
- `command_acceleration_peak_detail`.

The peak console report now prints the detailed split and the generalized
velocity vectors. The same quantities are exported in `clf_diagnostics.csv`
when `--save` is used.
