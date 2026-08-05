from formation_control.control import CLFQP, LinearClassK


def test_osqp_defaults_are_numerically_strict():
    qp = CLFQP.isotropic(
        control_dim=1,
        control_weight=1.0,
        slack_penalty=1.0,
        alpha=LinearClassK(gain=1.0),
    )

    assert qp.solver_options["eps_abs"] <= 1e-7
    assert qp.solver_options["eps_rel"] <= 1e-7
    assert qp.solver_options["max_iter"] >= 100_000
    assert qp.solver_options["polishing"] is True


def test_user_solver_options_override_defaults():
    qp = CLFQP.isotropic(
        control_dim=1,
        control_weight=1.0,
        slack_penalty=1.0,
        alpha=LinearClassK(gain=1.0),
        solver_options={"eps_abs": 1e-5, "max_iter": 12_345},
    )

    assert qp.solver_options["eps_abs"] == 1e-5
    assert qp.solver_options["max_iter"] == 12_345
    assert qp.solver_options["polishing"] is True
