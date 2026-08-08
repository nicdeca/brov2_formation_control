import numpy as np
import pytest

from formation_control.visualization import fov_domain_relaxation_curves


def test_fov_domain_relaxation_curves_match_conservative_and_physical_limits():
    alpha_conservative = 0.72
    rho_max = 1.0 - alpha_conservative**2

    curves = fov_domain_relaxation_curves(
        np.array([0.0, rho_max]),
        alpha_conservative=alpha_conservative,
        maximum_enlargement=rho_max,
    )

    assert curves.normalized_relaxation[0] == pytest.approx(0.0)
    assert curves.normalized_relaxation[1] == pytest.approx(1.0)
    assert curves.adaptive_limit[0] == pytest.approx(alpha_conservative)
    assert curves.adaptive_limit[1] == pytest.approx(1.0)


def test_fov_domain_relaxation_curves_match_intermediate_enlargement():
    alpha_conservative = 0.72
    rho_max = 1.0 - alpha_conservative**2
    rho = np.array([0.35 * rho_max])

    curves = fov_domain_relaxation_curves(
        rho,
        alpha_conservative=alpha_conservative,
        maximum_enlargement=rho_max,
    )

    expected_limit = np.sqrt(alpha_conservative**2 + rho[0])

    assert curves.normalized_relaxation[0] == pytest.approx(0.35)
    assert curves.adaptive_limit[0] == pytest.approx(expected_limit)
