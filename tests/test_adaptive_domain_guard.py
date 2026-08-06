import numpy as np
import pytest

from formation_control.control import (
    AdaptiveDomainLimitError,
    project_adaptive_domain,
)


def test_guard_does_nothing_inside_current_adaptive_domain():
    rho = np.array([0.1, 0.2, 0.05, 0.03])
    base = np.array([0.4, 0.3, 0.2, 0.1])
    maximum = np.ones(4)

    result = project_adaptive_domain(
        rho,
        base,
        maximum,
        margin=1e-6,
    )

    np.testing.assert_allclose(result.enlargement, rho)
    np.testing.assert_allclose(result.correction, np.zeros(4))
    assert not result.activated


def test_guard_applies_minimum_monotone_enlargement():
    rho = np.array([0.0, 0.1, 0.0, 0.0])
    base = np.array([0.2, -0.15, -0.02, 0.3])
    maximum = np.array([0.5, 0.7, 0.4, 0.4])
    margin = 1e-3

    result = project_adaptive_domain(
        rho,
        base,
        maximum,
        margin=margin,
    )

    expected = np.array(
        [
            0.0,
            0.151,
            0.021,
            0.0,
        ]
    )
    np.testing.assert_allclose(result.enlargement, expected)
    assert result.activated

    np.testing.assert_array_less(
        np.full(4, margin - 1e-12),
        base + result.enlargement,
    )


def test_guard_never_shrinks_existing_enlargement():
    rho = np.array([0.3, 0.2, 0.1, 0.15])
    base = np.array([1.0, 1.0, 1.0, 1.0])
    maximum = np.array([0.5, 0.5, 0.5, 0.5])

    result = project_adaptive_domain(rho, base, maximum)

    assert np.all(result.enlargement >= rho)


def test_disabled_channel_is_not_modified():
    rho = np.zeros(4)
    base = np.array([-0.2, 0.3, 0.4, 0.5])
    maximum = np.array([0.5, 0.5, 0.5, 0.5])

    result = project_adaptive_domain(
        rho,
        base,
        maximum,
        enabled=np.array([False, True, True, True]),
    )

    assert result.enlargement[0] == pytest.approx(0.0)


def test_guard_reports_exhausted_physical_reserve():
    rho = np.zeros(4)
    base = np.array([0.2, -0.8, 0.2, 0.2])
    maximum = np.array([0.5, 0.6, 0.5, 0.5])

    with pytest.raises(AdaptiveDomainLimitError):
        project_adaptive_domain(
            rho,
            base,
            maximum,
            margin=1e-6,
        )
