import numpy as np
import pytest

from formation_control.control import LinearClassK, SaturatingClassK


def test_linear_class_k():
    alpha = LinearClassK(gain=2.5)

    assert alpha(0.0) == pytest.approx(0.0)
    assert alpha(1.2) == pytest.approx(3.0)


def test_saturating_class_k():
    alpha = SaturatingClassK(gain=2.0, maximum=3.0)

    assert alpha(0.0) == pytest.approx(0.0)
    assert 0.0 < alpha(1.0) < 3.0
    assert alpha(10.0) < 3.0
    assert alpha(10.0) > alpha(1.0)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: LinearClassK(gain=1.0),
        lambda: SaturatingClassK(gain=1.0, maximum=2.0),
    ],
)
def test_class_k_rejects_negative_argument(factory):
    with pytest.raises(ValueError):
        factory()(-1.0)


def test_class_k_parameters_must_be_positive():
    with pytest.raises(ValueError):
        LinearClassK(gain=0.0)

    with pytest.raises(ValueError):
        SaturatingClassK(gain=1.0, maximum=np.inf)
