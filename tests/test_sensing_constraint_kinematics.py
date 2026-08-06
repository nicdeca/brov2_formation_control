import numpy as np

from formation_control.constraints import (
    DistanceDomain,
    FieldOfViewDomain,
    evaluate_sensing_constraint_kinematics,
)
from formation_control.geometry import PinholeCamera
from formation_control.models import BlueROV2Model


def test_distance_constraint_rates_have_opposite_signs():
    model = BlueROV2Model()
    camera = PinholeCamera.from_degrees(45.0, 30.0)
    distance = DistanceDomain(
        d_min=0.5,
        d_max=3.6,
        d_min_conservative=0.8,
        d_max_conservative=3.0,
    )
    fov = FieldOfViewDomain(
        alpha_h_conservative=0.72,
        alpha_v_conservative=0.72,
    )

    observer = np.zeros(13)
    target = np.zeros(13)
    observer[3] = 1.0
    target[3] = 1.0
    target[0] = 2.0
    target[7] = 0.5

    evaluation = evaluate_sensing_constraint_kinematics(
        model,
        camera,
        distance,
        fov,
        observer,
        target,
    )

    assert evaluation.rates[0] == 2.0
    assert evaluation.rates[1] == -2.0
    np.testing.assert_allclose(evaluation.image_coordinates, np.zeros(2))
    np.testing.assert_allclose(evaluation.image_rates, np.zeros(2))
