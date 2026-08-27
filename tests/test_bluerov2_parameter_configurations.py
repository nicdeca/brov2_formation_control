import numpy as np
import pytest

from formation_control.models.bluerov2 import (
    BlueROV2Model,
    BlueROV2Parameters,
)


def test_default_parameters_are_gazebo_configuration():
    default = BlueROV2Parameters()
    gazebo = BlueROV2Parameters.gazebo()
    model = BlueROV2Model()

    assert default == gazebo
    assert model.parameters == gazebo
    assert default.gravity == pytest.approx(9.8)
    assert default.mass == pytest.approx(13.4)
    assert default.volume == pytest.approx(0.0135)
    np.testing.assert_allclose(
        default.rigid_body_inertia_diagonal,
        np.array([0.2631892537313433, 0.2293092537313433, 0.37528]),
    )
    np.testing.assert_allclose(
        default.center_of_buoyancy,
        np.array([0.0, 0.0, -0.0208955223880597]),
    )
    np.testing.assert_allclose(
        default.added_mass_diagonal,
        np.array([1.272, 1.424, 3.736, 0.0378, 0.027, 0.0444]),
    )
    np.testing.assert_allclose(
        default.linear_damping_diagonal,
        np.array([13.7, 1.0, 23.0, 0.5, 0.8, 0.1]),
    )
    np.testing.assert_allclose(
        default.quadratic_damping_diagonal,
        np.array([14.1, 21.7, 19.0, 0.119, 0.047, 0.15]),
    )


def test_standard_parameters_match_reference_executable_values():
    parameters = BlueROV2Parameters.standard()

    assert parameters.gravity == pytest.approx(9.82)
    assert parameters.mass == pytest.approx(12.5)
    assert parameters.volume == pytest.approx(0.0135)
    np.testing.assert_allclose(
        parameters.rigid_body_inertia_diagonal,
        np.array(
            [
                0.25 * (12.5 / 13.0),
                0.221 * (12.5 / 13.0),
                0.356 * (12.5 / 13.0),
            ]
        ),
    )
    np.testing.assert_allclose(
        parameters.added_mass_diagonal,
        np.array([1.272, 1.424, 3.736, 0.0378, 0.027, 0.044]),
    )
    np.testing.assert_allclose(
        parameters.linear_damping_diagonal,
        np.array([13.7, 0.0, 33.0, 0.0, 0.8, 0.0]),
    )
    np.testing.assert_allclose(
        parameters.quadratic_damping_diagonal,
        np.array([141.0, 217.0, 190.0, 1.19, 0.47, 1.5]),
    )


def test_heavy_tube_parameters_match_reference_executable_values():
    parameters = BlueROV2Parameters.heavy_tube()

    assert parameters.mass == pytest.approx(16.4)
    assert parameters.volume == pytest.approx(0.0134)
    np.testing.assert_allclose(
        parameters.rigid_body_inertia_diagonal,
        np.array(
            [
                0.25 * (16.4 / 13.0),
                0.221 * (16.4 / 13.0),
                0.356 * (16.4 / 13.0),
            ]
        ),
    )
    np.testing.assert_allclose(
        parameters.added_mass_diagonal,
        np.array([6.36, 7.12, 18.68, 0.189, 0.135, 0.222]),
    )
    np.testing.assert_allclose(
        parameters.linear_damping_diagonal,
        np.array([13.7, 0.0, 33.0, 0.0, 0.8, 0.0]),
    )
    np.testing.assert_allclose(
        parameters.quadratic_damping_diagonal,
        np.array([141.0, 217.0, 190.0, 1.19, 0.47, 1.5]),
    )


@pytest.mark.parametrize("configuration", ["gazebo", "standard", "heavy_tube"])
def test_named_configuration_constructs_valid_model(configuration):
    parameters = BlueROV2Parameters.from_configuration(configuration)
    model = BlueROV2Model(parameters)

    assert np.all(np.linalg.eigvalsh(model.mass_matrix) > 0.0)


def test_unknown_configuration_is_rejected():
    with pytest.raises(ValueError, match="gazebo.*standard.*heavy_tube"):
        BlueROV2Parameters.from_configuration("unknown")
