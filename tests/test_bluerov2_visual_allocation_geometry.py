import numpy as np

from formation_control.actuation import BlueROV2HeavyThrusterConfiguration
from formation_control.visualization import BlueROV2HeavyVisualGeometry


def test_visual_geometry_uses_actuation_thruster_configuration():
    configuration = BlueROV2HeavyThrusterConfiguration.default_45deg(
        voltage=20,
        derating=0.75,
    )
    visual = BlueROV2HeavyVisualGeometry(thruster_configuration=configuration)

    assert visual.thruster_configuration is configuration
    np.testing.assert_allclose(
        visual.thruster_configuration.positions_body,
        configuration.positions_body,
    )
    np.testing.assert_allclose(
        visual.thruster_configuration.directions_body,
        configuration.directions_body,
    )
