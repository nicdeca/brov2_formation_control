import math

import numpy as np

from formation_control_ros.core_runtime import ControllerDiagnostics


def test_controller_diagnostic_scalars_are_float_convertible():
    diagnostics = ControllerDiagnostics(
        slack=np.float64(0.1),
        required_slack=np.float64(0.2),
        actuation_margin=np.float64(0.3),
        thruster_utilization=np.float64(0.4),
        relaxation_state=np.array([0.0, 0.1]),
        conservative_constraint_values=np.array([0.2, 0.3]),
        minimum_physical_margin=np.float64(0.5),
    )

    assert math.isfinite(float(diagnostics.slack))
    assert math.isfinite(float(diagnostics.required_slack))
    assert math.isfinite(float(diagnostics.actuation_margin))
    assert math.isfinite(float(diagnostics.thruster_utilization))
    assert math.isfinite(float(diagnostics.minimum_physical_margin))
