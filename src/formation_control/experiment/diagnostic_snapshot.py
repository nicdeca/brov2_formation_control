"""Versioned, ROS-independent experiment diagnostic snapshot schema.

The ROS wrapper can publish this fixed-size vector as
``std_msgs/msg/Float64MultiArray``.  The schema deliberately lives in the core
package so online publication and offline bag decoding share one source of
truth without introducing any ROS dependency into :mod:`formation_control`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

SNAPSHOT_SCHEMA_VERSION = 1.0

# Serialization order.  Missing fields are represented by NaN so the same
# schema works for leaders (no sensing edge) and followers.
_FIELD_WIDTHS: tuple[tuple[str, int], ...] = (
    ("schema_version", 1),
    ("role", 1),  # 0=follower, 1=leader
    ("fallback", 1),
    ("controller_time_s", 1),
    ("slack", 1),
    ("required_slack", 1),
    ("actuation_margin", 1),
    ("thruster_utilization", 1),
    ("minimum_physical_margin", 1),
    # Run-defining sensing-domain parameters.
    ("distance_domain", 4),  # [d_min, d_max, d_min_c, d_max_c]
    ("fov_domain", 2),  # [alpha_h_c, alpha_v_c]
    ("constraints_enabled", 4),
    ("adaptive_enabled", 1),
    ("domain_margin_ratio", 1),
    # Camera model, to make trajectory/FoV plots self describing.
    ("camera_half_angles", 2),  # radians
    ("camera_position_body", 3),
    ("camera_rotation_camera_from_body", 9),
    # Edge/adaptation state. Channel order is always
    # [collision, range, horizontal_fov, vertical_fov].
    ("conservative_values", 4),
    ("physical_values", 4),
    ("relaxation_state", 4),  # normalized s in [0,1]
    ("relaxation_rate", 4),  # normalized s_dot
    ("image_coordinates", 2),  # [alpha_h, alpha_v]
    ("desired_relative_position", 3),  # parent - follower, core NWU
    ("relative_position", 3),  # parent - follower, core NWU
    # Physical command.
    ("thruster_forces", 8),
    ("thruster_force_limits", 2),  # [reverse_magnitude, forward]
    ("body_wrench", 6),
    # Leader/reference fields. Followers leave these as NaN.
    ("reference_position", 3),
    ("reference_velocity", 3),
    ("reference_acceleration", 3),
    ("reference_quaternion", 4),  # scalar-first
    # Detailed physical CLF decomposition, matching CLFDiagnosticHistory in
    # examples/04_bluerov2_fov_clf_qp.py.
    ("clf_value", 1),
    ("clf_decay", 1),
    ("clf_drift", 1),
    ("clf_configuration_local_rate", 1),
    ("clf_parent_rate", 1),
    ("clf_velocity_backstepping_rate", 1),
    ("clf_dynamics_bias_rate", 1),
    ("clf_dynamics_bias_linear_rate", 1),
    ("clf_dynamics_bias_angular_rate", 1),
    ("clf_command_acceleration_rate", 1),
    ("clf_command_acceleration_linear_rate", 1),
    ("clf_command_acceleration_angular_rate", 1),
    ("clf_velocity_error_norm", 1),
    ("clf_velocity_error_linear_norm", 1),
    ("clf_velocity_error_angular_norm", 1),
    ("clf_filtered_velocity_derivative_norm", 1),
    ("clf_filtered_linear_acceleration_norm", 1),
    ("clf_filtered_angular_acceleration_norm", 1),
    ("clf_best_actuator_contribution", 1),
    ("clf_minimum_modeled_derivative", 1),
    ("clf_hard_residual", 1),
    # Vector-valued CLF diagnostics.
    ("generalized_velocity", 6),
    ("filtered_velocity", 6),
    ("desired_velocity", 6),
    ("filtered_velocity_derivative", 6),
    ("dynamics_bias", 6),
)

FIELD_WIDTHS = dict(_FIELD_WIDTHS)
FIELD_SLICES: dict[str, slice] = {}
_cursor = 0
for _name, _width in _FIELD_WIDTHS:
    FIELD_SLICES[_name] = slice(_cursor, _cursor + _width)
    _cursor += _width
SNAPSHOT_SIZE = _cursor


def field_names() -> tuple[str, ...]:
    """Return fields in serialization order."""
    return tuple(name for name, _ in _FIELD_WIDTHS)


def _as_width(value: object, width: int, *, name: str) -> FloatArray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != width:
        raise ValueError(
            f"{name} must contain {width} value(s), got {array.size}."
        )
    return array


def pack_snapshot(values: Mapping[str, object]) -> FloatArray:
    """Pack named values into the fixed version-1 snapshot vector.

    Unspecified fields are NaN. This is intentional: a leader has no camera
    edge, while a fallback sample may only have a subset of controller data.
    """
    unknown = set(values) - set(FIELD_WIDTHS)
    if unknown:
        raise KeyError(f"unknown diagnostic snapshot fields: {sorted(unknown)}")

    vector = np.full(SNAPSHOT_SIZE, np.nan, dtype=float)
    vector[FIELD_SLICES["schema_version"]] = SNAPSHOT_SCHEMA_VERSION

    for name, width in _FIELD_WIDTHS:
        if name == "schema_version" or name not in values:
            continue
        vector[FIELD_SLICES[name]] = _as_width(
            values[name],
            width,
            name=name,
        )

    return vector


def unpack_snapshot(
    vector: Sequence[float] | FloatArray,
) -> dict[str, FloatArray | float]:
    """Decode one version-1 diagnostic vector."""
    array = np.asarray(vector, dtype=float).reshape(-1)
    if array.size != SNAPSHOT_SIZE:
        raise ValueError(
            f"diagnostic snapshot has size {array.size}, expected {SNAPSHOT_SIZE}."
        )

    version = float(array[FIELD_SLICES["schema_version"]][0])
    if not np.isclose(version, SNAPSHOT_SCHEMA_VERSION):
        raise ValueError(
            f"unsupported diagnostic snapshot version {version}; "
            f"expected {SNAPSHOT_SCHEMA_VERSION}."
        )

    decoded: dict[str, FloatArray | float] = {}
    for name, width in _FIELD_WIDTHS:
        data = array[FIELD_SLICES[name]].copy()
        decoded[name] = float(data[0]) if width == 1 else data
    return decoded
