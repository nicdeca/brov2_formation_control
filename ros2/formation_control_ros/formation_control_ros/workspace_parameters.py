"""ROS parameter helpers for the ROS-independent workspace barrier config."""

from __future__ import annotations

from rclpy.node import Node

from .core_runtime import WorkspaceConfig


def declare_workspace_parameters(node: Node) -> None:
    """Declare switchable Marinarium workspace-barrier parameters."""
    node.declare_parameter("workspace_barrier_enabled", False)
    node.declare_parameter("workspace_adaptive", True)

    # Robot-center bounds in core NWU.  Physical bounds are conservative with
    # respect to the actual rigid tank walls; the upper-z bound is an
    # operational near-surface limit rather than a rigid ceiling.
    node.declare_parameter(
        "workspace_physical_lower",
        [-3.125, -1.225, -96.58],
    )
    node.declare_parameter(
        "workspace_physical_upper",
        [0.825, 5.575, -94.20],
    )
    node.declare_parameter(
        "workspace_conservative_lower",
        [-2.975, -1.075, -96.38],
    )
    node.declare_parameter(
        "workspace_conservative_upper",
        [0.675, 5.425, -94.75],
    )

    node.declare_parameter("workspace_barrier_weight", 0.10)
    node.declare_parameter("workspace_reference_margin", 0.05)
    node.declare_parameter("workspace_relaxation_recovery_gain", 0.8)
    node.declare_parameter("workspace_relaxation_domain_margin_ratio", 0.10)
    node.declare_parameter("workspace_minimum_constraint_margin", 1e-3)


def workspace_config_from_parameters(node: Node) -> WorkspaceConfig:
    def vector3(name: str) -> tuple[float, float, float]:
        values = tuple(float(value) for value in node.get_parameter(name).value)
        if len(values) != 3:
            raise ValueError(f"{name} must contain three values.")
        return values

    return WorkspaceConfig(
        enabled=bool(node.get_parameter("workspace_barrier_enabled").value),
        adaptive=bool(node.get_parameter("workspace_adaptive").value),
        physical_lower=vector3("workspace_physical_lower"),
        physical_upper=vector3("workspace_physical_upper"),
        conservative_lower=vector3("workspace_conservative_lower"),
        conservative_upper=vector3("workspace_conservative_upper"),
        barrier_weight=float(node.get_parameter("workspace_barrier_weight").value),
        reference_margin=float(node.get_parameter("workspace_reference_margin").value),
        relaxation_recovery_gain=float(
            node.get_parameter("workspace_relaxation_recovery_gain").value
        ),
        relaxation_domain_margin_ratio=float(
            node.get_parameter("workspace_relaxation_domain_margin_ratio").value
        ),
        minimum_constraint_margin=float(
            node.get_parameter("workspace_minimum_constraint_margin").value
        ),
    )
