"""ROS 2 diagnostics published with only standard message types."""

from __future__ import annotations

import math

from rclpy.node import Node
from std_msgs.msg import Bool, Float64, Float64MultiArray

from .core_runtime import ControllerDiagnostics


class DiagnosticsPublisher:
    def __init__(self, node: Node) -> None:
        self._fallback = node.create_publisher(
            Bool,
            "formation_control/fallback",
            10,
        )
        self._slack = node.create_publisher(
            Float64,
            "formation_control/slack",
            10,
        )
        self._required_slack = node.create_publisher(
            Float64,
            "formation_control/required_slack",
            10,
        )
        self._actuation_margin = node.create_publisher(
            Float64,
            "formation_control/actuation_margin",
            10,
        )
        self._utilization = node.create_publisher(
            Float64,
            "formation_control/thruster_utilization",
            10,
        )
        self._physical_margin = node.create_publisher(
            Float64,
            "formation_control/minimum_physical_margin",
            10,
        )
        self._relaxation = node.create_publisher(
            Float64MultiArray,
            "formation_control/domain_relaxation",
            10,
        )
        self._conservative_values = node.create_publisher(
            Float64MultiArray,
            "formation_control/conservative_constraint_values",
            10,
        )

        # Workspace diagnostics intentionally use separate topics so the
        # existing four-channel sensing-domain topic keeps its meaning.
        self._workspace_enabled = node.create_publisher(
            Bool,
            "formation_control/workspace_barrier_enabled",
            10,
        )
        self._workspace_adaptive = node.create_publisher(
            Bool,
            "formation_control/workspace_adaptive",
            10,
        )
        self._workspace_barrier_value = node.create_publisher(
            Float64,
            "formation_control/workspace_barrier_value",
            10,
        )
        self._workspace_relaxation = node.create_publisher(
            Float64MultiArray,
            "formation_control/workspace_relaxation",
            10,
        )
        self._workspace_conservative_values = node.create_publisher(
            Float64MultiArray,
            "formation_control/workspace_conservative_constraint_values",
            10,
        )
        self._workspace_physical_values = node.create_publisher(
            Float64MultiArray,
            "formation_control/workspace_physical_constraint_values",
            10,
        )
        self._workspace_physical_margin = node.create_publisher(
            Float64,
            "formation_control/workspace_minimum_physical_margin",
            10,
        )

    def publish(
        self,
        diagnostics: ControllerDiagnostics,
        *,
        fallback: bool,
    ) -> None:
        self._fallback.publish(Bool(data=fallback))
        self._slack.publish(Float64(data=diagnostics.slack))
        self._required_slack.publish(
            Float64(
                data=(
                    diagnostics.required_slack
                    if diagnostics.required_slack is not None
                    else math.nan
                )
            )
        )
        self._actuation_margin.publish(
            Float64(
                data=(
                    diagnostics.actuation_margin
                    if diagnostics.actuation_margin is not None
                    else math.nan
                )
            )
        )
        self._utilization.publish(
            Float64(data=diagnostics.thruster_utilization)
        )
        self._physical_margin.publish(
            Float64(data=diagnostics.minimum_physical_margin)
        )

        relaxation = Float64MultiArray()
        relaxation.data = [
            float(value) for value in diagnostics.relaxation_state
        ]
        self._relaxation.publish(relaxation)

        conservative = Float64MultiArray()
        conservative.data = [
            float(value)
            for value in diagnostics.conservative_constraint_values
        ]
        self._conservative_values.publish(conservative)

        self._workspace_enabled.publish(
            Bool(data=diagnostics.workspace_barrier_enabled)
        )
        self._workspace_adaptive.publish(
            Bool(data=diagnostics.workspace_adaptive)
        )
        self._workspace_barrier_value.publish(
            Float64(data=diagnostics.workspace_barrier_value)
        )

        workspace_relaxation = Float64MultiArray()
        workspace_relaxation.data = [
            float(value)
            for value in diagnostics.workspace_relaxation_state
        ]
        self._workspace_relaxation.publish(workspace_relaxation)

        workspace_conservative = Float64MultiArray()
        workspace_conservative.data = [
            float(value)
            for value in diagnostics.workspace_conservative_constraint_values
        ]
        self._workspace_conservative_values.publish(workspace_conservative)

        workspace_physical = Float64MultiArray()
        workspace_physical.data = [
            float(value)
            for value in diagnostics.workspace_physical_constraint_values
        ]
        self._workspace_physical_values.publish(workspace_physical)
        self._workspace_physical_margin.publish(
            Float64(data=diagnostics.workspace_minimum_physical_margin)
        )

    def publish_fallback(self) -> None:
        self._fallback.publish(Bool(data=True))
