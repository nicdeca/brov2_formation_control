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

    def publish_fallback(self) -> None:
        self._fallback.publish(Bool(data=True))
