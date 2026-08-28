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

        # Four-channel sensing-domain diagnostics.
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

        # Six-channel Stage-B workspace diagnostics.  These are deliberately
        # separate from the sensing-domain fields above.  The recorder and
        # initialization exporter already use these topic names.
        self._workspace_barrier_enabled = node.create_publisher(
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

    @staticmethod
    def _multi_array(values) -> Float64MultiArray:
        message = Float64MultiArray()
        message.data = [float(value) for value in values]
        return message

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
        self._relaxation.publish(
            self._multi_array(diagnostics.relaxation_state)
        )
        self._conservative_values.publish(
            self._multi_array(diagnostics.conservative_constraint_values)
        )

        self._workspace_barrier_enabled.publish(
            Bool(data=bool(diagnostics.workspace_barrier_enabled))
        )
        self._workspace_adaptive.publish(
            Bool(data=bool(diagnostics.workspace_adaptive))
        )
        self._workspace_barrier_value.publish(
            Float64(data=float(diagnostics.workspace_barrier_value))
        )
        self._workspace_relaxation.publish(
            self._multi_array(diagnostics.workspace_relaxation_state)
        )
        self._workspace_conservative_values.publish(
            self._multi_array(
                diagnostics.workspace_conservative_constraint_values
            )
        )
        self._workspace_physical_values.publish(
            self._multi_array(
                diagnostics.workspace_physical_constraint_values
            )
        )
        self._workspace_physical_margin.publish(
            Float64(data=float(diagnostics.workspace_minimum_physical_margin))
        )

    def publish_fallback(self) -> None:
        self._fallback.publish(Bool(data=True))
