"""Thin ROS publisher for plot-complete controller diagnostics."""

from __future__ import annotations

from collections.abc import Mapping

from std_msgs.msg import Float64MultiArray

from formation_control.experiment import pack_snapshot


class DiagnosticSnapshotPublisher:
    """Publish one atomic versioned diagnostic vector per control evaluation."""

    def __init__(
        self,
        node,
        *,
        topic: str = "formation_control/diagnostic_snapshot",
        qos_depth: int = 10,
    ) -> None:
        self._publisher = node.create_publisher(
            Float64MultiArray,
            topic,
            qos_depth,
        )

    def publish(self, values: Mapping[str, object]) -> None:
        message = Float64MultiArray()
        message.data = pack_snapshot(values).tolist()
        self._publisher.publish(message)

    def publish_fallback(self, *, role: float) -> None:
        """Publish a schema-valid partial sample when control falls back."""
        self.publish({"role": float(role), "fallback": 1.0})
