"""Admissible-domain and scalar-constraint primitives."""

from formation_control.geometry import NormalizedImagePoint

from .base import ConstraintEvaluation, ScalarConstraint
from .distance import MaximumDistanceConstraint, MinimumDistanceConstraint
from .domains import DistanceDomain, FieldOfViewDomain
from .fov import (
    HorizontalFieldOfViewConstraint,
    PositiveDepthConstraint,
    VerticalFieldOfViewConstraint,
)
from .sensing_kinematics import (
    SensingConstraintKinematics,
    evaluate_sensing_constraint_kinematics,
)

__all__ = [
    "ConstraintEvaluation",
    "DistanceDomain",
    "FieldOfViewDomain",
    "HorizontalFieldOfViewConstraint",
    "MaximumDistanceConstraint",
    "MinimumDistanceConstraint",
    "NormalizedImagePoint",
    "PositiveDepthConstraint",
    "ScalarConstraint",
    "SensingConstraintKinematics",
    "VerticalFieldOfViewConstraint",
    "evaluate_sensing_constraint_kinematics",
]
