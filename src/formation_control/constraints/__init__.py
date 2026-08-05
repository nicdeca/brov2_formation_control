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
    "VerticalFieldOfViewConstraint",
]
