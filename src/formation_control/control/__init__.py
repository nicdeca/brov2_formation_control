"""Command filtering, virtual control, adaptation, and CLF-QP primitives."""

from .adaptation import (
    AdaptiveDomainDynamics,
    AdaptiveDomainState,
    AdaptiveEnlargementLaw,
)
from .backstepping_clf import BacksteppingCLF, BacksteppingCLFEvaluation
from .bluerov2 import BlueROV2AgentController, BlueROV2AgentEvaluation
from .bluerov2_design import (
    BlueROV2ControllerDesign,
    BlueROV2ControlSpace,
    build_bluerov2_controller_design,
)
from .class_k import ClassKFunction, LinearClassK, SaturatingClassK
from .clf_feasibility import (
    CLFActuationFeasibility,
    box_clf_actuation_feasibility,
)
from .clf_qp import (
    CLFQP,
    CLFQPProblem,
    CLFQPResult,
    CLFQPSolverError,
)
from .control_constraints import PolyhedralControlSet
from .double_integrator import (
    DoubleIntegratorAgentController,
    DoubleIntegratorAgentEvaluation,
)
from .filters import (
    CommandFilter,
    CommandFilterEvaluation,
    FirstOrderCommandFilter,
    SecondOrderCommandFilter,
)
from .second_order import (
    SecondOrderCLFQPController,
    SecondOrderControllerEvaluation,
)
from .virtual_control import (
    VirtualVelocityEvaluation,
    VirtualVelocityGains,
    generalized_configuration_gradient,
    virtual_velocity_command,
)
from .wrench_space import (
    WrenchSpaceControllerDesign,
    build_wrench_space_controller,
    equivalent_wrench_weight,
    minimum_effort_allocation_matrix,
    wrench_polytope_control_set,
)

__all__ = [
    "AdaptiveDomainDynamics",
    "AdaptiveDomainState",
    "AdaptiveEnlargementLaw",
    "BacksteppingCLF",
    "BacksteppingCLFEvaluation",
    "BlueROV2AgentController",
    "BlueROV2AgentEvaluation",
    "BlueROV2ControllerDesign",
    "BlueROV2ControlSpace",
    "CLFActuationFeasibility",
    "CLFQP",
    "CLFQPProblem",
    "CLFQPResult",
    "CLFQPSolverError",
    "ClassKFunction",
    "CommandFilter",
    "CommandFilterEvaluation",
    "DoubleIntegratorAgentController",
    "DoubleIntegratorAgentEvaluation",
    "FirstOrderCommandFilter",
    "LinearClassK",
    "PolyhedralControlSet",
    "SaturatingClassK",
    "SecondOrderCLFQPController",
    "SecondOrderCommandFilter",
    "SecondOrderControllerEvaluation",
    "VirtualVelocityEvaluation",
    "VirtualVelocityGains",
    "WrenchSpaceControllerDesign",
    "box_clf_actuation_feasibility",
    "build_bluerov2_controller_design",
    "build_wrench_space_controller",
    "equivalent_wrench_weight",
    "generalized_configuration_gradient",
    "minimum_effort_allocation_matrix",
    "virtual_velocity_command",
    "wrench_polytope_control_set",
]
