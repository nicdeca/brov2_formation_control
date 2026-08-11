from formation_control_ros.core_runtime import (
    CoreControllerConfig,
    LeaderCoreRuntime,
)


def test_ros_core_defaults_to_thruster_space():
    config = CoreControllerConfig()

    assert config.control_space == "thruster"

    runtime = LeaderCoreRuntime(config)

    assert runtime.design.control_space == "thruster"
    assert runtime.design.decision_dimension == 8
    assert runtime.design.agent_controller.dynamics_controller.qp.uses_direct_box_solver
