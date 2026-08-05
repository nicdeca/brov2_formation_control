import numpy as np
import pytest

pytest.importorskip("qpsolvers")
pytest.importorskip("osqp")

from formation_control.control import (  # noqa: E402
    CLFQP,
    BacksteppingCLF,
    DoubleIntegratorAgentController,
    FirstOrderCommandFilter,
    LinearClassK,
    PolyhedralControlSet,
    SecondOrderCLFQPController,
)
from formation_control.models import DoubleIntegratorModel  # noqa: E402
from formation_control.potentials import EdgePotential, RelativePositionPotential  # noqa: E402
from formation_control.simulation import RK4Integrator  # noqa: E402


def build_agent_controller(acceleration_limit=8.0):
    dynamics_controller = SecondOrderCLFQPController(
        virtual_gain=1.5 * np.eye(3),
        command_filter=FirstOrderCommandFilter(
            signal_dim=3,
            bandwidth=5.0,
        ),
        clf=BacksteppingCLF(np.eye(3)),
        qp=CLFQP.isotropic(
            control_dim=3,
            control_weight=1.0,
            slack_penalty=1e4,
            alpha=LinearClassK(gain=2.0),
            control_set=PolyhedralControlSet.box(
                -acceleration_limit,
                acceleration_limit,
                dimension=3,
            ),
        ),
    )
    return DoubleIntegratorAgentController(
        dynamics_controller=dynamics_controller,
    )


def build_formation_potential():
    return EdgePotential(
        formation=RelativePositionPotential.isotropic(
            desired_relative_position=np.array([1.0, 0.0, 0.0]),
            gain=1.0,
        )
    )


def test_parent_velocity_enters_configuration_rate_offset():
    controller = build_agent_controller()
    edge_potential = build_formation_potential()
    state = np.array([-1.5, 0.2, 0.0, 0.1, 0.0, 0.0])
    parent_position = np.zeros(3)
    parent_velocity = np.array([0.4, -0.1, 0.2])

    filter_state = controller.initialize_filter(
        follower_state=state,
        parent_position=parent_position,
        edge_potential=edge_potential,
    )

    without_parent_velocity = controller.evaluate(
        follower_state=state,
        parent_position=parent_position,
        edge_potential=edge_potential,
        filter_state=filter_state,
    )
    with_parent_velocity = controller.evaluate(
        follower_state=state,
        parent_position=parent_position,
        parent_velocity=parent_velocity,
        edge_potential=edge_potential,
        filter_state=filter_state,
    )

    expected_offset = float(
        with_parent_velocity.potential.target_position_gradient @ parent_velocity
    )

    assert (
        with_parent_velocity.controller.clf.drift - without_parent_velocity.controller.clf.drift
    ) == pytest.approx(expected_offset)


def test_closed_loop_regulates_relative_position():
    controller = build_agent_controller()
    model = DoubleIntegratorModel(dimension=3)
    plant_integrator = RK4Integrator()
    filter_integrator = RK4Integrator()
    edge_potential = build_formation_potential()

    parent_position = np.zeros(3)
    state = np.array([-2.0, 0.4, -0.2, 0.0, 0.0, 0.0])
    filter_state = controller.initialize_filter(
        follower_state=state,
        parent_position=parent_position,
        edge_potential=edge_potential,
    )

    dt = 0.01
    for _ in range(800):
        evaluation = controller.evaluate(
            follower_state=state,
            parent_position=parent_position,
            edge_potential=edge_potential,
            filter_state=filter_state,
        )

        # Both dynamic states are advanced exactly once per control cycle.
        filter_state = filter_integrator.step(
            controller.dynamics_controller.command_filter,
            filter_state,
            evaluation.controller.desired_velocity,
            dt,
        )
        state = plant_integrator.step(
            model,
            state,
            evaluation.acceleration,
            dt,
        )

    position, velocity = model.split_state(state)
    relative_position = parent_position - position

    np.testing.assert_allclose(
        relative_position,
        np.array([1.0, 0.0, 0.0]),
        atol=2e-2,
    )
    np.testing.assert_allclose(velocity, np.zeros(3), atol=2e-2)


def test_camera_potential_is_rejected():
    from formation_control.geometry import PinholeCamera
    from formation_control.potentials import ImageCenteringPotential

    controller = build_agent_controller()
    edge_potential = EdgePotential(
        image_centering=ImageCenteringPotential(),
        camera=PinholeCamera.from_degrees(45.0, 45.0),
    )
    state = np.zeros(6)

    with pytest.raises(ValueError):
        controller.initialize_filter(
            follower_state=state,
            parent_position=np.array([2.0, 0.0, 0.0]),
            edge_potential=edge_potential,
        )
