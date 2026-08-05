import numpy as np
import pytest

pytest.importorskip("qpsolvers")
pytest.importorskip("osqp")

from formation_control.control import (  # noqa: E402
    CLFQP,
    BacksteppingCLF,
    BlueROV2AgentController,
    FirstOrderCommandFilter,
    LinearClassK,
    PolyhedralControlSet,
    SecondOrderCLFQPController,
)
from formation_control.geometry import (  # noqa: E402
    PinholeCamera,
    rotation_matrix_from_quaternion,
)
from formation_control.models import BlueROV2Model  # noqa: E402
from formation_control.potentials import (  # noqa: E402
    EdgePotential,
    ImageCenteringPotential,
    RelativePositionPotential,
)


def quaternion_from_yaw(yaw):
    return np.array(
        [
            np.cos(0.5 * yaw),
            0.0,
            0.0,
            np.sin(0.5 * yaw),
        ]
    )


def make_state(
    position,
    *,
    yaw=0.0,
    velocity=None,
):
    if velocity is None:
        velocity = np.zeros(6)
    return np.concatenate(
        (
            np.asarray(position, dtype=float),
            quaternion_from_yaw(yaw),
            np.asarray(velocity, dtype=float),
        )
    )


def build_controller(
    *,
    input_matrix=None,
    control_limit=100.0,
):
    model = BlueROV2Model()
    if input_matrix is None:
        input_matrix = np.eye(6)

    dynamics_controller = SecondOrderCLFQPController(
        virtual_gain=np.diag([0.8, 0.8, 0.8, 0.5, 0.5, 0.5]),
        command_filter=FirstOrderCommandFilter(
            signal_dim=6,
            bandwidth=4.0,
        ),
        clf=BacksteppingCLF(
            inertia=model.mass_matrix,
            input_matrix=input_matrix,
        ),
        qp=CLFQP.isotropic(
            control_dim=input_matrix.shape[1],
            control_weight=1.0,
            slack_penalty=1e4,
            alpha=LinearClassK(gain=1.0),
            control_set=PolyhedralControlSet.box(
                -control_limit,
                control_limit,
                dimension=input_matrix.shape[1],
            ),
        ),
    )

    return model, BlueROV2AgentController(
        dynamics_controller=dynamics_controller,
        model=model,
    )


def test_generalized_gradient_uses_body_translation_and_body_orientation():
    _, controller = build_controller()
    state = make_state(
        np.array([0.2, -0.1, 0.3]),
        yaw=0.35,
    )
    target = np.array([2.0, 0.7, 0.5])

    edge_potential = EdgePotential(
        formation=RelativePositionPotential.isotropic(
            np.array([1.5, 0.2, 0.0]),
            gain=1.2,
        ),
        image_centering=ImageCenteringPotential(
            horizontal_gain=0.8,
            vertical_gain=0.9,
        ),
        camera=PinholeCamera.from_degrees(55.0, 45.0),
    )

    potential, zeta = controller.evaluate_edge_potential(
        follower_state=state,
        parent_position=target,
        edge_potential=edge_potential,
    )

    rotation = rotation_matrix_from_quaternion(state[3:7])
    expected = np.concatenate(
        (
            rotation.T @ potential.observer_position_gradient,
            potential.observer_orientation_gradient_body,
        )
    )

    np.testing.assert_allclose(zeta, expected)


def test_filter_is_initialized_at_current_virtual_command():
    _, controller = build_controller()
    state = make_state(np.zeros(3))
    parent = np.array([2.0, 0.2, -0.1])
    edge_potential = EdgePotential(
        formation=RelativePositionPotential.isotropic(
            np.array([1.5, 0.0, 0.0]),
            gain=1.0,
        )
    )

    _, zeta = controller.evaluate_edge_potential(
        follower_state=state,
        parent_position=parent,
        edge_potential=edge_potential,
    )
    expected = controller.dynamics_controller.desired_velocity(zeta)

    filter_state = controller.initialize_filter(
        follower_state=state,
        parent_position=parent,
        edge_potential=edge_potential,
    )

    np.testing.assert_allclose(filter_state, expected)


def test_parent_motion_enters_clf_rate_offset():
    _, controller = build_controller()
    state = make_state(
        np.array([0.1, -0.2, 0.0]),
        velocity=np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    parent = np.array([2.1, 0.4, 0.2])
    parent_velocity = np.array([0.3, -0.1, 0.05])
    edge_potential = EdgePotential(
        formation=RelativePositionPotential.isotropic(
            np.array([1.6, 0.0, 0.0]),
            gain=1.0,
        )
    )

    filter_state = controller.initialize_filter(
        follower_state=state,
        parent_position=parent,
        edge_potential=edge_potential,
    )

    stationary_parent = controller.evaluate(
        follower_state=state,
        parent_position=parent,
        edge_potential=edge_potential,
        filter_state=filter_state,
    )
    moving_parent = controller.evaluate(
        follower_state=state,
        parent_position=parent,
        edge_potential=edge_potential,
        filter_state=filter_state,
        parent_linear_velocity_inertial=parent_velocity,
    )

    expected_offset = float(moving_parent.potential.target_position_gradient @ parent_velocity)

    assert (
        moving_parent.controller.clf.drift - stationary_parent.controller.clf.drift
    ) == pytest.approx(expected_offset)


def test_model_drift_is_used_in_backstepping_clf():
    model, controller = build_controller()
    state = make_state(
        np.array([0.0, 0.0, -1.0]),
        yaw=0.2,
        velocity=np.array([0.4, -0.1, 0.2, 0.05, -0.04, 0.03]),
    )
    parent = np.array([2.0, 0.4, -0.8])
    edge_potential = EdgePotential(
        formation=RelativePositionPotential.isotropic(
            np.array([1.5, 0.0, 0.0]),
            gain=1.0,
        )
    )

    filter_state = controller.initialize_filter(
        follower_state=state,
        parent_position=parent,
        edge_potential=edge_potential,
    )
    evaluation = controller.evaluate(
        follower_state=state,
        parent_position=parent,
        edge_potential=edge_potential,
        filter_state=filter_state,
    )

    _, _, velocity = model.split_state(state)
    filter_evaluation = evaluation.controller.filter
    velocity_error = velocity - filter_evaluation.output

    expected_drift = float(
        evaluation.generalized_configuration_gradient @ velocity
        - velocity_error
        @ (model.drift_wrench(state) + model.mass_matrix @ filter_evaluation.output_derivative)
    )

    assert evaluation.controller.clf.drift == pytest.approx(expected_drift)


def test_input_matrix_maps_optimized_coordinates_to_body_wrench():
    allocation = np.zeros((6, 8))
    allocation[:, :6] = np.eye(6)
    allocation[0, 6] = 0.5
    allocation[1, 7] = -0.4

    _, controller = build_controller(
        input_matrix=allocation,
        control_limit=200.0,
    )
    state = make_state(np.zeros(3))
    parent = np.array([2.0, 0.4, -0.2])
    edge_potential = EdgePotential(
        formation=RelativePositionPotential.isotropic(
            np.array([1.5, 0.0, 0.0]),
            gain=1.0,
        )
    )
    filter_state = controller.initialize_filter(
        follower_state=state,
        parent_position=parent,
        edge_potential=edge_potential,
    )

    evaluation = controller.evaluate(
        follower_state=state,
        parent_position=parent,
        edge_potential=edge_potential,
        filter_state=filter_state,
    )

    np.testing.assert_allclose(
        evaluation.wrench_body,
        allocation @ evaluation.optimized_input,
    )
    assert evaluation.optimized_input.shape == (8,)
    assert evaluation.wrench_body.shape == (6,)


def test_adapter_rejects_inertia_mismatch():
    model = BlueROV2Model()
    dynamics_controller = SecondOrderCLFQPController(
        virtual_gain=np.eye(6),
        command_filter=FirstOrderCommandFilter(
            signal_dim=6,
            bandwidth=4.0,
        ),
        clf=BacksteppingCLF(2.0 * model.mass_matrix),
        qp=CLFQP.isotropic(
            control_dim=6,
            control_weight=1.0,
            slack_penalty=100.0,
            alpha=LinearClassK(gain=1.0),
        ),
    )

    with pytest.raises(ValueError):
        BlueROV2AgentController(
            dynamics_controller=dynamics_controller,
            model=model,
        )
