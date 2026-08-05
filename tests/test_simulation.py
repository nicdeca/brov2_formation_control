import numpy as np
import pytest

from formation_control.models import (
    BlueROV2Model,
    DoubleIntegratorModel,
    SingleIntegratorModel,
)
from formation_control.simulation import (
    EulerIntegrator,
    RK4Integrator,
    SimulationResult,
    Simulator,
)


def test_euler_single_integrator_step() -> None:
    model = SingleIntegratorModel(dimension=2)
    integrator = EulerIntegrator()

    state = np.array([1.0, -2.0])
    control = np.array([0.5, 1.5])

    next_state = integrator.step(model, state, control, dt=0.2)

    np.testing.assert_allclose(next_state, np.array([1.1, -1.7]))


def test_rk4_double_integrator_matches_constant_acceleration_solution() -> None:
    model = DoubleIntegratorModel(dimension=2)
    integrator = RK4Integrator()

    state = np.array([1.0, -2.0, 0.5, 1.0])
    acceleration = np.array([2.0, -1.0])
    dt = 0.3

    next_state = integrator.step(model, state, acceleration, dt)

    position = state[:2] + dt * state[2:] + 0.5 * dt**2 * acceleration
    velocity = state[2:] + dt * acceleration
    expected = np.concatenate((position, velocity))
    np.testing.assert_allclose(next_state, expected)


def test_integrators_reject_nonpositive_step_size() -> None:
    model = SingleIntegratorModel()
    state = np.zeros(3)
    control = np.zeros(3)

    for integrator in (EulerIntegrator(), RK4Integrator()):
        with pytest.raises(ValueError, match="dt must be positive"):
            integrator.step(model, state, control, 0.0)


def test_simulator_advances_time_and_state() -> None:
    simulator = Simulator(
        model=SingleIntegratorModel(dimension=2),
        initial_state=np.zeros(2),
        dt=0.1,
        integrator=RK4Integrator(),
    )

    simulator.step(np.array([1.0, -2.0]))
    simulator.step(np.array([1.0, -2.0]))

    assert simulator.time == pytest.approx(0.2)
    np.testing.assert_allclose(simulator.state, np.array([0.2, -0.4]))


def test_simulator_reset_restores_initial_conditions() -> None:
    simulator = Simulator(
        model=SingleIntegratorModel(dimension=2),
        initial_state=np.array([1.0, 2.0]),
        dt=0.1,
        initial_time=3.0,
    )

    simulator.step(np.ones(2))
    reset_state = simulator.reset()

    assert simulator.time == pytest.approx(3.0)
    np.testing.assert_allclose(reset_state, np.array([1.0, 2.0]))


def test_simulator_projects_bluerov2_quaternion() -> None:
    model = BlueROV2Model()
    initial_state = np.zeros(13)
    initial_state[3:7] = np.array([2.0, 0.0, 0.0, 0.0])

    simulator = Simulator(model=model, initial_state=initial_state, dt=0.01)

    assert np.linalg.norm(simulator.state[3:7]) == pytest.approx(1.0)
    simulator.step(np.zeros(6))
    assert np.linalg.norm(simulator.state[3:7]) == pytest.approx(1.0)


def test_simulation_result_validates_history_alignment() -> None:
    result = SimulationResult(
        times=np.array([0.0, 0.1, 0.2]),
        states=np.zeros((3, 4)),
        controls=np.zeros((2, 2)),
    )

    assert result.num_steps == 2
    assert result.duration == pytest.approx(0.2)

    with pytest.raises(ValueError, match="one row per simulation interval"):
        SimulationResult(
            times=np.array([0.0, 0.1]),
            states=np.zeros((2, 4)),
            controls=np.zeros((2, 2)),
        )
