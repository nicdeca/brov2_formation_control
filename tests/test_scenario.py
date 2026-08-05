import numpy as np
import pytest

from formation_control.graphs import DirectedSensingGraph
from formation_control.simulation import FormationReference, FormationScenario


def test_reference_uses_p_target_minus_p_observer_convention() -> None:
    reference = FormationReference(
        offsets=np.array(
            [
                [10.0, 5.0, 0.0],
                [12.0, 5.0, 0.0],
                [10.0, 8.0, 0.0],
            ]
        )
    )

    np.testing.assert_allclose(reference.offsets[0], np.zeros(3))
    np.testing.assert_allclose(reference.relative_position(1, 0), [-2.0, 0.0, 0.0])
    np.testing.assert_allclose(reference.relative_position(2, 0), [0.0, -3.0, 0.0])


def test_edge_relative_positions_follow_graph_edge_order() -> None:
    graph = DirectedSensingGraph.rooted_chain(3)
    reference = FormationReference(
        offsets=np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 2.0],
            ]
        )
    )

    desired = reference.edge_relative_positions(graph)
    np.testing.assert_allclose(desired, [[-1.0, 0.0], [0.0, -2.0]])


def test_scenario_local_relative_data() -> None:
    graph = DirectedSensingGraph.rooted_star(3)
    reference = FormationReference(
        offsets=np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ]
        )
    )
    positions = np.array(
        [
            [1.0, 1.0, 0.0],
            [4.0, 1.0, 0.0],
            [1.0, 4.0, 0.0],
        ]
    )
    scenario = FormationScenario(graph, reference, positions)

    np.testing.assert_allclose(scenario.local_relative_positions(1), [[-3.0, 0.0, 0.0]])
    np.testing.assert_allclose(scenario.local_desired_relative_positions(1), [[-2.0, 0.0, 0.0]])
    assert scenario.local_relative_positions(0).shape == (0, 3)


def test_pairwise_distances_are_symmetric() -> None:
    graph = DirectedSensingGraph.rooted_star(3)
    reference = FormationReference.followers_on_circle(3, 2.0)
    scenario = FormationScenario.around_reference(graph, reference)

    distances = scenario.pairwise_distances()
    np.testing.assert_allclose(distances, distances.T)
    np.testing.assert_allclose(np.diag(distances), 0.0)


def test_around_reference_is_reproducible() -> None:
    graph = DirectedSensingGraph.rooted_star(4)
    reference = FormationReference.followers_on_circle(4, radius=2.0)

    first = FormationScenario.around_reference(
        graph,
        reference,
        root_position=np.array([4.0, 5.0, 1.0]),
        noise_std=0.1,
        seed=3,
    )
    second = FormationScenario.around_reference(
        graph,
        reference,
        root_position=np.array([4.0, 5.0, 1.0]),
        noise_std=0.1,
        seed=3,
    )

    np.testing.assert_allclose(first.initial_positions, second.initial_positions)


def test_scenario_rejects_incompatible_graph_and_reference() -> None:
    graph = DirectedSensingGraph.rooted_star(3)
    reference = FormationReference.followers_on_circle(4, radius=1.0)

    with pytest.raises(ValueError, match="same number of agents"):
        FormationScenario(graph, reference, np.zeros((3, 3)))
