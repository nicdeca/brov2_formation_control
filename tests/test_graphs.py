import numpy as np
import pytest

from formation_control.graphs import DirectedSensingGraph, SensingEdge


def test_rooted_star_uses_child_to_parent_edge_convention() -> None:
    graph = DirectedSensingGraph.rooted_star(4)

    assert graph.edges == (
        SensingEdge(1, 0),
        SensingEdge(2, 0),
        SensingEdge(3, 0),
    )
    assert graph.parent(0) is None
    assert graph.parent(2) == 0
    assert graph.children(0) == (1, 2, 3)
    assert graph.is_rooted_tree()


def test_rooted_chain_depth_and_order() -> None:
    graph = DirectedSensingGraph.rooted_chain(4)

    assert graph.edges == (
        SensingEdge(1, 0),
        SensingEdge(2, 1),
        SensingEdge(3, 2),
    )
    assert graph.depth(3) == 3
    assert graph.agents_by_depth() == (0, 1, 2, 3)


def test_adjacency_round_trip() -> None:
    adjacency = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ],
        dtype=int,
    )
    graph = DirectedSensingGraph.from_adjacency(adjacency)

    np.testing.assert_array_equal(graph.adjacency_matrix, adjacency)
    np.testing.assert_array_equal(graph.edge_array, np.array([[1, 0], [2, 1]]))


def test_cycle_is_not_a_rooted_tree() -> None:
    graph = DirectedSensingGraph.from_edges(3, [(1, 2), (2, 1)])

    assert not graph.is_rooted_tree()
    with pytest.raises(ValueError, match="rooted sensing tree"):
        graph.require_rooted_tree()


def test_random_tree_is_reproducible_and_valid() -> None:
    first = DirectedSensingGraph.random_rooted_tree(8, seed=7)
    second = DirectedSensingGraph.random_rooted_tree(8, seed=7)

    assert first == second
    assert first.is_rooted_tree()


def test_invalid_edges_are_rejected() -> None:
    with pytest.raises(ValueError, match="self-loops"):
        DirectedSensingGraph.from_edges(2, [(1, 1)])
    with pytest.raises(ValueError, match="duplicate"):
        DirectedSensingGraph.from_edges(2, [(1, 0), (1, 0)])
