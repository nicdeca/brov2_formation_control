"""Directed sensing-graph primitives for multi-robot formations.

The convention used throughout the package is

    (i, j) in E  <=>  robot i observes / uses information from robot j.

Accordingly, for the rooted sensing trees considered in the paper, every
follower has exactly one outgoing edge to its parent while the leader/root
has no outgoing edge.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

IntArray = NDArray[np.int64]


@dataclass(frozen=True, order=True)
class SensingEdge:
    """Directed sensing edge ``observer -> target``."""

    observer: int
    target: int


@dataclass(frozen=True)
class DirectedSensingGraph:
    """Immutable directed sensing graph.

    The graph itself is general; use :meth:`require_rooted_tree` when an
    algorithm requires the paper's rooted-tree sensing topology.
    """

    n_agents: int
    edges: tuple[SensingEdge, ...]
    root: int = 0

    def __post_init__(self) -> None:
        if self.n_agents < 1:
            raise ValueError("n_agents must be positive.")
        if not 0 <= self.root < self.n_agents:
            raise ValueError("root must index an existing agent.")

        seen: set[SensingEdge] = set()
        normalized: list[SensingEdge] = []
        for edge in self.edges:
            if not isinstance(edge, SensingEdge):
                raise TypeError("edges must contain SensingEdge objects.")
            if not 0 <= edge.observer < self.n_agents:
                raise ValueError(f"invalid observer index {edge.observer}.")
            if not 0 <= edge.target < self.n_agents:
                raise ValueError(f"invalid target index {edge.target}.")
            if edge.observer == edge.target:
                raise ValueError("self-loops are not allowed in sensing graphs.")
            if edge in seen:
                raise ValueError(f"duplicate sensing edge {edge}.")
            seen.add(edge)
            normalized.append(edge)

        object.__setattr__(self, "edges", tuple(sorted(normalized)))

    @classmethod
    def from_edges(
        cls,
        n_agents: int,
        edges: Iterable[tuple[int, int] | SensingEdge],
        *,
        root: int = 0,
    ) -> DirectedSensingGraph:
        """Construct a graph from ``(observer, target)`` edge pairs."""
        sensing_edges = tuple(
            edge if isinstance(edge, SensingEdge) else SensingEdge(*edge) for edge in edges
        )
        return cls(n_agents=n_agents, edges=sensing_edges, root=root)

    @classmethod
    def from_adjacency(
        cls,
        adjacency: NDArray[np.integer],
        *,
        root: int = 0,
    ) -> DirectedSensingGraph:
        """Construct from ``A`` with ``A[i, j] != 0`` meaning ``i -> j``."""
        adjacency = np.asarray(adjacency)
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("adjacency must be a square matrix.")

        rows, cols = np.nonzero(adjacency)
        return cls.from_edges(
            adjacency.shape[0],
            zip(rows.tolist(), cols.tolist(), strict=True),
            root=root,
        )

    @classmethod
    def rooted_star(cls, n_agents: int, *, root: int = 0) -> DirectedSensingGraph:
        """Return a rooted star in which every follower observes ``root``."""
        return cls.from_edges(
            n_agents,
            ((agent, root) for agent in range(n_agents) if agent != root),
            root=root,
        ).require_rooted_tree()

    @classmethod
    def rooted_chain(cls, n_agents: int, *, root: int = 0) -> DirectedSensingGraph:
        """Return a rooted chain whose edges point toward ``root``."""
        order = [root, *(agent for agent in range(n_agents) if agent != root)]
        edges = ((order[k], order[k - 1]) for k in range(1, len(order)))
        return cls.from_edges(n_agents, edges, root=root).require_rooted_tree()

    @classmethod
    def random_rooted_tree(
        cls,
        n_agents: int,
        *,
        root: int = 0,
        seed: int | None = None,
    ) -> DirectedSensingGraph:
        """Generate a reproducible random rooted sensing tree.

        Agents are added one at a time and choose a parent among agents that
        are already connected to the root. This guarantees a tree without
        requiring graph-repair logic afterward.
        """
        if n_agents < 1:
            raise ValueError("n_agents must be positive.")
        if not 0 <= root < n_agents:
            raise ValueError("root must index an existing agent.")

        rng = np.random.default_rng(seed)
        order = [root, *(agent for agent in range(n_agents) if agent != root)]
        edges: list[tuple[int, int]] = []
        connected = [root]
        for agent in order[1:]:
            parent = int(rng.choice(connected))
            edges.append((agent, parent))
            connected.append(agent)

        return cls.from_edges(n_agents, edges, root=root).require_rooted_tree()

    def __iter__(self) -> Iterator[SensingEdge]:
        return iter(self.edges)

    def __len__(self) -> int:
        return len(self.edges)

    @property
    def adjacency_matrix(self) -> IntArray:
        """Return the binary matrix ``A`` with ``A[i,j] = 1`` for ``i -> j``."""
        adjacency = np.zeros((self.n_agents, self.n_agents), dtype=np.int64)
        for edge in self.edges:
            adjacency[edge.observer, edge.target] = 1
        return adjacency

    @property
    def edge_array(self) -> IntArray:
        """Return edges as an ``(n_edges, 2)`` integer array."""
        if not self.edges:
            return np.empty((0, 2), dtype=np.int64)
        return np.asarray([(edge.observer, edge.target) for edge in self.edges], dtype=np.int64)

    def targets(self, observer: int) -> tuple[int, ...]:
        """Return robots directly observed by ``observer``."""
        self._validate_agent(observer)
        return tuple(edge.target for edge in self.edges if edge.observer == observer)

    def observers(self, target: int) -> tuple[int, ...]:
        """Return robots that directly observe ``target``."""
        self._validate_agent(target)
        return tuple(edge.observer for edge in self.edges if edge.target == target)

    def parent(self, agent: int) -> int | None:
        """Return the unique parent of ``agent`` in a rooted sensing tree.

        The root has no parent. A :class:`ValueError` is raised if the graph
        does not give the queried non-root agent exactly one target.
        """
        self._validate_agent(agent)
        targets = self.targets(agent)
        if agent == self.root:
            if targets:
                raise ValueError("the root has outgoing sensing edges and is not a tree root.")
            return None
        if len(targets) != 1:
            raise ValueError(f"agent {agent} does not have exactly one parent.")
        return targets[0]

    def children(self, agent: int) -> tuple[int, ...]:
        """Return followers whose parent is ``agent``."""
        return self.observers(agent)

    def is_rooted_tree(self) -> bool:
        """Check the paper's child-to-parent rooted-tree convention."""
        if len(self.edges) != self.n_agents - 1:
            return False
        if self.targets(self.root):
            return False
        if any(
            len(self.targets(agent)) != 1 for agent in range(self.n_agents) if agent != self.root
        ):
            return False

        for agent in range(self.n_agents):
            visited: set[int] = set()
            current = agent
            while current != self.root:
                if current in visited:
                    return False
                visited.add(current)
                targets = self.targets(current)
                if len(targets) != 1:
                    return False
                current = targets[0]
        return True

    def require_rooted_tree(self) -> DirectedSensingGraph:
        """Return ``self`` if it is a rooted sensing tree, otherwise raise."""
        if not self.is_rooted_tree():
            raise ValueError(
                "graph must be a rooted sensing tree: the root has no parent and "
                "every other agent has exactly one outgoing edge leading to the root."
            )
        return self

    def depth(self, agent: int) -> int:
        """Return the number of parent links from ``agent`` to the root."""
        self.require_rooted_tree()
        self._validate_agent(agent)
        depth = 0
        current = agent
        while current != self.root:
            parent = self.parent(current)
            assert parent is not None
            current = parent
            depth += 1
        return depth

    def agents_by_depth(self) -> tuple[int, ...]:
        """Return agents ordered from the root toward the leaves."""
        self.require_rooted_tree()
        return tuple(sorted(range(self.n_agents), key=lambda agent: (self.depth(agent), agent)))

    def _validate_agent(self, agent: int) -> None:
        if not 0 <= agent < self.n_agents:
            raise IndexError(f"agent index {agent} is outside [0, {self.n_agents}).")
