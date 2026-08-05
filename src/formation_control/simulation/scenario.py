"""Model-independent multi-robot formation scenarios."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from formation_control.graphs import DirectedSensingGraph, SensingEdge

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class FormationReference:
    """Desired robot offsets expressed in a common inertial frame.

    ``offsets[i]`` is the desired position of robot ``i`` relative to the
    formation root. The root offset is normalized to zero at construction.

    The package uses the paper's relative-position convention

        p_ij = p_j - p_i,

    so the desired relative position associated with sensing edge ``i -> j``
    is ``offsets[j] - offsets[i]``.
    """

    offsets: FloatArray
    root: int = 0

    def __post_init__(self) -> None:
        offsets = np.asarray(self.offsets, dtype=float)
        if offsets.ndim != 2:
            raise ValueError("offsets must have shape (n_agents, dimension).")
        if offsets.shape[0] < 1 or offsets.shape[1] < 1:
            raise ValueError("offsets must contain at least one agent and one dimension.")
        if not np.all(np.isfinite(offsets)):
            raise ValueError("offsets must contain only finite values.")
        if not 0 <= self.root < offsets.shape[0]:
            raise ValueError("root must index an existing formation offset.")

        normalized = offsets - offsets[self.root]
        normalized.setflags(write=False)
        object.__setattr__(self, "offsets", normalized)

    @property
    def n_agents(self) -> int:
        return self.offsets.shape[0]

    @property
    def dimension(self) -> int:
        return self.offsets.shape[1]

    def relative_position(self, observer: int, target: int) -> FloatArray:
        """Return desired ``p_target - p_observer``."""
        self._validate_agent(observer)
        self._validate_agent(target)
        return (self.offsets[target] - self.offsets[observer]).copy()

    def edge_relative_positions(self, graph: DirectedSensingGraph) -> FloatArray:
        """Return desired relative positions in ``graph.edges`` order."""
        self._validate_graph(graph)
        if len(graph) == 0:
            return np.empty((0, self.dimension), dtype=float)
        return np.vstack(
            [self.relative_position(edge.observer, edge.target) for edge in graph.edges]
        )

    @classmethod
    def followers_on_circle(
        cls,
        n_agents: int,
        radius: float,
        *,
        dimension: int = 3,
        root: int = 0,
        phase: float = 0.0,
    ) -> FormationReference:
        """Place the root at the origin and followers on a planar circle."""
        if n_agents < 1:
            raise ValueError("n_agents must be positive.")
        if radius <= 0.0:
            raise ValueError("radius must be positive.")
        if dimension not in (2, 3):
            raise ValueError("dimension must be 2 or 3.")
        if not 0 <= root < n_agents:
            raise ValueError("root must index an existing agent.")

        offsets = np.zeros((n_agents, dimension), dtype=float)
        followers = [agent for agent in range(n_agents) if agent != root]
        for index, agent in enumerate(followers):
            angle = phase + 2.0 * np.pi * index / max(len(followers), 1)
            offsets[agent, 0] = radius * np.cos(angle)
            offsets[agent, 1] = radius * np.sin(angle)
        return cls(offsets=offsets, root=root)

    def _validate_agent(self, agent: int) -> None:
        if not 0 <= agent < self.n_agents:
            raise IndexError(f"agent index {agent} is outside [0, {self.n_agents}).")

    def _validate_graph(self, graph: DirectedSensingGraph) -> None:
        if graph.n_agents != self.n_agents:
            raise ValueError(
                "graph and formation reference must contain the same number of agents."
            )
        if graph.root != self.root:
            raise ValueError("graph and formation reference must use the same root.")


@dataclass(frozen=True)
class FormationScenario:
    """Initial geometry and desired formation for a multi-robot example.

    This object is intentionally model-independent: it stores positions,
    topology, and the desired formation, but not velocities, quaternions, or
    controller parameters. Examples can lift these positions into the state
    representation of any model.
    """

    graph: DirectedSensingGraph
    reference: FormationReference
    initial_positions: FloatArray

    def __post_init__(self) -> None:
        initial_positions = np.asarray(self.initial_positions, dtype=float)
        expected_shape = (self.graph.n_agents, self.reference.dimension)
        if initial_positions.shape != expected_shape:
            raise ValueError(
                "initial_positions must have shape "
                f"{expected_shape}, got {initial_positions.shape}."
            )
        if not np.all(np.isfinite(initial_positions)):
            raise ValueError("initial_positions must contain only finite values.")
        if self.reference.n_agents != self.graph.n_agents:
            raise ValueError(
                "graph and formation reference must contain the same number of agents."
            )
        if self.reference.root != self.graph.root:
            raise ValueError("graph and formation reference must use the same root.")

        stored = initial_positions.copy()
        stored.setflags(write=False)
        object.__setattr__(self, "initial_positions", stored)

    @property
    def n_agents(self) -> int:
        return self.graph.n_agents

    @property
    def dimension(self) -> int:
        return self.reference.dimension

    def relative_position(
        self,
        observer: int,
        target: int,
        *,
        positions: FloatArray | None = None,
    ) -> FloatArray:
        """Return current ``p_target - p_observer`` for a supplied configuration."""
        positions = (
            self.initial_positions if positions is None else self._validate_positions(positions)
        )
        self._validate_agent(observer)
        self._validate_agent(target)
        return (positions[target] - positions[observer]).copy()

    def desired_relative_position(self, observer: int, target: int) -> FloatArray:
        """Return desired ``p_target - p_observer``."""
        return self.reference.relative_position(observer, target)

    def edge_relative_positions(self, *, positions: FloatArray | None = None) -> FloatArray:
        """Return current relative positions in ``graph.edges`` order."""
        positions = (
            self.initial_positions if positions is None else self._validate_positions(positions)
        )
        if len(self.graph) == 0:
            return np.empty((0, self.dimension), dtype=float)
        return np.vstack(
            [positions[edge.target] - positions[edge.observer] for edge in self.graph.edges]
        )

    def desired_edge_relative_positions(self) -> FloatArray:
        """Return desired relative positions in ``graph.edges`` order."""
        return self.reference.edge_relative_positions(self.graph)

    def local_edges(self, observer: int) -> tuple[SensingEdge, ...]:
        """Return outgoing sensing edges of ``observer``."""
        self._validate_agent(observer)
        return tuple(edge for edge in self.graph.edges if edge.observer == observer)

    def local_relative_positions(
        self,
        observer: int,
        *,
        positions: FloatArray | None = None,
    ) -> FloatArray:
        """Return current relative positions observed by one robot."""
        positions = (
            self.initial_positions if positions is None else self._validate_positions(positions)
        )
        edges = self.local_edges(observer)
        if not edges:
            return np.empty((0, self.dimension), dtype=float)
        return np.vstack([positions[edge.target] - positions[observer] for edge in edges])

    def local_desired_relative_positions(self, observer: int) -> FloatArray:
        """Return desired relative positions corresponding to ``local_edges``."""
        edges = self.local_edges(observer)
        if not edges:
            return np.empty((0, self.dimension), dtype=float)
        return np.vstack(
            [self.reference.relative_position(edge.observer, edge.target) for edge in edges]
        )

    def pairwise_distances(self, *, positions: FloatArray | None = None) -> FloatArray:
        """Return the symmetric matrix of pairwise Euclidean distances."""
        positions = (
            self.initial_positions if positions is None else self._validate_positions(positions)
        )
        displacements = positions[:, None, :] - positions[None, :, :]
        return np.linalg.norm(displacements, axis=2)

    @classmethod
    def around_reference(
        cls,
        graph: DirectedSensingGraph,
        reference: FormationReference,
        *,
        root_position: FloatArray | None = None,
        noise_std: float = 0.0,
        seed: int | None = None,
    ) -> FormationScenario:
        """Build initial positions around a translated desired formation.

        Gaussian perturbations are useful for examples but no feasibility
        checks are performed here; safety belongs to the constraint layer.
        """
        if noise_std < 0.0:
            raise ValueError("noise_std must be nonnegative.")
        if graph.n_agents != reference.n_agents or graph.root != reference.root:
            raise ValueError("graph and reference are incompatible.")

        if root_position is None:
            root_position = np.zeros(reference.dimension, dtype=float)
        root_position = np.asarray(root_position, dtype=float)
        if root_position.shape != (reference.dimension,):
            raise ValueError(
                "root_position must have shape "
                f"({reference.dimension},), got {root_position.shape}."
            )

        positions = root_position + reference.offsets
        if noise_std > 0.0:
            rng = np.random.default_rng(seed)
            positions = positions + rng.normal(scale=noise_std, size=positions.shape)

        return cls(graph=graph, reference=reference, initial_positions=positions)

    def _validate_agent(self, agent: int) -> None:
        if not 0 <= agent < self.n_agents:
            raise IndexError(f"agent index {agent} is outside [0, {self.n_agents}).")

    def _validate_positions(self, positions: FloatArray) -> FloatArray:
        positions = np.asarray(positions, dtype=float)
        expected_shape = (self.n_agents, self.dimension)
        if positions.shape != expected_shape:
            raise ValueError(f"positions must have shape {expected_shape}, got {positions.shape}.")
        if not np.all(np.isfinite(positions)):
            raise ValueError("positions must contain only finite values.")
        return positions
