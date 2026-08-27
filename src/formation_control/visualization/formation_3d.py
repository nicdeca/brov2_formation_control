"""Three-dimensional plotting and animation for formation trajectories."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from formation_control.geometry import (
    PinholeCamera,
    rotation_matrix_from_quaternion,
)
from formation_control.graphs import DirectedSensingGraph
from formation_control.models.base import FloatArray
from formation_control.simulation import FormationTrajectory

from .edge_quality import (
    edge_quality_color,
    edge_quality_colormap,
    validate_edge_quality,
)
from .style import apply_visualization_style
from .vehicle_geometry import RigidBodyWireframe


@dataclass(frozen=True, slots=True)
class DesiredVehicleStyle3D:
    """Visual style for understated desired-vehicle wireframes."""

    color: str = "0.45"
    alpha: float = 0.24
    linestyle: str = "--"
    linewidth_scale: float = 0.85

    def __post_init__(self) -> None:
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError("alpha must lie in [0, 1].")
        if self.linewidth_scale <= 0.0:
            raise ValueError("linewidth_scale must be positive.")


def _xyz(positions: FloatArray) -> FloatArray:
    if positions.shape[-1] < 3:
        raise ValueError("3-D visualization requires at least three coordinates.")
    return positions[..., :3]


def _axis_limits(
    points: FloatArray,
    padding_fraction: float = 0.12,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    minimum = np.min(points, axis=(0, 1))
    maximum = np.max(points, axis=(0, 1))

    center = 0.5 * (minimum + maximum)
    span = max(float(np.max(maximum - minimum)), 1.0)
    half_span = 0.5 * span * (1.0 + 2.0 * padding_fraction)

    return tuple(
        (float(center[index] - half_span), float(center[index] + half_span)) for index in range(3)
    )


def _set_equal_axes(axes: Axes, points: FloatArray) -> None:
    x_limits, y_limits, z_limits = _axis_limits(points)
    axes.set_xlim(*x_limits)
    axes.set_ylim(*y_limits)
    axes.set_zlim(*z_limits)

    if hasattr(axes, "set_box_aspect"):
        axes.set_box_aspect((1.0, 1.0, 1.0))


def _validate_camera_agents(
    camera_agents: Iterable[int] | None,
    n_agents: int,
) -> tuple[int, ...]:
    if camera_agents is None:
        return tuple(range(n_agents))

    agents = tuple(int(agent) for agent in camera_agents)
    if len(set(agents)) != len(agents):
        raise ValueError("camera_agents must not contain duplicates.")
    if any(agent < 0 or agent >= n_agents for agent in agents):
        raise ValueError("camera_agents contains an invalid agent index.")
    return agents


def _camera_frustum_points(
    position: FloatArray,
    quaternion: FloatArray,
    camera: PinholeCamera,
    depth: float,
) -> tuple[FloatArray, FloatArray]:
    """Return inertial camera origin and four far-plane corners."""
    rotation_body_to_inertial = rotation_matrix_from_quaternion(quaternion)
    extrinsics = camera.extrinsics

    origin = position + (rotation_body_to_inertial @ extrinsics.position_camera_in_body)

    horizontal = depth * camera.horizontal_scale
    vertical = depth * camera.vertical_scale

    corners_camera = np.array(
        [
            [depth, -horizontal, -vertical],
            [depth, horizontal, -vertical],
            [depth, horizontal, vertical],
            [depth, -horizontal, vertical],
        ],
        dtype=float,
    )

    rotation_body_from_camera = extrinsics.rotation_camera_from_body.T
    corners_body = (rotation_body_from_camera @ corners_camera.T).T
    corners_inertial = origin + (rotation_body_to_inertial @ corners_body.T).T

    return origin, corners_inertial


def _set_line_3d(artist, start: FloatArray, end: FloatArray) -> None:
    artist.set_data([start[0], end[0]], [start[1], end[1]])
    artist.set_3d_properties([start[2], end[2]])


def _draw_static_camera_frustum(
    axes: Axes,
    position: FloatArray,
    quaternion: FloatArray,
    camera: PinholeCamera,
    depth: float,
) -> None:
    origin, corners = _camera_frustum_points(
        position,
        quaternion,
        camera,
        depth,
    )

    for corner in corners:
        axes.plot(
            [origin[0], corner[0]],
            [origin[1], corner[1]],
            [origin[2], corner[2]],
            linewidth=0.8,
            alpha=0.45,
        )

    for first, second in ((0, 1), (1, 2), (2, 3), (3, 0)):
        axes.plot(
            [corners[first, 0], corners[second, 0]],
            [corners[first, 1], corners[second, 1]],
            [corners[first, 2], corners[second, 2]],
            linewidth=0.8,
            alpha=0.45,
        )


def _add_wireframe(
    axes: Axes,
    geometry: RigidBodyWireframe,
    position: FloatArray,
    quaternion: FloatArray,
    *,
    color,
    alpha_scale: float = 1.0,
    linestyle: str | None = None,
    linewidth_scale: float = 1.0,
) -> list[Line3DCollection]:
    rotation = rotation_matrix_from_quaternion(quaternion)
    transformed = geometry.transform(position, rotation)

    artists = []
    for part, segments in zip(
        geometry.parts,
        transformed,
        strict=True,
    ):
        artist = Line3DCollection(
            segments,
            linewidths=part.linewidth * linewidth_scale,
            alpha=part.alpha * alpha_scale,
            linestyles=part.linestyle if linestyle is None else linestyle,
            colors=[color],
        )
        axes.add_collection3d(artist)
        artists.append(artist)

    return artists


def _update_wireframe(
    artists: list[Line3DCollection],
    geometry: RigidBodyWireframe,
    position: FloatArray,
    quaternion: FloatArray,
) -> None:
    rotation = rotation_matrix_from_quaternion(quaternion)
    transformed = geometry.transform(position, rotation)

    for artist, segments in zip(
        artists,
        transformed,
        strict=True,
    ):
        artist.set_segments(segments)


def _add_edge_quality_colorbar(
    figure: Figure,
    axes: Axes,
) -> None:
    mappable = ScalarMappable(
        norm=Normalize(vmin=0.0, vmax=1.0),
        cmap=edge_quality_colormap(),
    )
    mappable.set_array([])
    colorbar = figure.colorbar(
        mappable,
        ax=axes,
        pad=0.08,
        shrink=0.72,
    )
    colorbar.set_label("connection robustness")
    colorbar.set_ticks((0.0, 0.5, 1.0))
    colorbar.set_ticklabels(("disconnect", "reduced", "robust"))


def plot_formation_3d(
    trajectory: FormationTrajectory,
    graph: DirectedSensingGraph,
    *,
    desired_positions: FloatArray | None = None,
    camera: PinholeCamera | None = None,
    camera_agents: Iterable[int] | None = None,
    camera_depth: float = 0.8,
    vehicle_geometry: RigidBodyWireframe | None = None,
    edge_quality: FloatArray | None = None,
    show_edge_quality_colorbar: bool = True,
    paper_quality: bool = False,
    show_body_forward: bool = True,
    body_axis_length: float = 0.45,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot XYZ paths, final sensing graph, and optional orientation geometry."""
    apply_visualization_style(paper_quality=paper_quality)
    if graph.n_agents != trajectory.n_agents:
        raise ValueError("graph and trajectory must contain the same number of agents.")
    if camera_depth <= 0.0:
        raise ValueError("camera_depth must be positive.")
    if body_axis_length <= 0.0:
        raise ValueError("body_axis_length must be positive.")

    positions = _xyz(trajectory.positions)
    edges = tuple(graph.edges)
    edge_quality = validate_edge_quality(
        edge_quality,
        n_samples=trajectory.n_samples,
        n_edges=len(edges),
    )

    figure = plt.figure()
    axes = figure.add_subplot(111, projection="3d")

    agent_colors = []
    for agent in range(trajectory.n_agents):
        path_artist = axes.plot(
            positions[:, agent, 0],
            positions[:, agent, 1],
            positions[:, agent, 2],
            label=f"agent {agent}",
        )[0]
        agent_colors.append(path_artist.get_color())
        axes.scatter(
            positions[-1, agent, 0],
            positions[-1, agent, 1],
            positions[-1, agent, 2],
            s=45,
        )

    desired = None
    if desired_positions is not None:
        desired = np.asarray(desired_positions, dtype=float)
        if desired.shape[0] != trajectory.n_agents or desired.shape[1] < 3:
            raise ValueError("desired_positions must have shape (n_agents, dimension >= 3).")
        axes.scatter(
            desired[:, 0],
            desired[:, 1],
            desired[:, 2],
            marker="x",
            s=65,
            label="desired",
        )

    final_positions = positions[-1]
    for edge_index, edge in enumerate(edges):
        observer = final_positions[edge.observer]
        target = final_positions[edge.target]
        edge_color = (
            edge_quality_color(edge_quality[-1, edge_index]) if edge_quality is not None else None
        )
        axes.plot(
            [observer[0], target[0]],
            [observer[1], target[1]],
            [observer[2], target[2]],
            linestyle="--",
            linewidth=1.5 if edge_quality is not None else 1.0,
            color=edge_color,
        )

    if edge_quality is not None and show_edge_quality_colorbar:
        _add_edge_quality_colorbar(figure, axes)

    if trajectory.quaternions is not None:
        final_quaternions = trajectory.quaternions[-1]

        if vehicle_geometry is not None:
            for agent in range(trajectory.n_agents):
                _add_wireframe(
                    axes,
                    vehicle_geometry,
                    final_positions[agent],
                    final_quaternions[agent],
                    color=agent_colors[agent],
                )

        if show_body_forward:
            for agent in range(trajectory.n_agents):
                rotation = rotation_matrix_from_quaternion(final_quaternions[agent])
                endpoint = final_positions[agent] + body_axis_length * rotation[:, 0]
                _set_line_3d(
                    axes.plot([], [], [], linewidth=1.5)[0],
                    final_positions[agent],
                    endpoint,
                )

        if camera is not None:
            agents = _validate_camera_agents(
                camera_agents,
                trajectory.n_agents,
            )
            for agent in agents:
                _draw_static_camera_frustum(
                    axes,
                    final_positions[agent],
                    final_quaternions[agent],
                    camera,
                    camera_depth,
                )

    all_points = positions
    if desired is not None:
        all_points = np.concatenate(
            (positions, desired[None, :, :3]),
            axis=0,
        )

    _set_equal_axes(axes, all_points)
    axes.set_xlabel(r"$x$ [m]")
    axes.set_ylabel(r"$y$ [m]")
    axes.set_zlabel(r"$z$ [m]")
    axes.legend()

    if title is not None:
        axes.set_title(title)

    figure.tight_layout()
    return figure, axes


@dataclass(slots=True)
class FormationAnimation3D:
    """Matplotlib objects associated with a 3-D formation animation."""

    figure: Figure
    axes: Axes
    animation: FuncAnimation


def animate_formation_3d(
    trajectory: FormationTrajectory,
    graph: DirectedSensingGraph,
    *,
    desired_positions: FloatArray | None = None,
    desired_position_history: FloatArray | None = None,
    desired_vehicle_style: DesiredVehicleStyle3D | None = None,
    camera: PinholeCamera | None = None,
    camera_agents: Iterable[int] | None = None,
    camera_depth: float = 0.8,
    vehicle_geometry: RigidBodyWireframe | None = None,
    edge_quality: FloatArray | None = None,
    show_edge_quality_colorbar: bool = True,
    paper_quality: bool = False,
    show_body_forward: bool = True,
    body_axis_length: float = 0.45,
    title: str | None = None,
    interval_ms: int = 30,
    trail_length: int | None = None,
    frame_stride: int = 1,
    elevation: float = 25.0,
    azimuth: float = -60.0,
) -> FormationAnimation3D:
    """Animate XYZ motion, orientations, trails, and directed sensing edges."""
    apply_visualization_style(paper_quality=paper_quality)
    if graph.n_agents != trajectory.n_agents:
        raise ValueError("graph and trajectory must contain the same number of agents.")
    if interval_ms <= 0:
        raise ValueError("interval_ms must be positive.")
    if trail_length is not None and trail_length <= 0:
        raise ValueError("trail_length must be positive when supplied.")
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive.")
    if camera_depth <= 0.0:
        raise ValueError("camera_depth must be positive.")
    if body_axis_length <= 0.0:
        raise ValueError("body_axis_length must be positive.")
    if camera is not None and trajectory.quaternions is None:
        raise ValueError("camera visualization requires trajectory.quaternions.")
    if show_body_forward and trajectory.quaternions is None:
        show_body_forward = False

    positions = _xyz(trajectory.positions)
    edges = tuple(graph.edges)
    edge_quality = validate_edge_quality(
        edge_quality,
        n_samples=trajectory.n_samples,
        n_edges=len(edges),
    )

    figure = plt.figure()
    axes = figure.add_subplot(111, projection="3d")
    axes.view_init(elev=elevation, azim=azimuth)

    desired = None
    desired_history = None
    all_points = positions

    if desired_positions is not None and desired_position_history is not None:
        raise ValueError(
            "Supply either desired_positions or desired_position_history, not both."
        )

    if desired_position_history is not None:
        desired_history = np.asarray(desired_position_history, dtype=float)
        if (
            desired_history.ndim != 3
            or desired_history.shape[0] != trajectory.n_samples
            or desired_history.shape[1] != trajectory.n_agents
            or desired_history.shape[2] < 3
        ):
            raise ValueError(
                "desired_position_history must have shape "
                "(n_samples, n_agents, dimension >= 3)."
            )
        finite_desired = desired_history[..., :3][
            np.all(np.isfinite(desired_history[..., :3]), axis=-1)
        ]
        if finite_desired.size:
            all_points = np.concatenate(
                (
                    positions.reshape(-1, trajectory.n_agents, 3),
                    desired_history[..., :3],
                ),
                axis=0,
            )
    elif desired_positions is not None:
        desired = np.asarray(desired_positions, dtype=float)
        if desired.shape[0] != trajectory.n_agents or desired.shape[1] < 3:
            raise ValueError(
                "desired_positions must have shape "
                "(n_agents, dimension >= 3)."
            )
        all_points = np.concatenate(
            (positions, desired[None, :, :3]),
            axis=0,
        )
        axes.scatter(
            desired[:, 0],
            desired[:, 1],
            desired[:, 2],
            marker="x",
            s=65,
            label="desired",
        )

    # Ignore NaN reference samples when fixing the scene limits. This allows
    # reference vehicles to appear only once a reference becomes available.
    finite_points = all_points.reshape(-1, 3)
    finite_points = finite_points[np.all(np.isfinite(finite_points), axis=1)]
    if finite_points.size == 0:
        finite_points = positions.reshape(-1, 3)
    limit_points = finite_points[:, None, :]

    _set_equal_axes(axes, limit_points)
    axes.set_xlabel(r"$x$ [m]")
    axes.set_ylabel(r"$y$ [m]")
    axes.set_zlabel(r"$z$ [m]")

    markers = [
        axes.plot(
            [],
            [],
            [],
            marker="o",
            linestyle="None",
            label=f"agent {agent}",
        )[0]
        for agent in range(trajectory.n_agents)
    ]
    agent_colors = [marker.get_color() for marker in markers]
    trails = [
        axes.plot([], [], [], linewidth=1.2, alpha=0.7)[0] for _ in range(trajectory.n_agents)
    ]
    edge_artists = [
        axes.plot(
            [],
            [],
            [],
            linestyle="--",
            linewidth=1.8 if edge_quality is not None else 1.0,
            alpha=0.85 if edge_quality is not None else 0.7,
            color=(
                edge_quality_color(edge_quality[0, edge_index])
                if edge_quality is not None
                else None
            ),
        )[0]
        for edge_index, _ in enumerate(edges)
    ]

    if edge_quality is not None and show_edge_quality_colorbar:
        _add_edge_quality_colorbar(figure, axes)

    vehicle_artists: list[list[Line3DCollection]] = []
    if vehicle_geometry is not None:
        if trajectory.quaternions is None:
            raise ValueError("vehicle_geometry requires trajectory.quaternions.")
        initial_quaternions = trajectory.quaternions[0]
        for agent in range(trajectory.n_agents):
            vehicle_artists.append(
                _add_wireframe(
                    axes,
                    vehicle_geometry,
                    positions[0, agent],
                    initial_quaternions[agent],
                    color=agent_colors[agent],
                )
            )

    desired_vehicle_artists: list[list[Line3DCollection]] = []
    if desired_history is not None and vehicle_geometry is not None:
        if trajectory.quaternions is None:
            raise ValueError(
                "desired vehicle wireframes require trajectory.quaternions."
            )
        style = desired_vehicle_style or DesiredVehicleStyle3D()
        initial_quaternions = trajectory.quaternions[0]
        initial_desired = desired_history[0, :, :3]
        for agent in range(trajectory.n_agents):
            position = initial_desired[agent]
            if not np.all(np.isfinite(position)):
                position = np.zeros(3)
            artists = _add_wireframe(
                axes,
                vehicle_geometry,
                position,
                initial_quaternions[agent],
                color=style.color,
                alpha_scale=style.alpha,
                linestyle=style.linestyle,
                linewidth_scale=style.linewidth_scale,
            )
            visible = bool(np.all(np.isfinite(initial_desired[agent])))
            for artist in artists:
                artist.set_visible(visible)
            desired_vehicle_artists.append(artists)

    body_axes = []
    if show_body_forward:
        body_axes = [axes.plot([], [], [], linewidth=1.5)[0] for _ in range(trajectory.n_agents)]

    camera_agent_indices: tuple[int, ...] = ()
    camera_artists: dict[int, list] = {}
    if camera is not None:
        camera_agent_indices = _validate_camera_agents(
            camera_agents,
            trajectory.n_agents,
        )
        for agent in camera_agent_indices:
            # Four rays from the camera origin plus four far-plane edges.
            camera_artists[agent] = [
                axes.plot([], [], [], linewidth=0.8, alpha=0.45)[0] for _ in range(8)
            ]

    time_text = axes.text2D(
        0.02,
        0.98,
        "",
        transform=axes.transAxes,
        va="top",
    )

    axes.legend(loc="best")

    if title is not None:
        axes.set_title(title)

    frame_indices = list(range(0, trajectory.n_samples, frame_stride))
    if frame_indices[-1] != trajectory.n_samples - 1:
        frame_indices.append(trajectory.n_samples - 1)

    def update(frame: int):
        current = positions[frame]

        for agent, marker in enumerate(markers):
            marker.set_data(
                [current[agent, 0]],
                [current[agent, 1]],
            )
            marker.set_3d_properties([current[agent, 2]])

            start = 0 if trail_length is None else max(0, frame - trail_length + 1)
            trail = positions[start : frame + 1, agent]
            trails[agent].set_data(trail[:, 0], trail[:, 1])
            trails[agent].set_3d_properties(trail[:, 2])

        for edge_index, (artist, edge) in enumerate(zip(edge_artists, edges, strict=True)):
            _set_line_3d(
                artist,
                current[edge.observer],
                current[edge.target],
            )
            if edge_quality is not None:
                artist.set_color(edge_quality_color(edge_quality[frame, edge_index]))

        if trajectory.quaternions is not None:
            current_quaternions = trajectory.quaternions[frame]

            if vehicle_geometry is not None:
                for agent, artists in enumerate(vehicle_artists):
                    _update_wireframe(
                        artists,
                        vehicle_geometry,
                        current[agent],
                        current_quaternions[agent],
                    )

                if desired_history is not None:
                    desired_current = desired_history[frame, :, :3]
                    for agent, artists in enumerate(desired_vehicle_artists):
                        visible = bool(
                            np.all(np.isfinite(desired_current[agent]))
                        )
                        for artist in artists:
                            artist.set_visible(visible)
                        if visible:
                            # Only desired positions are specified by the
                            # formation/reference layer. Reusing the measured
                            # attitude avoids inventing a desired orientation.
                            _update_wireframe(
                                artists,
                                vehicle_geometry,
                                desired_current[agent],
                                current_quaternions[agent],
                            )

            for agent, artist in enumerate(body_axes):
                rotation = rotation_matrix_from_quaternion(current_quaternions[agent])
                endpoint = current[agent] + body_axis_length * rotation[:, 0]
                _set_line_3d(
                    artist,
                    current[agent],
                    endpoint,
                )

            if camera is not None:
                for agent in camera_agent_indices:
                    origin, corners = _camera_frustum_points(
                        current[agent],
                        current_quaternions[agent],
                        camera,
                        camera_depth,
                    )
                    artists = camera_artists[agent]

                    for index in range(4):
                        _set_line_3d(
                            artists[index],
                            origin,
                            corners[index],
                        )

                    for artist_index, (first, second) in enumerate(
                        ((0, 1), (1, 2), (2, 3), (3, 0)),
                        start=4,
                    ):
                        _set_line_3d(
                            artists[artist_index],
                            corners[first],
                            corners[second],
                        )

        time_text.set_text(f"t = {trajectory.times[frame]:.2f} s")

        return [
            *markers,
            *trails,
            *edge_artists,
            *body_axes,
            *(artist for artists in vehicle_artists for artist in artists),
            *(
                artist
                for artists in desired_vehicle_artists
                for artist in artists
            ),
            *(artist for artists in camera_artists.values() for artist in artists),
            time_text,
        ]

    animation = FuncAnimation(
        figure,
        update,
        frames=frame_indices,
        interval=interval_ms,
        blit=False,
        repeat=False,
    )

    return FormationAnimation3D(
        figure=figure,
        axes=axes,
        animation=animation,
    )
