"""README demo: seven BlueROV2 vehicles acquiring a formation.

The leader remains stationary while six followers start from a visibly
displaced, asymmetric configuration and converge to the desired balanced-tree
formation under the same sensing-constrained CLF-QP controller used elsewhere
in the project.

This example is intentionally short and visually compact.  It is meant to
communicate the central formation-control behavior in the repository README,
not to replace the longer validation experiments.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from formation_control.visualization import (
    BlueROV2HeavyVisualGeometry,
    animate_formation_3d,
    apply_visualization_style,
    save_animation,
)

from moving_formation_validation import (
    build_readme_demo_scenario,
    connection_quality_history,
    print_summary,
    simulate,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument(
        "--animation-format",
        choices=("gif", "mp4"),
        default="gif",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/media/formation_animation.gif"),
    )
    parser.add_argument(
        "--adaptive",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--paper-quality", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    if args.duration <= 0.0:
        parser.error("--duration must be positive")
    if args.dt <= 0.0:
        parser.error("--dt must be positive")
    if args.frame_stride <= 0:
        parser.error("--frame-stride must be positive")

    apply_visualization_style(paper_quality=args.paper_quality)

    scenario = build_readme_demo_scenario()
    result = simulate(
        motion="velocity_command",
        duration=args.duration,
        dt=args.dt,
        velocity_command=np.zeros(3),
        velocity_command_bandwidth=0.2,
        thruster_voltage=16,
        thrust_derating=1.0,
        control_space="thruster",
        adaptive=args.adaptive,
        use_parent_velocity_in_clf=False,
        virtual_linear_speed_limit=1.5,
        virtual_angular_speed_limit=2.0,
        relaxation_recovery_gain=0.8,
        relaxation_domain_margin_ratio=0.1,
        slack_linear_penalty=100.0,
        slack_quadratic_penalty=5e3,
        realism=None,
        random_seed=7,
        scenario=scenario,
    )
    print_summary(result, motion="velocity_command")

    followers = tuple(
        agent
        for agent in range(result.scenario.n_agents)
        if agent != result.scenario.graph.root
    )
    vehicle_geometry = BlueROV2HeavyVisualGeometry(
        thruster_configuration=result.allocation.configuration
    ).wireframe()

    animation = animate_formation_3d(
        result.trajectory,
        result.scenario.graph,
        camera=result.camera,
        camera_agents=followers,
        camera_depth=0.75,
        vehicle_geometry=vehicle_geometry,
        edge_quality=connection_quality_history(result),
        paper_quality=args.paper_quality,
        show_body_forward=False,
        trail_length=80,
        frame_stride=args.frame_stride,
        title="Distributed underwater formation control",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_animation(
        animation.animation,
        args.output,
        paper_quality=args.paper_quality,
        fps=30,
    )
    print(f"README animation written to {args.output}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
