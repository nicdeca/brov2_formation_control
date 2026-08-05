"""Saving utilities for publication figures and animations."""

from __future__ import annotations

from pathlib import Path

from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.figure import Figure


def save_figure(
    figure: Figure,
    path: str | Path,
    *,
    paper_quality: bool = False,
    transparent: bool = False,
) -> Path:
    """Save a Matplotlib figure and create parent directories as needed.

    If ``path`` has no suffix, normal-quality figures default to PNG and
    paper-quality figures default to vector PDF.
    """
    output = Path(path)
    if not output.suffix:
        output = output.with_suffix(".pdf" if paper_quality else ".png")
    output.parent.mkdir(parents=True, exist_ok=True)

    suffix = output.suffix.lower()
    raster = suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    dpi = 600 if paper_quality and raster else 300 if raster else None

    figure.savefig(
        output,
        dpi=dpi,
        bbox_inches="tight",
        transparent=transparent,
    )
    return output


def save_animation(
    animation: FuncAnimation,
    path: str | Path,
    *,
    paper_quality: bool = False,
    fps: int = 30,
) -> Path:
    """Save an animation as MP4 or GIF.

    Paper quality increases raster resolution and MP4 bitrate.  MP4 export
    requires ``ffmpeg``; GIF export uses Pillow.
    """
    if fps <= 0:
        raise ValueError("fps must be positive.")

    output = Path(path)
    if not output.suffix:
        output = output.with_suffix(".mp4")
    output.parent.mkdir(parents=True, exist_ok=True)

    suffix = output.suffix.lower()
    dpi = 200 if paper_quality else 120

    if suffix == ".mp4":
        writer = FFMpegWriter(
            fps=fps,
            bitrate=6000 if paper_quality else 3000,
        )
    elif suffix == ".gif":
        writer = PillowWriter(fps=fps)
    else:
        raise ValueError("animation output must use .mp4 or .gif extension.")

    try:
        animation.save(
            output,
            writer=writer,
            dpi=dpi,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            "animation writer is unavailable; install ffmpeg for MP4 or use GIF output."
        ) from error

    return output
