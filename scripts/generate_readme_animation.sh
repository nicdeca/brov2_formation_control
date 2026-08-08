#!/usr/bin/env bash
set -euo pipefail

mkdir -p docs/media

uv run python examples/09_bluerov2_readme_formation_demo.py \
    --duration 30 \
    --frame-stride 5 \
    --animation-format gif \
    --output docs/media/formation_animation.gif \
    --no-show

echo
echo "README animation written to:"
echo "  docs/media/formation_animation.gif"
