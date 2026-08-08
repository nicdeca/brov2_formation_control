#!/usr/bin/env bash
set -euo pipefail

mkdir -p docs/media

uv run python examples/07_bluerov2_realistic_trajectory_validation.py \
    --control-space thruster \
    --adaptive \
    --save-animation \
    --animation-format gif \
    --frame-stride 20 \
    --output-dir docs/media \
    --no-show

echo
echo "README animation written to:"
echo "  docs/media/formation_animation.gif"
