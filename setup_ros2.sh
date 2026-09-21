#!/usr/bin/env bash

# Source this file from the repository root (or from any directory).
# The ROS-specific virtual environment is intentionally local-only and is
# recreated automatically after a fresh clone.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "$REPO_DIR/../.." && pwd)"
ROS_VENV="$REPO_DIR/.venv-ros"

if [ ! -x "$ROS_VENV/bin/python" ]; then
    if ! command -v uv >/dev/null 2>&1; then
        echo "ERROR: uv is required to create $ROS_VENV." >&2
        echo "Install uv first, then source setup_ros2.sh again." >&2
        return 1 2>/dev/null || exit 1
    fi

    echo "Creating local ROS Python environment at .venv-ros ..."
    uv venv "$ROS_VENV" --python python3
    uv pip install --python "$ROS_VENV/bin/python" -e "$REPO_DIR"
fi

source "$ROS_VENV/bin/activate"
source /opt/ros/jazzy/setup.bash

export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

if [ -f "$WS_DIR/install/setup.bash" ]; then
    source "$WS_DIR/install/setup.bash"
fi
