#!/bin/bash
# Bring up the two-robot (glub + splash) manual control session in tmux.
#
#   left column            right column
#   ┌──────────────────┐   ┌──────────────────┐
#   │ manual  glub     │   │ micro-XRCE agent │
#   ├──────────────────┤   │                  │
#   │ manual  splash   │   ├──────────────────┤
#   ├──────────────────┤   │ QGroundControl   │
#   │ odom_ekf splash  │   │                  │
#   ├──────────────────┤   │                  │
#   │ odom_ekf glub    │   │                  │
#   └──────────────────┘   └──────────────────┘
#
# Usage:
#   ./tmux_two_robot_manual.sh              # build the session and run everything
#   ./tmux_two_robot_manual.sh --type-only  # type the commands into the panes, don't press Enter
#
# Run it from a shell that already has the ROS env active (pixi/robostack), the
# panes inherit that environment and only source the workspace install/ on top.
set -euo pipefail

SESSION="${SESSION:-brov2_manual}"

ROS_WS="${ROS_WS:-$HOME/ros_ws}"                            # holds install/
XRCE_DIR="${XRCE_DIR:-$HOME/brov2_formation_control}"
QGC_DIR="${QGC_DIR:-$HOME/Desktop/BROV2}"
QGC_APP="${QGC_APP:-./QGroundControl-x86_64.AppImage}"

GLUB_JOY="${GLUB_JOY:-0}"
SPLASH_JOY="${SPLASH_JOY:-1}"

TYPE_ONLY="${TYPE_ONLY:-0}"

for arg in "$@"; do
    case "$arg" in
        --type-only|-t) TYPE_ONLY=1 ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown argument: $arg" >&2; exit 1 ;;
    esac
done

command -v tmux >/dev/null || { echo "tmux is not installed" >&2; exit 1; }
[ -f "$ROS_WS/install/setup.bash" ] || { echo "no install/setup.bash in $ROS_WS" >&2; exit 1; }
command -v ros2 >/dev/null || echo "warning: 'ros2' not on PATH - activate the pixi/conda env first" >&2

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "session '$SESSION' already exists, attaching"
    if [ -n "${TMUX:-}" ]; then exec tmux switch-client -t "$SESSION"; else exec tmux attach -t "$SESSION"; fi
fi

# Always executed: puts the pane in the right workspace with the overlay sourced.
setup_ros_pane() {
    tmux send-keys -t "$1" "source $ROS_WS/install/setup.bash" C-m
}

# Respects --type-only, so the launch line can be left sitting at the prompt.
send_cmd() {
    local pane="$1"; shift
    tmux send-keys -t "$pane" "$*"
    [ "$TYPE_ONLY" = 1 ] || tmux send-keys -t "$pane" C-m
}

# --- panes -------------------------------------------------------------------
# Split the window in half first, then cut each column into equal rows.
manual_glub=$(tmux new-session -d -s "$SESSION" -n manual -c "$ROS_WS" -P -F '#{pane_id}')

xrce=$(tmux split-window -h -t "$manual_glub" -c "$XRCE_DIR" -P -F '#{pane_id}')
qgc=$(tmux split-window -v -t "$xrce" -c "$QGC_DIR" -P -F '#{pane_id}')

# -l is the size of the *new* pane, so 75/66/50 leaves four equal quarters.
manual_splash=$(tmux split-window -v -l 75% -t "$manual_glub" -c "$ROS_WS" -P -F '#{pane_id}')
ekf_splash=$(tmux split-window -v -l 66% -t "$manual_splash" -c "$ROS_WS" -P -F '#{pane_id}')
ekf_glub=$(tmux split-window -v -l 50% -t "$ekf_splash" -c "$ROS_WS" -P -F '#{pane_id}')

for pane in "$manual_glub" "$manual_splash" "$ekf_splash" "$ekf_glub"; do
    setup_ros_pane "$pane"
done

# --- commands ----------------------------------------------------------------
# Agent first so the PX4 clients have something to connect to.
send_cmd "$xrce" "micro-xrce-dds-agent udp4 -p 8888"
send_cmd "$qgc"  "$QGC_APP"

send_cmd "$ekf_glub"      "ros2 launch bluerov2_control brov_odom_ekf.launch.py ns:=glub"
send_cmd "$ekf_splash"    "ros2 launch bluerov2_control brov_odom_ekf.launch.py ns:=splash"
send_cmd "$manual_glub"   "ros2 launch bluerov2_control manual_control.launch.py ns:=glub joy_dev:=$GLUB_JOY"
send_cmd "$manual_splash" "ros2 launch bluerov2_control manual_control.launch.py ns:=splash joy_dev:=$SPLASH_JOY"

tmux select-pane -t "$manual_glub"

if [ -n "${TMUX:-}" ]; then
    tmux switch-client -t "$SESSION"
else
    tmux attach -t "$SESSION"
fi
