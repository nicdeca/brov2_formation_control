#!/usr/bin/env bash
#
# brov2_tmux_session.sh
#
# Recreates a 6-pane tmux layout (2 columns x 3 rows) for the
# brov2_formation_control workflow, with each pane's working
# directory and command pre-loaded.
#
# By default the commands are TYPED into each pane but NOT executed
# (no Enter is sent), so you can glance over everything before
# running it — same as the staged commands in the screenshot.
# Set AUTO_RUN=1 to have every pane execute immediately instead.
#
# Usage:
#   ./brov2_tmux_session.sh            # stage commands, don't run
#   AUTO_RUN=1 ./brov2_tmux_session.sh # stage and run immediately
#   tmux attach -t brov2               # (re)attach later

set -euo pipefail

SESSION="brov2"
ROS2_DIR="$HOME/brov2_formation_control/ros2"
REPO_DIR="$HOME/brov2_formation_control"
BROV2_DIR="$HOME"
QGC_DIR="$HOME/Desktop/BROV2/"
AUTO_RUN="${AUTO_RUN:-0}"

# Kill any stale session with the same name so this script is re-runnable.
tmux kill-session -t "$SESSION" 2>/dev/null || true

# send_cmd <pane_id> <command string>
# Types the command into the pane; only presses Enter if AUTO_RUN=1.
send_cmd() {
  local pane="$1"
  local cmd="$2"
  if [ "$AUTO_RUN" = "1" ]; then
    tmux send-keys -t "$pane" "$cmd" C-m
  else
    tmux send-keys -t "$pane" "$cmd"
  fi
}

# --- Build the 2x3 grid -----------------------------------------------
# Layout:
#   [ TL ] [ TR ]
#   [ ML ] [ MR ]
#   [ BL ] [ BR ]

tmux new-session -d -s "$SESSION" -n formation -c "$ROS2_DIR"
PANE_TL=$(tmux display-message -p -t "$SESSION:formation" '#{pane_id}')

PANE_TR=$(tmux split-window -h -t "$PANE_TL" -c "$HOME" -P -F '#{pane_id}')
PANE_ML=$(tmux split-window -v -t "$PANE_TL" -c "$ROS2_DIR" -P -F '#{pane_id}')
PANE_MR=$(tmux split-window -v -t "$PANE_TR" -c "$ROS2_DIR" -P -F '#{pane_id}')
PANE_BL=$(tmux split-window -v -t "$PANE_ML" -c "$ROS2_DIR" -P -F '#{pane_id}')
PANE_BR=$(tmux split-window -v -t "$PANE_MR" -c "$ROS2_DIR" -P -F '#{pane_id}')

tmux select-layout -t "$SESSION:formation" tiled

# --- Top-left: micro-XRCE-DDS agent ------------------------------------
send_cmd "$PANE_TL" "micro-xrce-dds-agent udp4 -p 8888"

# --- Top-right: QGroundControl -----------------------------------------
send_cmd "$PANE_TR" "cd $QGC_DIR && ls && ./QGroundControl-x86_64.AppImage"

# --- Mid-left: launch the 2-robot sim -----------------------------------
send_cmd "$PANE_ML" "source install/setup.bash && ros2 launch formation_control_ros multi_bluerov2_sim.launch.py robot_count:=2 robot_1_name:=splash robot_2_name:=glub px4_dir:=$BROV2_DIR/PX4-Autopilot"

# --- Mid-right: launch the 2-robot experiment ---------------------------
send_cmd "$PANE_MR" "source install/setup.bash && ros2 launch formation_control_ros two_robot_experiment.launch.py dry_run:=false leader_refernce_mode:=velocity workspace_barrier_enabled:=false workspace_adaptive:=false leader:=splash follower:=glub"

# --- Bottom-left: run the experiment script -----------------------------
send_cmd "$PANE_BL" "cd .. && python3 scripts/run_two_robot_experiment.py"

# --- Bottom-right: record the experiment ---------------------------------
send_cmd "$PANE_BR" "source install/setup.bash && cd .. && scripts/record_formation_experiment.sh --name two_robot_experiment --robots splash,glub --edge glub:splash"

tmux select-pane -t "$PANE_TL"

echo "Session '$SESSION' ready."
if [ "$AUTO_RUN" != "1" ]; then
  echo "Commands are staged but NOT executed. Attach and press Enter in each pane you want to run:"
fi
echo "  tmux attach -t $SESSION"

exec tmux attach -t "$SESSION"
