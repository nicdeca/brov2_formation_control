#!/usr/bin/env bash
set -euo pipefail

# Offline Gazebo state-log playback for final video rendering.
#
# Usage:
#   scripts/playback_gazebo_log.sh [LOG_DIR]
#
# If LOG_DIR is omitted, the newest Gazebo log containing state.tlog is used.
#
# IMPORTANT (Gazebo Harmonic):
# Passing --gui-config together with --playback prevents the GUI from entering
# its special playback mode and can open the Quick Start dialog instead.
# Therefore this script temporarily installs our project config as
# ~/.gz/sim/8/playback_gui.config, launches playback normally, then restores
# the user's previous playback config on exit.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROJECT_GUI_CONFIG="${PROJECT_ROOT}/ros2/formation_control_ros/config/gazebo_playback_video.config"

if [[ ! -f "${PROJECT_GUI_CONFIG}" ]]; then
  echo "ERROR: project playback GUI config not found:" >&2
  echo "  ${PROJECT_GUI_CONFIG}" >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# Select Gazebo log.
# ---------------------------------------------------------------------------
if [[ $# -ge 1 ]]; then
  LOG_DIR="$(realpath "$1")"
else
  LOG_DIR=""
  while IFS= read -r candidate; do
    if [[ -f "${candidate}/state.tlog" ]]; then
      LOG_DIR="${candidate}"
      break
    fi
  done < <(
    find "${HOME}/.gz/sim/log" \
      -mindepth 1 -maxdepth 1 -type d \
      -printf '%T@ %p\n' 2>/dev/null \
      | sort -nr \
      | cut -d' ' -f2-
  )
fi

if [[ -z "${LOG_DIR}" || ! -f "${LOG_DIR}/state.tlog" ]]; then
  echo "ERROR: no Gazebo state log was found." >&2
  echo "Pass a directory containing state.tlog, or check:" >&2
  echo "  ${HOME}/.gz/sim/log/" >&2
  exit 3
fi

echo "Gazebo playback log:"
echo "  ${LOG_DIR}"
echo
echo "state.tlog:"
ls -lh "${LOG_DIR}/state.tlog"
echo

# ---------------------------------------------------------------------------
# Resource paths needed by the recorded world / BlueROV models.
# ---------------------------------------------------------------------------
PX4_DIR="${PX4_AUTOPILOT_DIR:-${HOME}/Gits/KTH-PX4/PX4-Autopilot}"

RESOURCE_PATHS=()
if [[ -d "${PX4_DIR}/Tools/simulation/gz/models" ]]; then
  RESOURCE_PATHS+=("${PX4_DIR}/Tools/simulation/gz/models")
fi
if [[ -d "${PX4_DIR}/Tools/simulation/gz/worlds" ]]; then
  RESOURCE_PATHS+=("${PX4_DIR}/Tools/simulation/gz/worlds")
fi

# Preserve any resource paths already configured in the shell.
if [[ -n "${GZ_SIM_RESOURCE_PATH:-}" ]]; then
  RESOURCE_PATHS+=("${GZ_SIM_RESOURCE_PATH}")
fi

if (( ${#RESOURCE_PATHS[@]} > 0 )); then
  export GZ_SIM_RESOURCE_PATH="$(IFS=:; echo "${RESOURCE_PATHS[*]}")"
fi

# ---------------------------------------------------------------------------
# Gazebo Harmonic playback GUI configuration.
#
# Harmonic is Gazebo Sim major version 8. The GUI only recognizes playback
# when --playback is used without a custom --gui-config argument. Its normal
# playback layout is read from ~/.gz/sim/8/playback_gui.config.
# ---------------------------------------------------------------------------
GZ_SIM_MAJOR="${GZ_SIM_MAJOR:-8}"
PLAYBACK_CONFIG_DIR="${HOME}/.gz/sim/${GZ_SIM_MAJOR}"
PLAYBACK_CONFIG="${PLAYBACK_CONFIG_DIR}/playback_gui.config"
BACKUP_CONFIG=""

mkdir -p "${PLAYBACK_CONFIG_DIR}"

restore_playback_config() {
  local status=$?
  if [[ -n "${BACKUP_CONFIG}" && -f "${BACKUP_CONFIG}" ]]; then
    mv -f "${BACKUP_CONFIG}" "${PLAYBACK_CONFIG}"
  else
    rm -f "${PLAYBACK_CONFIG}"
  fi
  exit "${status}"
}
trap restore_playback_config EXIT INT TERM

if [[ -f "${PLAYBACK_CONFIG}" ]]; then
  BACKUP_CONFIG="$(mktemp "${PLAYBACK_CONFIG}.backup.XXXXXX")"
  cp -a "${PLAYBACK_CONFIG}" "${BACKUP_CONFIG}"
fi

cp -f "${PROJECT_GUI_CONFIG}" "${PLAYBACK_CONFIG}"

echo "Using playback GUI configuration:"
echo "  ${PLAYBACK_CONFIG}"
echo
echo "Playback should open directly (no Quick Start dialog)."
echo "It starts paused."
echo
echo "Recommended workflow:"
echo "  1. scrub to just before the mission start;"
echo "  2. verify / adjust the fixed camera;"
echo "  3. start the VideoRecorder (MP4);"
echo "  4. press Play;"
echo "  5. stop the VideoRecorder at the mission end."
echo

# Launch playback paused. Do not pass --gui-config here: Gazebo Harmonic
# detects playback mode and loads ~/.gz/sim/8/playback_gui.config automatically.
# Omitting -r keeps playback paused so the camera / VideoRecorder can be set up
# before advancing the log.
gz sim -v 4 --playback "${LOG_DIR}"
