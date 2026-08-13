#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Record one formation-control experiment to a self-contained run directory.

Usage:
  scripts/record_formation_experiment.sh \
    --name NAME \
    --robots robot0,robot1[,robot2,...] \
    --edge follower:parent [--edge follower:parent ...] \
    [--output-root outputs/experiments]

Current two-robot example:
  scripts/record_formation_experiment.sh \
    --name two_robot_1p8 \
    --robots itrl_rov_1,itrl_rov_3 \
    --edge itrl_rov_3:itrl_rov_1

Ctrl-C stops rosbag cleanly.
EOF
}

NAME=""
ROBOTS=""
OUTPUT_ROOT="outputs/experiments"
EDGES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      NAME="$2"; shift 2 ;;
    --robots)
      ROBOTS="$2"; shift 2 ;;
    --edge)
      EDGES+=("$2"); shift 2 ;;
    --output-root)
      OUTPUT_ROOT="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

if [[ -z "$NAME" || -z "$ROBOTS" ]]; then
  usage >&2
  exit 2
fi

IFS=',' read -r -a ROBOT_ARRAY <<< "$ROBOTS"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_ROOT}/${STAMP}_${NAME}"
mkdir -p "$RUN_DIR"

MANIFEST="$RUN_DIR/run_manifest.yaml"
{
  echo "schema_version: 1"
  echo "name: \"$NAME\""
  echo "created_local: \"$(date --iso-8601=seconds)\""
  echo "robots:"
  for robot in "${ROBOT_ARRAY[@]}"; do
    echo "  - \"$robot\""
  done
  echo "edges:"
  for edge in "${EDGES[@]}"; do
    follower="${edge%%:*}"
    parent="${edge#*:}"
    echo "  - observer: \"$follower\""
    echo "    target: \"$parent\""
  done
} > "$MANIFEST"

# Global experiment lifecycle information is recorded once.
TOPICS=(
  /clock
  /formation_control/experiment_phase
)

for robot in "${ROBOT_ARRAY[@]}"; do
  prefix="/${robot}"
  TOPICS+=(
    "${prefix}/fmu/out/vehicle_odometry"
    "${prefix}/fmu/out/vehicle_control_mode"
    "${prefix}/fmu/in/vehicle_thrust_setpoint"
    "${prefix}/fmu/in/vehicle_torque_setpoint"
    "${prefix}/formation_control/diagnostic_snapshot"

    # Existing human-readable sensing/controller diagnostics.  The offline
    # exporter continues to use the atomic diagnostic snapshot as its
    # authoritative synchronized source.
    "${prefix}/formation_control/fallback"
    "${prefix}/formation_control/slack"
    "${prefix}/formation_control/required_slack"
    "${prefix}/formation_control/actuation_margin"
    "${prefix}/formation_control/thruster_utilization"
    "${prefix}/formation_control/minimum_physical_margin"
    "${prefix}/formation_control/conservative_constraint_values"
    "${prefix}/formation_control/domain_relaxation"

    # Stage-B workspace diagnostics.  These are intentionally separate from
    # the four-channel sensing-domain diagnostics above.
    "${prefix}/formation_control/workspace_barrier_enabled"
    "${prefix}/formation_control/workspace_adaptive"
    "${prefix}/formation_control/workspace_barrier_value"
    "${prefix}/formation_control/workspace_relaxation"
    "${prefix}/formation_control/workspace_conservative_constraint_values"
    "${prefix}/formation_control/workspace_physical_constraint_values"
    "${prefix}/formation_control/workspace_minimum_physical_margin"
  )
done

cat <<EOF
Run directory: $RUN_DIR
Manifest:      $MANIFEST
Robots:        ${ROBOT_ARRAY[*]}
Edges:         ${EDGES[*]:-(none)}
Recording ${#TOPICS[@]} explicit topics.
Press Ctrl-C to stop.
EOF

exec ros2 bag record -o "$RUN_DIR/bag" "${TOPICS[@]}"
