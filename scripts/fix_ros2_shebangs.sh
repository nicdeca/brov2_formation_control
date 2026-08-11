#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS_DIR="$(cd "$REPO_DIR/../.." && pwd)"
EXEC_DIR="$WS_DIR/install/formation_control_ros/lib/formation_control_ros"

for executable in \
    leader_controller \
    follower_controller \
    offboard_heartbeat_wrench
do
    sed -i '1c #!/usr/bin/env python3' \
        "$EXEC_DIR/$executable"
done
