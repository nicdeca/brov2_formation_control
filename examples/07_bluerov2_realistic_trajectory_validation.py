"""Realistic seven-BlueROV trajectory validation."""

import sys

from moving_formation_validation import main

if __name__ == "__main__":
    if "--realistic" not in sys.argv and "--no-realistic" not in sys.argv:
        sys.argv.append("--realistic")
    main("trajectory")
