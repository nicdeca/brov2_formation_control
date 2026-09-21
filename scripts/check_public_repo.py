#!/usr/bin/env python3
"""Fail on common machine-local artifacts before making the repository public."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve().relative_to(REPO_ROOT).as_posix()

FORBIDDEN_TRACKED_PREFIXES = (
    ".venv/",
    ".venv-ros/",
    ".venv_ros/",
    ".pytest_cache/",
    ".ruff_cache/",
)

# Deliberately target machine-specific paths, while allowing standard system
# paths such as /opt/ros/jazzy/setup.bash.
LOCAL_PATH_PATTERNS = (
    re.compile(r"/home/[^/\\s]+/"),
    re.compile(r"/Users/[^/\\s]+/"),
    re.compile(r"/mnt/[^/\\s]+/"),
    re.compile(r"~/discower_ws(?:/|\\b)"),
    re.compile(r"/path/to/"),
    re.compile(r"[A-Za-z]:\\\\Users\\\\[^\\\\\\s]+\\\\"),
)

TEXT_SUFFIXES = {
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".txt",
    ".cfg",
    ".ini",
    ".xml",
    ".sdf",
}


def tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def main() -> int:
    tracked = tracked_files()
    problems: list[str] = []

    for name in tracked:
        if name.startswith(FORBIDDEN_TRACKED_PREFIXES):
            problems.append(f"tracked local/generated file: {name}")

    for name in tracked:
        if name == SELF:
            continue
        path = REPO_ROOT / name
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            for pattern in LOCAL_PATH_PATTERNS:
                if pattern.search(line):
                    problems.append(
                        f"machine-local path: {name}:{line_number}: {line.strip()}"
                    )
                    break

    if problems:
        print("Public-repository check failed:\n")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("Public-repository check passed: no tracked local environments or machine-local paths found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
