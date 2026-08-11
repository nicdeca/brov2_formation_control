"""Portable, ROS-independent representation of a recorded formation run."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FormationExperimentHistory:
    """Named NumPy histories plus JSON-serializable run metadata."""

    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {key: np.asarray(value) for key, value in self.arrays.items()}
        payload["__metadata_json__"] = np.asarray(
            json.dumps(self.metadata, sort_keys=True)
        )
        np.savez_compressed(path, **payload)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "FormationExperimentHistory":
        path = Path(path)
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["__metadata_json__"]))
            arrays = {
                key: np.asarray(archive[key])
                for key in archive.files
                if key != "__metadata_json__"
            }
        return cls(arrays=arrays, metadata=metadata)
