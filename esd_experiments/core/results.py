"""Result-file helpers for separating simulations from plotting."""

from __future__ import annotations

import json
from pathlib import Path
import hashlib
from typing import Any

import numpy as np


METADATA_KEY = "metadata_json"


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def stable_parameter_key(payload: dict[str, Any], length: int = 16) -> str:
    """Stable hash for an experiment's parameter payload."""

    if length <= 0:
        raise ValueError("length must be positive")
    normalized = _json_ready(payload)
    raw = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:length]


def save_result(path: str | Path, arrays: dict[str, Any], metadata: dict[str, Any]) -> None:
    """Save numeric result arrays plus JSON metadata to a compressed NPZ file."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: np.asarray(value) for name, value in arrays.items()}
    payload[METADATA_KEY] = np.asarray(json.dumps(_json_ready(metadata), sort_keys=True))
    np.savez_compressed(out, **payload)


def load_result(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load a result NPZ file written by :func:`save_result`."""

    with np.load(Path(path), allow_pickle=False) as data:
        if METADATA_KEY not in data.files:
            raise ValueError(f"{path} is missing required {METADATA_KEY!r}")
        arrays = {name: data[name] for name in data.files if name != METADATA_KEY}
        metadata = json.loads(str(data[METADATA_KEY]))
    return arrays, metadata


def records_to_arrays(records: list[dict[str, float]]) -> dict[str, np.ndarray]:
    """Convert a homogeneous list of scalar records into column arrays."""

    if not records:
        return {}
    keys = list(records[0].keys())
    return {
        key: np.asarray([record[key] for record in records], dtype=float)
        for key in keys
    }


def arrays_to_records(arrays: dict[str, np.ndarray]) -> list[dict[str, float]]:
    """Convert column arrays back into a list of scalar records."""

    if not arrays:
        return []
    keys = list(arrays.keys())
    size = len(np.asarray(arrays[keys[0]]))
    records: list[dict[str, float]] = []
    for idx in range(size):
        records.append({key: float(np.asarray(arrays[key])[idx]) for key in keys})
    return records
