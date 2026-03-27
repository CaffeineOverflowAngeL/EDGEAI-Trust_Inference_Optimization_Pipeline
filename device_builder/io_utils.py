from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Sequence


def log(*args: Any) -> None:
    print(*args)
    sys.stdout.flush()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=False))


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024) if path.exists() else 0.0


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def load_json_shape(text: Optional[str], name: str) -> Optional[List[int]]:
    if text is None:
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON for {name}: {exc}") from exc
    if not isinstance(value, list) or not all(isinstance(x, int) for x in value):
        raise ValueError(f"{name} must be a JSON list of integers, got: {value!r}")
    return value


def validate_user_shape_against_model(
    user_shape: Optional[Sequence[int]],
    inferred_shape: Sequence[int],
    arg_name: str,
) -> None:
    if user_shape is None:
        return
    if len(user_shape) != len(inferred_shape):
        raise ValueError(
            f"{arg_name} rank mismatch: got {list(user_shape)}, model expects rank {len(inferred_shape)} with shape {list(inferred_shape)}"
        )
    mismatches = []
    for idx, (got, expected) in enumerate(zip(user_shape, inferred_shape)):
        if expected > 0 and got != expected:
            mismatches.append(f"dim {idx}: got {got}, model expects {expected}")
    if mismatches:
        raise ValueError(
            f"{arg_name} is incompatible with the ONNX model input shape. "
            f"Provided: {list(user_shape)} | Model: {list(inferred_shape)} | "
            + "; ".join(mismatches)
        )