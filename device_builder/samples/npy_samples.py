from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def load_sample_input(sample_path: Path, expected_shape: Sequence[int], expected_dtype: np.dtype) -> np.ndarray:
    arr = np.load(sample_path)
    arr = np.asarray(arr)
    if tuple(arr.shape) == tuple(expected_shape):
        pass
    elif len(arr.shape) + 1 == len(expected_shape) and tuple(arr.shape) == tuple(expected_shape[1:]):
        arr = np.expand_dims(arr, axis=0)
    else:
        raise ValueError(f"Sample input shape {tuple(arr.shape)} incompatible with expected {tuple(expected_shape)}")
    return np.ascontiguousarray(arr.astype(expected_dtype, copy=False))


def load_sample_output(sample_path: Path) -> np.ndarray:
    return np.ascontiguousarray(np.load(sample_path))