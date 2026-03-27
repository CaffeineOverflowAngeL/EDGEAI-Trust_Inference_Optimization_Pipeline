from __future__ import annotations

from typing import Dict
import numpy as np


def compare_single_output(
    reference: np.ndarray,
    candidate: np.ndarray,
    atol: float,
    rtol: float,
) -> Dict[str, float | bool]:
    reference = np.asarray(reference, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)

    if reference.shape != candidate.shape:
        if reference.size == candidate.size:
            reference = reference.reshape(candidate.shape)
        else:
            raise ValueError(
                f"Output shape mismatch: reference shape {reference.shape}, "
                f"candidate shape {candidate.shape}, "
                f"reference size {reference.size}, candidate size {candidate.size}"
            )

    abs_err = np.abs(reference - candidate)
    rel_base = np.maximum(np.abs(reference), 1e-8)
    rel_err = abs_err / rel_base

    return {
        "matches": bool(np.allclose(reference, candidate, atol=atol, rtol=rtol)),
        "max_abs_error": float(np.max(abs_err)) if reference.size else 0.0,
        "mean_abs_error": float(np.mean(abs_err)) if reference.size else 0.0,
        "max_rel_error": float(np.max(rel_err)) if reference.size else 0.0,
        "atol": float(atol),
        "rtol": float(rtol),
    }