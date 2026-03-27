from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import onnxruntime as ort


def make_ort_session(
    model_path: Path,
    providers: Sequence[Any],
    graph_optimization: str = "all",
    execution_mode: str = "sequential",
    optimized_model_path: Optional[Path] = None,
) -> ort.InferenceSession:
    so = ort.SessionOptions()
    graph_opt_map = {
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    so.graph_optimization_level = graph_opt_map[graph_optimization]
    exec_mode_map = {
        "sequential": ort.ExecutionMode.ORT_SEQUENTIAL,
        "parallel": ort.ExecutionMode.ORT_PARALLEL,
    }
    so.execution_mode = exec_mode_map[execution_mode]
    if optimized_model_path is not None:
        so.optimized_model_filepath = str(optimized_model_path)

    normalized = []
    for item in providers:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            normalized.append((item[0], item[1]))
        else:
            normalized.append(item)

    return ort.InferenceSession(str(model_path), sess_options=so, providers=normalized)


def run_ort_once(session: ort.InferenceSession, input_name: str, sample: np.ndarray) -> List[np.ndarray]:
    return session.run(None, {input_name: sample})


def benchmark_ort_session(
    session: ort.InferenceSession,
    input_name: str,
    sample: np.ndarray,
    warmup: int,
    runs: int,
) -> Dict[str, float]:
    feed = {input_name: sample}
    for _ in range(warmup):
        session.run(None, feed)
    start = time.perf_counter()
    for _ in range(runs):
        session.run(None, feed)
    end = time.perf_counter()
    return {"latency_ms": (end - start) / runs * 1000.0}