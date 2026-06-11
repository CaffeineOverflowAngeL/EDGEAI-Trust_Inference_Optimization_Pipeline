from __future__ import annotations

import platform
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import onnx
import onnxruntime as ort

from .io_utils import file_size_mb, now_ts

try:
    import pycuda.driver as cuda  # type: ignore
    PYCUDA_AVAILABLE = True
except Exception:
    PYCUDA_AVAILABLE = False

try:
    import tensorrt as trt  # type: ignore
    TRT_AVAILABLE = True
except Exception:
    TRT_AVAILABLE = False


def collect_system_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "timestamp": now_ts(),
        "python": sys.version,
        "platform": platform.platform(),
        "onnxruntime": ort.__version__,
        "onnx": onnx.__version__,
        "tensorrt": None,
        "cuda_device": None,
    }

    if TRT_AVAILABLE:
        try:
            info["tensorrt"] = trt.__version__
        except Exception:
            info["tensorrt"] = "unknown"

    if PYCUDA_AVAILABLE:
        try:
            context = cuda.Context.get_current()
            if context is not None:
                dev = context.get_device()
                info["cuda_device"] = {
                    "name": dev.name(),
                    "compute_capability": dev.compute_capability(),
                    "total_memory_mb": int(dev.total_memory() / (1024 * 1024)),
                }
        except Exception:
            info["cuda_device"] = "unavailable"

    return info


def build_manifest(
    model_path: Path,
    preset: str,
    input_info: Dict[str, Any],
    onnx_optimization: Dict[str, Any],
    optimized_onnx_path: Path,
    ort_optimized_export_path: Optional[Path],
    trt_artifacts: Optional[Any],
    ort_benchmarks: Dict[str, Any],
    accuracy: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "source_model": str(model_path),
        "preset": preset,
        "input": input_info,
        "system": collect_system_info(),
        "onnx_optimization": onnx_optimization,
        "artifacts": {
            "optimized_onnx": {
                "path": str(optimized_onnx_path),
                "size_mb": file_size_mb(optimized_onnx_path),
            },
            "ort_optimized_export": {
                "path": str(ort_optimized_export_path) if ort_optimized_export_path else None,
                "size_mb": file_size_mb(ort_optimized_export_path) if ort_optimized_export_path else 0.0,
            },
            "tensorrt": asdict(trt_artifacts) if trt_artifacts is not None else None,
        },
        "benchmarks": {
            "onnxruntime": ort_benchmarks,
            "tensorrt": trt_artifacts.benchmark if trt_artifacts is not None else None,
        },
        "accuracy": accuracy,
    }
