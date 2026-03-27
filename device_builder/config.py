from __future__ import annotations

from typing import Any, Dict

DEFAULT_RUNS = 500
DEFAULT_WARMUP = 25
DEFAULT_WORKSPACE_MB = 2048
DEFAULT_TIMING_CACHE = "trt_timing.cache"
DEFAULT_MANIFEST = "manifest.json"
DEFAULT_REPORT = "benchmark_report.json"

PRESETS: Dict[str, Dict[str, Any]] = {
    "cpu_portable": {
        "build_onnx": True,
        "build_trt": False,
        "ort_providers": [["CPUExecutionProvider", {}]],
        "graph_optimization": "all",
        "execution_mode": "sequential",
    },
    "ort_cuda_fallback": {
        "build_onnx": True,
        "build_trt": False,
        "ort_providers": [["CUDAExecutionProvider", {}], ["CPUExecutionProvider", {}]],
        "graph_optimization": "all",
        "execution_mode": "sequential",
    },
    "gpu_latency": {
        "build_onnx": True,
        "build_trt": True,
        "precision": "fp16",
        "dynamic": False,
        "workspace_mb": 4096,
        "ort_providers": [["CUDAExecutionProvider", {}], ["CPUExecutionProvider", {}]],
    },
    "gpu_flexible": {
        "build_onnx": True,
        "build_trt": True,
        "precision": "fp16",
        "dynamic": True,
        "workspace_mb": 4096,
        "ort_providers": [["CUDAExecutionProvider", {}], ["CPUExecutionProvider", {}]],
    },
    "gpu_throughput": {
        "build_onnx": True,
        "build_trt": True,
        "precision": "fp16",
        "dynamic": True,
        "workspace_mb": 6144,
        "ort_providers": [["CUDAExecutionProvider", {}], ["CPUExecutionProvider", {}]],
    },
    "gpu_int8": {
        "build_onnx": True,
        "build_trt": True,
        "precision": "int8",
        "dynamic": False,
        "workspace_mb": 4096,
        "ort_providers": [["CUDAExecutionProvider", {}], ["CPUExecutionProvider", {}]],
    },
    "gpu_fp32": {
        "build_onnx": True,
        "build_trt": True,
        "precision": "fp32",
        "dynamic": False,
        "workspace_mb": 2048,
        "ort_providers": [["CUDAExecutionProvider", {}], ["CPUExecutionProvider", {}]],
    },
}