from __future__ import annotations

import argparse

from .config import DEFAULT_RUNS, DEFAULT_TIMING_CACHE, DEFAULT_WARMUP, PRESETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Device-configurable ONNX/TensorRT artifact builder")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()), default="gpu_latency")

    parser.add_argument("--input-shape", type=str, default=None)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--profile-min", type=str, default=None)
    parser.add_argument("--profile-opt", type=str, default=None)
    parser.add_argument("--profile-max", type=str, default=None)

    parser.add_argument("--workspace-mb", type=int, default=None)
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default=None)
    parser.add_argument("--timing-cache", type=str, default=DEFAULT_TIMING_CACHE)
    parser.add_argument("--calib-dir", type=str, default=None)
    parser.add_argument("--max-calib-samples", type=int, default=32)

    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)

    parser.add_argument("--sample-input", type=str, default=None)
    parser.add_argument("--sample-output", type=str, default=None)
    parser.add_argument("--sample-output-index", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)

    parser.add_argument("--disable-onnxsim", action="store_true")
    parser.add_argument("--skip-ort-optimized-export", action="store_true")
    return parser.parse_args()