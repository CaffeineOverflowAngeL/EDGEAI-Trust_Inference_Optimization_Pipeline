# EDGEAI-Trust Inference Optimization Pipeline

<div align="center">
  <img src="assets/edgeai_trust_logo.webp" width="420px" alt="EDGEAI-Trust logo">
  <br>
  <img src="assets/edgeai_infpipeline_diagram.png" width="65%" alt="Inference optimization pipeline">
</div>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-2196f3.svg" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/ONNX_Runtime-supported-673ab7.svg" alt="ONNX Runtime supported">
  <img src="https://img.shields.io/badge/TVM-CPU%20%7C%20CUDA-9c27b0.svg" alt="TVM CPU and CUDA">
  <img src="https://img.shields.io/badge/TensorRT-FP16%20%7C%20FP32%20%7C%20INT8-ff9800.svg" alt="TensorRT FP16, FP32, and INT8">
</p>

A framework for benchmarking ONNX inference backends and producing validated,
device-specific deployment artifacts.

Supported backends:

- ONNX Runtime CPU and CUDA
- NVIDIA TensorRT FP32, FP16, and INT8
- Apache TVM CPU and CUDA

## Repository Structure

```text
bencmarks/        Backend performance exploration
device_builder/   Artifact generation and numerical validation
scripts/          Command-line entry points
tests/            TVM diagnostics
```

## Environment Setup

Use Python 3.11 or newer and select dependencies for the target backend.

```bash
git clone https://github.com/CaffeineOverflowAngeL/EDGEAI-Trust_Inference_Optimization_Pipeline.git
cd EDGEAI-Trust_Inference_Optimization_Pipeline

python3 -m venv edgeai_venv
source edgeai_venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

The core requirements support ONNX Runtime CPU and CUDA execution. CUDA
execution also requires a compatible NVIDIA driver on the target machine.

For TensorRT on CUDA 12:

```bash
python -m pip install -r requirements-cuda12.txt

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

python -m pip install pycuda==2025.1.2
```

PyCUDA may compile from source. Set `CUDA_HOME` to the actual toolkit location
for the target machine before installing it. A C++ compiler, Python development
headers, and CUDA development headers are required for a source build.

Check the available runtime providers:

```bash
python - <<'PY'
import onnxruntime as ort
print(ort.get_available_providers())
PY
```

TVM is not installed by the requirement files because it must be built for the
target CPU/GPU. Use `build_tvm.sh` when TVM support is required.

## Basic Example

Build, benchmark, and validate a TensorRT FP32 artifact:

```bash
python scripts/build_model.py \
  --model path/to/model.onnx \
  --out-dir build_out/model_fp32 \
  --preset gpu_fp32 \
  --sample-input path/to/input.npy \
  --sample-output path/to/output.npy \
  --benchmark \
  --warmup 10 \
  --runs 100
```

Generated outputs include:

```text
build_out/model_fp32/
├── *.optimized.onnx
├── *.plan
├── benchmark_report.json
└── manifest.json
```

Use `--preset cpu_portable` for CPU-only execution. Other presets include
`ort_cuda_fallback`, `gpu_latency`, `gpu_flexible`, `gpu_throughput`,
`gpu_int8`, and `gpu_fp32`.

Validation of optimized output against representative samples is strongly recommended, especially for FP16 and INT8 variants.

See [device_builder/README.md](device_builder/README.md) for pipeline details.
