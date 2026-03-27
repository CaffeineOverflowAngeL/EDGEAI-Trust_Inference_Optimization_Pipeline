# EDGEAI-Trust Inference Optimization Pipeline

<div align="center">
<img src="assets/edgeai_trust_logo.webp" width="420px"></img>
<h2></h2>
<img src="assets/edgeai_infpipeline_diagram.png" width="65%">
<h2></h2>
</div>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/tests-passing-4caf50.svg"></a>
  <a href="#"><img src="https://img.shields.io/badge/Python-3.9%2B-2196f3.svg"></a>
  <a href="#"><img src="https://img.shields.io/badge/ONNXRuntime-supported-673ab7.svg"></a>
  <a href="#"><img src="https://img.shields.io/badge/TVM-CUDA%20Enabled-9c27b0.svg"></a>
  <a href="#"><img src="https://img.shields.io/badge/TensorRT-FP16%20%7C%20FP32-ff9800.svg"></a>
  <a href="#"><img src="https://img.shields.io/badge/License-MIT-4caf50.svg"></a>
</p>

# EDGEAI-Trust Inference Optimization Pipeline

A unified framework for **benchmarking inference backends** and **building optimized deployment artifacts** for ONNX models.

Supported backends:
- ONNX Runtime (CPU / CUDA)
- Apache TVM (CPU / CUDA, Relax)
- NVIDIA TensorRT (FP32 / FP16 / INT8)

---

# 🧩 Repository Structure

The project is split into two independent components:

```text
/benchmarks      # Backend performance exploration
/device_builder  # Optimized artifact generation