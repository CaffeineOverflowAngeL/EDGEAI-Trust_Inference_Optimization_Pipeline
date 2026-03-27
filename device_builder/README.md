# Device Builder — Optimized Executable Generation

The `device_builder` pipeline provides a **production-oriented workflow** for transforming an ONNX model into a **device-specific, optimized executable artifact**, ready for integration into C++ inference systems.

Unlike `benchmark.py`, which explores performance across backends, this pipeline focuses on:

- deterministic builds  
- reproducible optimization  
- numerical validation  
- deployment-ready outputs  

---

### Pipeline Overview

The builder performs the following steps:

1. **ONNX Optimization**
   - Shape inference  
   - Optional graph simplification (`onnxsim`)  
   - Clean optimized ONNX export  

2. **Backend Compilation**
   - TensorRT engine generation (`.plan`)
   - Static or dynamic shape support  
   - Precision selection (FP32 / FP16 / INT8)

3. **Execution & Benchmarking**
   - ONNX Runtime reference execution  
   - TensorRT execution  
   - Warmup + latency measurement  

4. **Numerical Validation**
   - `TRT vs ORT` → conversion correctness  
   - `Sample vs ORT` → model correctness  
   - `Sample vs TRT` → deployment correctness  

5. **Artifact Generation**
   - Optimized ONNX model  
   - TensorRT engine  
   - Benchmark report  
   - Full build manifest  

---

### Basic Usage

```bash
python scripts/build_model.py \
  --model path/to/model.onnx \
  --out-dir build_out \
  --preset gpu_fp32 \
  --sample-input input.npy \
  --sample-output output.npy \
  --benchmark