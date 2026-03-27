#!/usr/bin/env python3
import tvm
import numpy as np
from tvm import relax

print("TVM Diagnostic Check")
print("---------------------------")
print("TVM version:", tvm.__version__)
print("Available targets:", tvm.target.Target.list_kinds())

# Try CUDA device
cuda_enabled = tvm.runtime.enabled("cuda")
print("CUDA runtime enabled:", cuda_enabled)

# Pick device
dev = tvm.device("cuda" if cuda_enabled else "cpu")
print("Selected device:", dev)

# Generate test tensor
data = np.random.rand(1, 12, 128, 256).astype("float32")

# Try all known ways of creating a TVM tensor
print("\nTensor creation tests:")
tensors = {}

try:
    from tvm.runtime._tensor import tensor
    tensors["runtime._tensor.tensor"] = tensor(data, dev)
except Exception as e:
    print("[WARN] tvm.runtime._tensor.tensor failed:", e)

try:
    from tvm.runtime import ndarray
    tensors["runtime.ndarray.array"] = ndarray.array(data, dev)
except Exception as e:
    print("[WARN] tvm.runtime.ndarray.array failed:", e)

try:
    tensors["tvm.nd.array"] = tvm.nd.array(data, dev)
except Exception as e:
    print("[WARN] tvm.nd.array failed:", e)

# Show results
for k, v in tensors.items():
    print(f"{k:<28}: type={type(v)}, device={v.device}, shape={v.shape}")

# Confirm one is actually on GPU (if available)
if cuda_enabled:
    for k, v in tensors.items():
        if "cuda" in str(v.device):
            print(f"\n✅ Success: {k} created a GPU tensor.")
            break
    else:
        print("\n❌ No tensor was created on GPU — check CUDA linkage.")
else:
    print("\nℹ️ CUDA not enabled — all tensors are CPU.")
