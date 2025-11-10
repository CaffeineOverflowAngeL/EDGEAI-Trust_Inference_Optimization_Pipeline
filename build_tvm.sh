#!/usr/bin/env bash
set -euo pipefail
LOG=~/tvm_rebuild_$(date +%Y%m%d_%H%M%S).log
TVM_DIR=${TVM_DIR:-"$HOME/tvm"}
BUILD_DIR="$TVM_DIR/build"

echo "Logging to $LOG"
exec > >(tee -a "$LOG") 2>&1

echo "=== START TVM rebuild: $(date) ==="
echo "TVM_DIR = $TVM_DIR"
if [ ! -d "$TVM_DIR" ]; then
  echo "ERROR: TVM_DIR does not exist: $TVM_DIR"
  exit 1
fi

cd "$TVM_DIR"

echo "--- Git status / branch ---"
git rev-parse --abbrev-ref HEAD || true
git status --porcelain || true
git log -1 --pretty=oneline || true

echo "--- Ensure submodules are initialized ---"
git submodule update --init --recursive

echo "--- Remove old build dir (if exists) ---"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "--- CMake configure (with python module ON) ---"
cmake .. \
  -DUSE_CUDA=ON \
  -DUSE_CUDNN=ON \
  -DUSE_CUBLAS=ON \
  -DUSE_LLVM=ON \
  -DTVM_BUILD_PYTHON_MODULE=ON \
  -DUSE_TENSORRT_CODEGEN=OFF \
  -DUSE_TENSORRT_RUNTIME=OFF \
  -DCMAKE_BUILD_TYPE=Release

echo "--- Start build (single core for logs) ---"
# first try a single-job build to get clean errors, then parallel if successful
make -j1 VERBOSE=1

echo "--- If single-job build succeeded, run parallel rebuild for speed ---"
make -j"$(nproc)" || true

echo "--- Install python bindings (editable) ---"
cd "$TVM_DIR/python"
# ensure pip from the environment is used
python3 -m pip uninstall -y tvm || true
python3 -m pip install -e .

echo "--- Verification (python) ---"
python3 - <<'PY'
import sys, tvm
from tvm import relax
print("python executable:", sys.executable)
print("tvm.__file__:", tvm.__file__)
print("tvm.__version__:", getattr(tvm, "__version__", "unknown"))
print("CUDA available:", tvm.cuda().exist)
print("PlanDevices:", hasattr(relax.transform, "PlanDevices"))
print("BindTarget:", hasattr(relax.transform, "BindTarget"))
# print short list of members to inspect
members = [m for m in dir(relax.transform) if not m.startswith("_")]
print("relax.transform members (sample):", members[:40])
PY

echo "=== FINISHED TVM rebuild: $(date) ==="
echo "Log saved in $LOG"
