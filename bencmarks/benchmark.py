#!/usr/bin/env python3
"""
benchmark.py

Inference benchmark optimizer for ONNX models using:
 - ONNX Runtime (CPU / CUDA)
 - Apache TVM (CPU / CUDA) with meta-schedule / autoscheduler fallbacks
 - NVIDIA TensorRT (FP32 / FP16) with modern API compatibility

Features / improvements:
 - argparse for CLI options (model path, runs, warmup, tuning trials, device selection)
 - Safer handling of FP16 initializers → FP32 conversion
 - Robust TVM Relax build pipeline with explicit device planning & BindParams
 - Meta-schedule / autoscheduler tuning fallbacks (safe guarded)
 - TensorRT: modern builder APIs and graceful fallbacks for older bindings
 - Improved logging and clearer warnings

Author: Angelos Christos Maroudis
"""

import argparse
import os
import time
import sys
import json
import traceback

import numpy as np
import onnx
import onnxruntime as ort
from onnx import version_converter, shape_inference

# Try-import TVM
try:
    import tvm
    from tvm import relax
    from tvm.relax.frontend.onnx import from_onnx
    from tvm.runtime._tensor import tensor
    TVM_AVAILABLE = True
except Exception as e:
    TVM_AVAILABLE = False
    _tvm_err = e

# Try-import TensorRT and pycuda
TRT_AVAILABLE = False
TRT_PYCUDA = False
try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except Exception as e:
    _trt_err = e
try:
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    import pycuda.gpuarray as gpuarray
    TRT_PYCUDA = True
except Exception as e:
    _pycuda_err = e

# -------------------------
# Defaults
# -------------------------
DEFAULT_MODEL = "Buyutech_Model_Tests_v0.1/model/model.onnx"
DEFAULT_RUNS = 2000
DEFAULT_WARMUP = 10
DEFAULT_TUNING_TRIALS = 256*10
DEFAULT_TUNING_DIR = "tvm_tuning_logs"

# -------------------------
# Utilities
# -------------------------

def size_mb(p):
    try:
        return os.path.getsize(p) / (1024 * 1024)
    except Exception:
        return 0.0


def log(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


# -------------------------
# ONNX helpers
# -------------------------

def enforce_pure_fp32(model_path):
    from onnx import numpy_helper

    model = onnx.load(model_path)
    graph = model.graph

    changed = False

    for t in graph.initializer:
        if t.data_type == onnx.TensorProto.FLOAT16:
            log(f"[INFO] Converting initializer → FP32: {t.name}")
            arr = numpy_helper.to_array(t).astype(np.float32)
            t.CopyFrom(numpy_helper.from_array(arr, t.name))
            changed = True

    for node in list(graph.input) + list(graph.output):
        if node.type and node.type.tensor_type:
            tt = node.type.tensor_type
            if tt.elem_type == onnx.TensorProto.FLOAT16:
                log(f"[INFO] Fixing input/output dtype → FP32: {node.name}")
                tt.elem_type = onnx.TensorProto.FLOAT
                changed = True

    for vi in graph.value_info:
        if vi.type and vi.type.tensor_type:
            tt = vi.type.tensor_type
            if tt.elem_type == onnx.TensorProto.FLOAT16:
                log(f"[INFO] Fixing intermediate tensor → FP32: {vi.name}")
                tt.elem_type = onnx.TensorProto.FLOAT
                changed = True

    for node in graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR and attr.t.data_type == onnx.TensorProto.FLOAT16:
                log(f"[INFO] Fixing node attribute → FP32: {node.name}.{attr.name}")
                arr = numpy_helper.to_array(attr.t).astype(np.float32)
                attr.t.CopyFrom(numpy_helper.from_array(arr, attr.name))
                changed = True

    if changed:
        fp32_path = model_path.replace('.onnx', '_fp32.onnx')
        onnx.save(model, fp32_path)
        log(f"[INFO] Model fully converted to FP32 → {fp32_path}")
        return fp32_path

    return model_path


def upgrade_opset(model_path, target_opset=13):
    try:
        model = onnx.load(model_path)
        opset = model.opset_import[0].version
        if opset < target_opset:
            log(f"[INFO] Upgrading opset from {opset} → {target_opset}")
            upgraded = version_converter.convert_version(model, target_opset)
            new_path = model_path.replace('.onnx', f'_opset{target_opset}.onnx')
            onnx.save(upgraded, new_path)
            return new_path
    except Exception as e:
        log("[WARN] Opset upgrade failed:", e)
    return model_path


# -------------------------
# ORT benchmarking
# -------------------------

def make_ort_session(model_path, providers):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_profiling = False
    sess = ort.InferenceSession(model_path, sess_options=so, providers=providers)
    inp = sess.get_inputs()[0]
    inp_name = inp.name
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    dtype = np.float32
    sample = np.random.rand(*shape).astype(dtype)
    return sess, inp_name, sample


def measure_latency_ort(sess, input_name, sample, warmup=DEFAULT_WARMUP, runs=DEFAULT_RUNS):
    for _ in range(warmup):
        sess.run(None, {input_name: sample})
    start = time.time()
    for _ in range(runs):
        sess.run(None, {input_name: sample})
    end = time.time()
    return (end - start) / runs * 1000  # ms


def benchmark_ort(name, model_path, providers):
    log(f"[ORT {name}] benchmarking...")
    sess, inp_name, sample = make_ort_session(model_path, providers)
    latency_ms = measure_latency_ort(sess, inp_name, sample)
    log(f"[ORT {name}] {latency_ms:.2f} ms")
    return latency_ms, sample


# -------------------------
# TVM utilities
# -------------------------

def try_tune_with_meta_schedule(mod, params, target, work_dir=DEFAULT_TUNING_DIR, trials=DEFAULT_TUNING_TRIALS):
    try:
        from tvm import meta_schedule as ms
    except Exception as e:
        raise ImportError(f"meta_schedule import failed: {e}")

    # Prefer tune_relax, but fall back to general APIs
    if hasattr(ms, "tune_relax"):
        log("[TVM] ms.tune_relax detected — running tuning...")
        return ms.tune_relax(mod, target=target, params=params, work_dir=work_dir, max_trials_global=trials)
    if hasattr(ms, "tune_relay"):
        log("[TVM] ms.tune_relay detected — running tuning...")
        return ms.tune_relay(mod, target=target, params=params, work_dir=work_dir, max_trials_global=trials)
    if hasattr(ms, "tune"):
        log("[TVM] ms.tune detected — running tuning...")
        return ms.tune(mod, target=target, params=params, work_dir=work_dir, max_trials_global=trials)
    raise ImportError("No compatible meta_schedule tuning API found")


def try_tune_with_autoscheduler(task, trials=DEFAULT_TUNING_TRIALS, work_dir=DEFAULT_TUNING_DIR):
    from tvm import auto_scheduler
    log("[TVM] autoscheduler available: running tuning (this can be slow)...")
    measure_option = auto_scheduler.MeasureOption(
        builder=auto_scheduler.LocalBuilder(),
        runner=auto_scheduler.LocalRunner(number=3, repeat=1, min_repeat_ms=100, timeout=10)
    )
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=trials,
        measure_option=measure_option,
        callbacks=[auto_scheduler.RecordToFile(os.path.join(work_dir, "autosched.json"))]
    )
    task.tune(tune_option)
    log("[TVM] autoscheduler tuning finished")


def build_and_run_tvm(model_path, sample, device_kind="cpu", target_opts=None,
                      warmup=DEFAULT_WARMUP, runs=DEFAULT_RUNS):
    if not TVM_AVAILABLE:
        raise RuntimeError(f"TVM not available: {_tvm_err}")

    import tvm
    from tvm import relax
    from tvm.relax.frontend.onnx import from_onnx
    from tvm.runtime._tensor import tensor
    try:
        from onnx import numpy_helper
    except Exception:
        numpy_helper = None

    onnx_model = onnx.load(model_path)
    input_name = onnx_model.graph.input[0].name
    shape_dict = {input_name: tuple(sample.shape)}

    # Extract initializers (weights) to numpy arrays
    initial_params = {}
    if numpy_helper is not None:
        for init in onnx_model.graph.initializer:
            try:
                arr = numpy_helper.to_array(init)
                arr32 = np.ascontiguousarray(arr.astype(np.float32))
                initial_params[init.name] = arr32
            except Exception:
                pass
    if initial_params:
        log(f"[TVM Relax] extracted {len(initial_params)} initializers from ONNX")

    # Import ONNX → Relax
    log("[TVM Relax] Importing ONNX → Relax IR...")
    res = from_onnx(onnx_model, shape_dict)
    if isinstance(res, tuple) and len(res) == 2:
        mod, params_from_import = res
    else:
        mod = res
        params_from_import = {}

    # Merge params (imported override initial)
    merged_params = {}
    merged_params.update(initial_params)
    merged_params.update(params_from_import or {})

    # Normalize params to numpy float32
    normalized = {}
    for k, v in (merged_params or {}).items():
        try:
            if hasattr(v, "asnumpy"):
                npv = v.asnumpy().astype(np.float32)
            else:
                npv = np.array(v, copy=False)
                if npv.dtype != np.float32:
                    npv = npv.astype(np.float32)
            npv = np.ascontiguousarray(npv)
            normalized[k] = npv
        except Exception:
            normalized[k] = v

    log("[TVM Relax] params keys:", list(normalized.keys())[:60])
    for k in list(normalized.keys())[:10]:
        try:
            log("   ", k, np.array(normalized[k]).shape)
        except Exception:
            log("   ", k, "<unreadable shape>")

    # Target and device
    if device_kind == "cuda":
        target = tvm.target.Target({
            "kind": "cuda",
            "arch": "sm_86",               # RTX 3060 / Ampere
            "max_num_threads": 1024,
            "max_threads_per_block": 1024, # REQUIRED for meta-schedule
            "thread_warp_size": 32,
        })
        dev = tvm.cuda(0)
    else:
        target = tvm.target.Target("llvm")
        dev = tvm.cpu()

    is_cuda = ("cuda" in str(target).lower()) or (device_kind == "cuda")

    # Convert params to device tensors when CUDA is used
    if is_cuda:
        log("[TVM Relax] Converting params to device tensors for CUDA...")
        device_params = {}
        for k, npv in normalized.items():
            try:
                if not isinstance(npv, np.ndarray):
                    npv = np.array(npv, dtype=np.float32)
                npv = np.ascontiguousarray(npv.astype(np.float32))
                dev_t = tensor(npv, dev)
                device_params[k] = dev_t
            except Exception as e:
                log(f"[WARN] failed to create device tensor for {k}: {e}; keeping numpy")
                device_params[k] = npv
    else:
        device_params = normalized

    build_params = device_params or {}
    log(f"[TVM build] target={target} (passing {len(build_params)} params into build)")

    ex = None

    # -----------------------------------------------------------
    # CUDA path: adaptive transforms, tuning, and scheduling
    # -----------------------------------------------------------
    if is_cuda:
        try:
            log("[TVM Relax] Preparing CUDA transform pipeline...")

            # Step 1: Legalize ops
            try:
                mod = relax.transform.LegalizeOps()(mod)
                log("[TVM Relax] LegalizeOps applied.")
            except Exception as e:
                log("[WARN] LegalizeOps failed or skipped:", e)

            # Step 2: check params — skip BindParams if only input
            main_func = mod["main"]
            param_names = [str(p) for p in main_func.params]
            need_bind = len(param_names) > 1
            if need_bind:
                log(f"[TVM Relax] main() params detected {param_names} — will BindParams.")
            else:
                log(f"[TVM Relax] main() has only {param_names} — skipping BindParams step.")

            # Step 3: build transform pipeline (version-safe)
            pipeline = [relax.transform.LiftTransformParams()]
            if need_bind and hasattr(relax.transform, "BindParams"):
                pipeline.append(relax.transform.BindParams("main", normalized))
            if hasattr(relax.transform, "RealizeVDevice"):
                pipeline.append(relax.transform.RealizeVDevice())
            elif hasattr(relax.transform, "PlanDevices"):
                pipeline.append(relax.transform.PlanDevices("cuda"))
            if hasattr(relax.transform, "AnnotateTIROpPattern"):
                pipeline.append(relax.transform.AnnotateTIROpPattern())
            if hasattr(relax.transform, "FuseTIR"):
                pipeline.append(relax.transform.FuseTIR())

            seq = tvm.transform.Sequential(pipeline)
            with tvm.transform.PassContext(opt_level=3):
                mod = seq(mod)
            log("[TVM Relax] CUDA transform pipeline applied successfully.")

            # Step 4: meta-schedule tuning (safe)
            tuned_db = None
            try:
                from tvm import meta_schedule as ms
                log("[TVM] meta_schedule available: attempting tuning...")

                if hasattr(ms, "tune_relax"):
                    log("[TVM] ms.tune_relax detected — running tuning...")
                    tuned_db = ms.tune_relax(mod, target=target,
                                             params=normalized,
                                             work_dir=DEFAULT_TUNING_DIR,
                                             max_trials_global=DEFAULT_TUNING_TRIALS)
                elif hasattr(ms, "tune_tir"):
                    log("[TVM] ms.tune_tir detected — running tuning...")
                    tuned_db = ms.tune_tir(mod, target=target,
                                           work_dir=DEFAULT_TUNING_DIR,
                                           max_trials_global=DEFAULT_TUNING_TRIALS)
                elif hasattr(ms, "tune_tasks"):
                    log("[TVM] ms.tune_tasks detected — running tuning...")
                    try:
                        tasks = ms.extract_task_from_ir(mod, target) \
                                 if hasattr(ms, "extract_task_from_ir") else None
                    except Exception:
                        tasks = None
                    if tasks:
                        tuned_db = ms.tune_tasks(tasks, task_weights=[1]*len(tasks),
                                                 work_dir=DEFAULT_TUNING_DIR,
                                                 max_trials_global=DEFAULT_TUNING_TRIALS)
                    else:
                        log("[WARN] No tasks extracted for tune_tasks; skipping.")
                else:
                    log("[TVM] No compatible tuning API found; skipping.")
            except Exception as e:
                log(f"[WARN] meta_schedule tuning failed: {e}")
                tuned_db = None

            # Step 5: build (with tuned DB or fallback)
            if tuned_db is not None:
                try:
                    from tvm import meta_schedule as ms
                    with ms.ApplyHistoryBest(tuned_db):
                        with tvm.transform.PassContext(opt_level=3):
                            ex = relax.build(mod, target=target, params=build_params)
                    log("[TVM] Built with ApplyHistoryBest from tuned DB.")
                except Exception as e:
                    log("[WARN] Building with tuned DB failed:", e)
                    tuned_db = None

            if tuned_db is None:
                try:
                    with tvm.target.Target("cuda"):
                        try:
                            mod = tvm.tir.transform.DefaultGPUSchedule()(mod)
                            log("[TVM] DefaultGPUSchedule applied.")
                        except Exception as e:
                            log("[WARN] DefaultGPUSchedule failed or unavailable:", e)
                    with tvm.transform.PassContext(opt_level=3):
                        ex = relax.build(mod, target=target, params=build_params)
                    log("[TVM] Built without tuning (DefaultGPUSchedule fallback).")
                except Exception as e:
                    raise RuntimeError(f"relax.build failed after CUDA transform: {e}") from e

        except Exception as e:
            log(f"[WARN] CUDA transform pipeline failed or skipped: {e}")
            with tvm.transform.PassContext(opt_level=3):
                ex = relax.build(mod, target=target, params=build_params)

    # -----------------------------------------------------------
    # CPU fallback
    # -----------------------------------------------------------
    else:
        try:
            with tvm.transform.PassContext(opt_level=3):
                ex = relax.build(mod, target=target, params=build_params)
        except Exception as e:
            raise RuntimeError(f"TVM CPU build failed: {e}")

    # -----------------------------------------------------------
    # Run the built model
    # -----------------------------------------------------------
    try:
        vm = relax.VirtualMachine(ex, dev)
    except Exception:
        vm = relax.VirtualMachine(ex, dev)

    # Prepare input tensor
    try:
        inp = tvm.nd.array(sample.astype("float32"), dev)
    except Exception:
        try:
            inp = tensor(sample.astype("float32"), dev)
        except Exception:
            inp = sample.astype("float32")

    # Execute
    try:
        for _ in range(warmup):
            vm["main"](inp)
        start = time.time()
        for _ in range(runs):
            vm["main"](inp)
        end = time.time()
    except Exception as e:
        log("[WARN] vm['main'] call failed:", e)
        try:
            f = vm.module["main"]
            for _ in range(warmup):
                f(inp)
            start = time.time()
            for _ in range(runs):
                f(inp)
            end = time.time()
        except Exception as e2:
            raise RuntimeError(f"Failed to invoke Relax VM: {e2}")

    avg_ms = (end - start) / runs * 1000
    return avg_ms


# -------------------------
# TensorRT helpers
# -------------------------

def build_trt_engine_from_onnx(onnx_path, max_batch=1, fp16=False):
    if not TRT_AVAILABLE:
        raise RuntimeError(f"TensorRT not available: {_trt_err}")

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            msg = "Failed to parse ONNX model with TensorRT:\n"
            for i in range(parser.num_errors):
                msg += str(parser.get_error(i)) + "\n"
            raise RuntimeError(msg)

    config = builder.create_builder_config()
    # Set a reasonable workspace limit
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20)
    except Exception:
        pass

    if fp16:
        try:
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            else:
                log("[TRT] FP16 not supported on this platform, building FP32 instead.")
        except Exception:
            try:
                config.set_flag(trt.BuilderFlag.FP16)
            except Exception:
                log("[TRT] Could not set FP16 flag; continuing with FP32")

    # Build serialized engine if available in API, else fallback to older build_cuda_engine
    serialized_engine = None
    engine = None
    try:
        if hasattr(builder, 'build_serialized_network'):
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                raise RuntimeError('Failed to build serialized TensorRT engine')
            runtime = trt.Runtime(logger)
            engine = runtime.deserialize_cuda_engine(serialized_engine)
        else:
            # older API
            engine = builder.build_engine(network, config)
            if engine is None:
                raise RuntimeError('Failed to build TensorRT engine')
    except Exception as e:
        raise RuntimeError(f"TensorRT engine build failed: {e}")

    return engine


def run_trt_engine(engine, sample, warmup=DEFAULT_WARMUP, runs=DEFAULT_RUNS):
    if not TRT_AVAILABLE or not TRT_PYCUDA:
        raise RuntimeError("TensorRT or pycuda not available")

    context = engine.create_execution_context()

    # Prepare bindings and device buffers
    bindings = []
    dptrs = []
    stream = cuda.Stream()

    for binding in engine:
        try:
            shape = engine.get_binding_shape(binding)
            # handle -1 dims conservatively by using the sample's shape
            shape = tuple([s if s != -1 else sample.shape[i] if i < len(sample.shape) else 1 for i, s in enumerate(shape)])
            size = int(np.prod(shape))
        except Exception:
            size = sample.size

        # Determine dtype: prefer engine.get_binding_dtype if available
        try:
            if hasattr(engine, 'get_binding_dtype'):
                dtype = trt.nptype(engine.get_binding_dtype(binding))
            else:
                # fallback assume float32
                dtype = np.float32
        except Exception:
            dtype = np.float32

        dev_mem = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
        dptrs.append(dev_mem)
        bindings.append(int(dev_mem))

    # Copy input to device
    h_input = sample.astype(np.float32).ravel()
    try:
        cuda.memcpy_htod(dptrs[0], h_input)
    except Exception as e:
        raise RuntimeError(f"Failed to copy input to GPU: {e}")

    # Warmup + timed runs
    for _ in range(warmup):
        context.execute_v2(bindings)
    start = time.time()
    for _ in range(runs):
        context.execute_v2(bindings)
    end = time.time()
    avg_ms = (end - start) / runs * 1000
    return avg_ms


# -------------------------
# Runner orchestration
# -------------------------

def run_all_benchmarks(model_path, input_shape, args):
    results = {}
    sample = np.random.RandomState(42).randn(*input_shape).astype(np.float32)

    # ORT Baseline (CPU)
    try:
        lat_base, _ = benchmark_ort("Baseline (CPU)", model_path, ["CPUExecutionProvider"])
        results["ORT Baseline (CPU)"] = (lat_base, model_path)
    except Exception as e:
        log(f"[WARN] ORT baseline failed: {e}")
        results["ORT Baseline (CPU)"] = (float('inf'), model_path)

    # ORT GraphOpt (CPU)
    try:
        lat_opt, _ = benchmark_ort("Graph Opt (CPU)", model_path, ["CPUExecutionProvider"])
        results["ORT GraphOpt (CPU)"] = (lat_opt, model_path)
    except Exception as e:
        log(f"[WARN] ORT graph opt failed: {e}")

    # ORT CUDA (if available)
    try:
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            lat_gpu, _ = benchmark_ort("GPU (CUDA)", model_path, ["CUDAExecutionProvider"])
            results["ORT (CUDA)"] = (lat_gpu, model_path)
        else:
            log("[INFO] ORT CUDAExecutionProvider not available; skipping ORT GPU bench")
    except Exception as e:
        log(f"[WARN] ORT GPU bench failed: {e}")

    # TVM CPU
    if TVM_AVAILABLE and args.run_tvm:
        try:
            target_opts = {"target": "llvm", "ctx": tvm.cpu()}
            log("[TVM CPU] building and running ...")
            lat_tvm_cpu = build_and_run_tvm(model_path, sample, device_kind="cpu", target_opts=target_opts, warmup=args.warmup, runs=args.runs)
            results["TVM (CPU)"] = (lat_tvm_cpu, model_path)
            log(f"[TVM CPU] {lat_tvm_cpu:.2f} ms")
        except Exception as e:
            log(f"[WARN] TVM CPU failed: {e}\n", traceback.format_exc())
    else:
        if not TVM_AVAILABLE:
            log("[INFO] TVM not installed — skipping TVM benchmarks.")

    # TVM CUDA
    if TVM_AVAILABLE and args.run_tvm and tvm.runtime.enabled("cuda"):
        try:
            target_opts = {"target": "cuda", "ctx": tvm.cuda()}
            log("[TVM GPU] building and running ...")
            lat_tvm_gpu = build_and_run_tvm(model_path, sample, device_kind="cuda", target_opts=target_opts, warmup=args.warmup, runs=args.runs)
            results["TVM (CUDA)"] = (lat_tvm_gpu, model_path)
            log(f"[TVM CUDA] {lat_tvm_gpu:.2f} ms")
        except Exception as e:
            log(f"[WARN] TVM GPU failed: {e}\n", traceback.format_exc())
    else:
        if TVM_AVAILABLE:
            if not tvm.runtime.enabled("cuda"):
                log("[WARN] TVM built without CUDA enabled — skipping TVM GPU.")

    # TensorRT
    if TRT_AVAILABLE and TRT_PYCUDA and args.run_trt:
        try:
            log("[TensorRT FP32] building engine ... (this may take a while)")
            engine_fp32 = build_trt_engine_from_onnx(model_path, fp16=False)
            lat_trt_fp32 = run_trt_engine(engine_fp32, sample, warmup=args.warmup, runs=args.runs)
            results["TensorRT (FP32)"] = (lat_trt_fp32, model_path)
            log(f"[TensorRT FP32] {lat_trt_fp32:.2f} ms")
        except Exception as e:
            log(f"[WARN] TensorRT FP32 failed: {e}\n", traceback.format_exc())

        try:
            log("[TensorRT FP16] building engine (if supported) ...")
            engine_fp16 = build_trt_engine_from_onnx(model_path, fp16=True)
            lat_trt_fp16 = run_trt_engine(engine_fp16, sample, warmup=args.warmup, runs=args.runs)
            results["TensorRT (FP16)"] = (lat_trt_fp16, model_path)
            log(f"[TensorRT FP16] {lat_trt_fp16:.2f} ms")
        except Exception as e:
            log(f"[WARN] TensorRT FP16 failed: {e}\n", traceback.format_exc())
    else:
        if not TRT_AVAILABLE or not TRT_PYCUDA:
            log("[INFO] TensorRT or pycuda not available — skipping TensorRT benchmarks.")
            if not TRT_AVAILABLE:
                log(f"       Install TensorRT Python bindings. Error: {_trt_err}")
            if not TRT_PYCUDA:
                log(f"       Install pycuda or cuda-python. Error: {_pycuda_err}")

    # Report
    log("\n📊 Final Performance Report")
    log("-------------------------------------------------------------")
    log(f"{'Backend':<22} | {'Latency (ms)':>12} | {'Speedup':>7} | {'Size (MB)':>8}")
    log("-------------------------------------------------------------")
    base = results.get("ORT Baseline (CPU)", (None,))[0] or 1.0
    for k, (lat, path) in results.items():
        s = size_mb(path)
        speedup = base / lat if lat and lat != float('inf') else 0.0
        log(f"{k:<22} | {lat:12.2f} | {speedup:7.2f}× | {s:8.6f}")
    log("-------------------------------------------------------------")
    return results


# -------------------------
# CLI
# -------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Modern benchmark runner for ONNX models (ORT / TVM / TensorRT)')
    p.add_argument('--model', '-m', default=DEFAULT_MODEL, help='Path to ONNX model')
    p.add_argument('--runs', type=int, default=DEFAULT_RUNS, help='Number of timed runs')
    p.add_argument('--warmup', type=int, default=DEFAULT_WARMUP, help='Warmup runs')
    p.add_argument('--tuning-trials', type=int, default=DEFAULT_TUNING_TRIALS, help='Meta-schedule/autoscheduler trials')
    p.add_argument('--no-tvm', dest='run_tvm', action='store_false', help='Skip TVM benchmarks')
    p.add_argument('--no-trt', dest='run_trt', action='store_false', help='Skip TensorRT benchmarks')
    p.add_argument('--input-shape', type=str, default=None, help='Override input shape as JSON list, e.g. "[1,3,224,224]"')
    p.add_argument('--opset', type=int, default=13, help='OpSet to upgrade ONNX to (if lower)')
    return p.parse_args()


def main():
    args = parse_args()

    model_path = args.model
    if not os.path.exists(model_path):
        log(f"Model not found: {model_path}")
        sys.exit(2)

    # FP16 -> FP32 conversion if needed
    model_path = enforce_pure_fp32(model_path)
    model_path = upgrade_opset(model_path, target_opset=args.opset)

    log("[INFO] Running ONNX shape inference...")
    model = onnx.load(model_path)
    inferred = shape_inference.infer_shapes(model)
    shaped_path = model_path.replace('.onnx', f'_opset{args.opset}_shaped.onnx')
    onnx.save(inferred, shaped_path)
    log(f"[INFO] Shape inference complete → {shaped_path}")

    if args.input_shape:
        try:
            input_shape = tuple(json.loads(args.input_shape))
        except Exception:
            log("Invalid --input-shape JSON; falling back to (1,12,128,256)")
            input_shape = (1, 12, 128, 256)
    else:
        input_shape = (1, 12, 128, 256)

    # set globals for runs/warmup tuning
    args.runs = args.runs
    args.warmup = args.warmup

    run_all_benchmarks(shaped_path, input_shape, args)


if __name__ == '__main__':
    main()
