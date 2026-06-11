from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..io_utils import log

try:
    import tensorrt as trt  # type: ignore
    TRT_AVAILABLE = True
except Exception as e:
    TRT_AVAILABLE = False
    TRT_IMPORT_ERROR = e

try:
    import pycuda.driver as cuda  # type: ignore
    PYCUDA_AVAILABLE = True
except Exception as e:
    PYCUDA_AVAILABLE = False
    PYCUDA_IMPORT_ERROR = e


@dataclass
class TrtBuildArtifacts:
    engine_path: Optional[str]
    timing_cache_path: Optional[str]
    calibration_cache_path: Optional[str]
    precision: str
    dynamic: bool
    workspace_mb: int
    benchmark: Optional[Dict[str, Any]] = None


def _ensure_pycuda_context() -> None:
    if not PYCUDA_AVAILABLE:
        raise RuntimeError(f"pycuda not available: {PYCUDA_IMPORT_ERROR}")

    # Delay CUDA initialization so CLI help and CPU-only presets work without
    # an accessible GPU, even when PyCUDA is installed.
    import pycuda.autoinit  # noqa: F401  # type: ignore


def trt_dtype_nptype(dtype: Any) -> np.dtype:
    return np.dtype(trt.nptype(dtype))


def load_calibration_samples(
    calib_dir: Path,
    input_shape: Sequence[int],
    max_samples: int = 32,
) -> List[np.ndarray]:
    files = sorted(calib_dir.glob("*.npy"))
    if not files:
        raise RuntimeError(f"No .npy calibration files found in {calib_dir}")

    samples: List[np.ndarray] = []
    for fp in files[:max_samples]:
        arr = np.load(fp)
        arr = np.asarray(arr, dtype=np.float32)
        if tuple(arr.shape) != tuple(input_shape):
            raise ValueError(
                f"Calibration file {fp} has shape {arr.shape}, expected {tuple(input_shape)}"
            )
        samples.append(np.ascontiguousarray(arr))
    return samples


class NpyEntropyCalibrator(trt.IInt8EntropyCalibrator2 if TRT_AVAILABLE else object):
    def __init__(self, samples: List[np.ndarray], cache_file: Path):
        if not TRT_AVAILABLE or not PYCUDA_AVAILABLE:
            raise RuntimeError("TensorRT and pycuda are required for INT8 calibration")
        super().__init__()

        if not samples:
            raise ValueError("Calibration samples list is empty")

        self.samples = [np.ascontiguousarray(s.astype(np.float32)) for s in samples]
        self.cache_file = cache_file
        self.index = 0
        self.batch_size = 1
        self.device_input = cuda.mem_alloc(self.samples[0].nbytes)

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names: Sequence[str]) -> Optional[List[int]]:
        if self.index >= len(self.samples):
            return None

        sample = self.samples[self.index]
        cuda.memcpy_htod(self.device_input, sample)
        self.index += 1
        return [int(self.device_input)]

    def read_calibration_cache(self) -> Optional[bytes]:
        if self.cache_file.exists():
            return self.cache_file.read_bytes()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        self.cache_file.write_bytes(cache)


def parse_trt_network_from_onnx(onnx_path: Path, logger: Any) -> Tuple[Any, Any, Any]:
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        ok = parser.parse(f.read())

    if not ok:
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("Failed to parse ONNX with TensorRT:\n" + "\n".join(errors))

    return builder, network, parser


def apply_precision_flags(
    builder: Any,
    config: Any,
    precision: str,
    calibrator: Optional[Any] = None,
) -> None:
    precision = precision.lower()

    if precision == "fp32":
        return

    if precision == "fp16":
        if not getattr(builder, "platform_has_fast_fp16", False):
            log("[TensorRT] Warning: platform_has_fast_fp16=False, enabling FP16 anyway.")
        config.set_flag(trt.BuilderFlag.FP16)
        return

    if precision == "int8":
        if not getattr(builder, "platform_has_fast_int8", True):
            log("[TensorRT] Warning: platform_has_fast_int8=False, enabling INT8 anyway.")
        if calibrator is None:
            raise ValueError("INT8 requested but no calibrator provided")
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calibrator
        return

    raise ValueError(f"Unsupported precision: {precision}")


def build_trt_engine(
    onnx_path: Path,
    engine_path: Path,
    precision: str,
    workspace_mb: int,
    dynamic: bool,
    input_name: str,
    static_shape: Optional[List[int]],
    profile_min: Optional[List[int]],
    profile_opt: Optional[List[int]],
    profile_max: Optional[List[int]],
    timing_cache_path: Optional[Path],
    calibration_samples: Optional[List[np.ndarray]],
    calibration_cache_path: Optional[Path],
) -> TrtBuildArtifacts:
    if not TRT_AVAILABLE:
        raise RuntimeError(f"TensorRT not available: {TRT_IMPORT_ERROR}")
    if not PYCUDA_AVAILABLE:
        raise RuntimeError(f"pycuda not available: {PYCUDA_IMPORT_ERROR}")

    _ensure_pycuda_context()

    logger = trt.Logger(trt.Logger.WARNING)
    builder, network, _ = parse_trt_network_from_onnx(onnx_path, logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_mb) << 20)

    calibrator = None
    calib_cache_str: Optional[str] = None

    if dynamic:
        if profile_min is None or profile_opt is None or profile_max is None:
            raise ValueError(
                "Dynamic TensorRT build requires --profile-min, --profile-opt, and --profile-max"
            )

        profile = builder.create_optimization_profile()
        profile.set_shape(
            input_name,
            tuple(profile_min),
            tuple(profile_opt),
            tuple(profile_max),
        )
        config.add_optimization_profile(profile)
    else:
        if static_shape is None:
            raise ValueError("Static TensorRT build requires static input shape")
        network.get_input(0).shape = tuple(static_shape)

    if precision.lower() == "int8":
        if calibration_samples is None or calibration_cache_path is None:
            raise ValueError("INT8 build requires calibration samples and calibration cache path")

        calibrator = NpyEntropyCalibrator(calibration_samples, calibration_cache_path)
        calib_cache_str = str(calibration_cache_path)

        if dynamic:
            calib_profile = builder.create_optimization_profile()
            calib_profile.set_shape(
                input_name,
                tuple(profile_opt),
                tuple(profile_opt),
                tuple(profile_opt),
            )
            config.set_calibration_profile(calib_profile)

    apply_precision_flags(builder, config, precision, calibrator=calibrator)

    if timing_cache_path is not None:
        try:
            cache_bytes = timing_cache_path.read_bytes() if timing_cache_path.exists() else b""
            timing_cache = config.create_timing_cache(cache_bytes)
            config.set_timing_cache(timing_cache, ignore_mismatch=False)
            if timing_cache_path.exists():
                log(f"[TensorRT] Loaded timing cache: {timing_cache_path}")
        except Exception as exc:
            log(f"[TensorRT] Warning: timing cache unavailable: {exc}")

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("TensorRT build_serialized_network returned None")

    engine_path.write_bytes(bytes(serialized_engine))
    log(f"[TensorRT] Wrote engine: {engine_path}")

    if timing_cache_path is not None:
        try:
            cache = config.get_timing_cache()
            if cache is not None:
                timing_cache_path.write_bytes(cache.serialize())
                log(f"[TensorRT] Wrote timing cache: {timing_cache_path}")
        except Exception as exc:
            log(f"[TensorRT] Warning: failed to save timing cache: {exc}")

    return TrtBuildArtifacts(
        engine_path=str(engine_path),
        timing_cache_path=str(timing_cache_path) if timing_cache_path else None,
        calibration_cache_path=calib_cache_str,
        precision=precision,
        dynamic=dynamic,
        workspace_mb=workspace_mb,
    )


def load_trt_engine(engine_path: Path) -> Any:
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize engine: {engine_path}")
    return engine


def _tensor_name(engine: Any, index: int) -> str:
    if hasattr(engine, "get_tensor_name"):
        return engine.get_tensor_name(index)
    return engine[index]


def _is_input(engine: Any, name: str) -> bool:
    if hasattr(engine, "get_tensor_mode"):
        return engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
    return engine.binding_is_input(name)


def _tensor_dtype(engine: Any, name: str) -> np.dtype:
    if hasattr(engine, "get_tensor_dtype"):
        return trt_dtype_nptype(engine.get_tensor_dtype(name))
    return trt_dtype_nptype(engine.get_binding_dtype(name))


def _tensor_shape(engine: Any, context: Any, name: str) -> Tuple[int, ...]:
    if hasattr(context, "get_tensor_shape"):
        return tuple(context.get_tensor_shape(name))
    return tuple(context.get_binding_shape(engine[name]))


def _execute_with_context(
    engine: Any,
    context: Any,
    stream: Any,
    allocations: Dict[str, Any],
    input_tensor_name: str,
    output_tensor_names: List[str],
    input_arr: np.ndarray,
    host_outputs: Dict[str, np.ndarray],
) -> List[np.ndarray]:
    cuda.memcpy_htod_async(allocations[input_tensor_name], input_arr, stream)

    if hasattr(context, "execute_async_v3"):
        ok = context.execute_async_v3(stream_handle=stream.handle)
    else:
        bindings = [0] * len(engine)
        for idx in range(len(engine)):
            name = engine[idx]
            bindings[idx] = int(allocations[name])
        ok = context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    if not ok:
        raise RuntimeError("TensorRT execution failed")

    for name in output_tensor_names:
        cuda.memcpy_dtoh_async(host_outputs[name], allocations[name], stream)

    stream.synchronize()
    return [host_outputs[name].copy() for name in output_tensor_names]


def prepare_trt_runner(
    engine_path: Path,
    sample: np.ndarray,
    dynamic: bool,
) -> Callable[[], List[np.ndarray]]:
    if not TRT_AVAILABLE or not PYCUDA_AVAILABLE:
        raise RuntimeError("TensorRT and pycuda are required")

    _ensure_pycuda_context()

    engine = load_trt_engine(engine_path)
    context = engine.create_execution_context()
    stream = cuda.Stream()

    count = engine.num_io_tensors if hasattr(engine, "num_io_tensors") else len(engine)

    input_tensor_name: Optional[str] = None
    output_tensor_names: List[str] = []

    for i in range(count):
        name = _tensor_name(engine, i)
        if _is_input(engine, name):
            input_tensor_name = name
        else:
            output_tensor_names.append(name)

    if input_tensor_name is None:
        raise RuntimeError("No input tensor found")

    if dynamic:
        if hasattr(context, "set_input_shape"):
            context.set_input_shape(input_tensor_name, tuple(sample.shape))
        else:
            raise RuntimeError("Dynamic shape runtime requires set_input_shape support")

    allocations: Dict[str, Any] = {}
    host_outputs: Dict[str, np.ndarray] = {}

    all_names = [input_tensor_name] + output_tensor_names
    for name in all_names:
        dtype = _tensor_dtype(engine, name)
        shape = _tensor_shape(engine, context, name)

        if any(dim < 0 for dim in shape):
            raise RuntimeError(f"Unresolved TensorRT shape for {name}: {shape}")

        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        allocations[name] = cuda.mem_alloc(nbytes)

        if name != input_tensor_name:
            host_outputs[name] = np.empty(shape, dtype=dtype)

    if hasattr(context, "set_tensor_address"):
        for name, ptr in allocations.items():
            context.set_tensor_address(name, int(ptr))

    input_arr = np.ascontiguousarray(
        sample.astype(_tensor_dtype(engine, input_tensor_name), copy=False)
    )

    def runner() -> List[np.ndarray]:
        return _execute_with_context(
            engine=engine,
            context=context,
            stream=stream,
            allocations=allocations,
            input_tensor_name=input_tensor_name,
            output_tensor_names=output_tensor_names,
            input_arr=input_arr,
            host_outputs=host_outputs,
        )

    return runner


def run_trt_once(
    engine_path: Path,
    sample: np.ndarray,
    dynamic: bool,
) -> List[np.ndarray]:
    runner = prepare_trt_runner(engine_path, sample, dynamic)
    return runner()


def benchmark_trt_engine(
    engine_path: Path,
    sample: np.ndarray,
    warmup: int,
    runs: int,
    dynamic: bool,
) -> Dict[str, float]:
    if not TRT_AVAILABLE or not PYCUDA_AVAILABLE:
        raise RuntimeError("TensorRT and pycuda are required for benchmarking engines")

    runner = prepare_trt_runner(engine_path, sample, dynamic)

    for _ in range(warmup):
        runner()

    start = time.perf_counter()
    for _ in range(runs):
        runner()
    end = time.perf_counter()

    return {"latency_ms": (end - start) / runs * 1000.0}
