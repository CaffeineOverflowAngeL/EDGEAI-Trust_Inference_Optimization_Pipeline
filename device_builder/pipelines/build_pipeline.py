from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import onnxruntime as ort

from ..backends.onnx_utils import (
    infer_primary_input_info,
    load_onnx_model,
    optimize_onnx_model,
    save_onnx,
)
from ..backends.ort_backend import (
    benchmark_ort_session,
    make_ort_session,
    run_ort_once,
)
from ..backends.trt_backend import (
    benchmark_trt_engine,
    build_trt_engine,
    load_calibration_samples,
    run_trt_once,
)
from ..config import DEFAULT_MANIFEST, DEFAULT_REPORT, PRESETS
from ..io_utils import (
    ensure_dir,
    load_json_shape,
    log,
    validate_user_shape_against_model,
    write_json,
)
from ..manifest import build_manifest
from ..samples.npy_samples import load_sample_input, load_sample_output
from ..validation import compare_single_output


def run_build(args: Any) -> int:
    preset = dict(PRESETS[args.preset])

    model_path = Path(args.model).resolve()
    out_dir = Path(args.out_dir).resolve()
    ensure_dir(out_dir)

    static_input_shape = load_json_shape(args.input_shape, "--input-shape")
    profile_min = load_json_shape(args.profile_min, "--profile-min")
    profile_opt = load_json_shape(args.profile_opt, "--profile-opt")
    profile_max = load_json_shape(args.profile_max, "--profile-max")

    model = load_onnx_model(model_path)
    input_name, inferred_shape, input_dtype = infer_primary_input_info(model)

    validate_user_shape_against_model(static_input_shape, inferred_shape, "--input-shape")
    validate_user_shape_against_model(profile_min, inferred_shape, "--profile-min")
    validate_user_shape_against_model(profile_opt, inferred_shape, "--profile-opt")
    validate_user_shape_against_model(profile_max, inferred_shape, "--profile-max")

    if static_input_shape is None and all(dim > 0 for dim in inferred_shape):
        static_input_shape = inferred_shape

    dynamic = bool(args.dynamic or preset.get("dynamic", False))
    precision = str(args.precision or preset.get("precision", "fp32"))
    workspace_mb = int(args.workspace_mb or preset.get("workspace_mb", 2048))

    if not dynamic and static_input_shape is None:
        raise RuntimeError(
            "No static input shape is available. Provide --input-shape or use --dynamic with profile shapes."
        )

    log(f"[INFO] Input tensor: {input_name}")
    log(f"[INFO] Inferred model shape: {inferred_shape}")
    log(f"[INFO] Static input shape: {static_input_shape}")
    log(f"[INFO] Input dtype: {input_dtype}")
    log(f"[INFO] Preset: {args.preset}")
    log(f"[INFO] Precision: {precision}")
    log(f"[INFO] Dynamic TensorRT: {dynamic}")

    optimized_onnx, onnx_opt_info = optimize_onnx_model(
        model,
        try_simplify=not args.disable_onnxsim,
    )
    optimized_onnx_path = out_dir / f"{model_path.stem}.optimized.onnx"
    save_onnx(optimized_onnx, optimized_onnx_path)
    log(f"[ONNX] Wrote optimized model: {optimized_onnx_path}")

    providers = preset.get("ort_providers", [["CPUExecutionProvider", {}]])
    available = ort.get_available_providers()
    active_providers = []
    for provider in providers:
        name = provider[0] if isinstance(provider, (list, tuple)) else provider
        if name in available:
            active_providers.append(provider)
    if not active_providers:
        active_providers = [["CPUExecutionProvider", {}]]

    ort_optimized_export_path: Optional[Path] = None
    if not args.skip_ort_optimized_export:
        ort_optimized_export_path = out_dir / f"{model_path.stem}.ort_optimized.onnx"
        try:
            _ = make_ort_session(
                optimized_onnx_path,
                providers=active_providers,
                graph_optimization=str(preset.get("graph_optimization", "all")),
                execution_mode=str(preset.get("execution_mode", "sequential")),
                optimized_model_path=ort_optimized_export_path,
            )
            log(f"[ORT] Wrote ORT-optimized export: {ort_optimized_export_path}")
        except Exception as exc:
            log(f"[ORT] Warning: failed to export optimized graph: {exc}")
            ort_optimized_export_path = None

    sample_shape = profile_opt if dynamic else static_input_shape
    if sample_shape is None:
        raise RuntimeError("Could not determine sample shape")

    if args.sample_input:
        sample = load_sample_input(Path(args.sample_input), sample_shape, input_dtype)
        log(f"[SAMPLE] Using real input sample: {args.sample_input}")
    else:
        sample = np.random.RandomState(42).randn(*sample_shape).astype(input_dtype)
        log("[SAMPLE] Using synthetic random input")

    ort_benchmarks: Dict[str, Any] = {}
    validation: Dict[str, Any] = {}

    ref_sess = make_ort_session(
        optimized_onnx_path,
        providers=[["CPUExecutionProvider", {}]],
        graph_optimization="all",
    )
    ref_outputs = run_ort_once(ref_sess, input_name, sample)

    if args.benchmark:
        try:
            sess = make_ort_session(
                optimized_onnx_path,
                providers=active_providers,
                graph_optimization=str(preset.get("graph_optimization", "all")),
                execution_mode=str(preset.get("execution_mode", "sequential")),
            )
            ort_benchmarks["optimized"] = benchmark_ort_session(
                sess,
                input_name,
                sample,
                args.warmup,
                args.runs,
            )
            ort_benchmarks["providers"] = sess.get_providers()
            log(f"[ORT] Latency: {ort_benchmarks['optimized']['latency_ms']:.4f} ms")
        except Exception as exc:
            ort_benchmarks["error"] = str(exc)
            log(f"[ORT] Benchmark failed: {exc}")

    trt_artifacts = None
    if preset.get("build_trt", False):
        calibration_samples = None
        calibration_cache_path = (
            out_dir / f"{model_path.stem}.{precision}.calib.cache"
            if precision == "int8"
            else None
        )

        if precision == "int8":
            if args.calib_dir is None:
                raise RuntimeError("INT8 preset selected but --calib-dir was not provided")
            calib_shape = static_input_shape if static_input_shape is not None else profile_opt
            if calib_shape is None:
                raise RuntimeError("INT8 calibration requires a known calibration shape")
            calibration_samples = load_calibration_samples(
                Path(args.calib_dir),
                calib_shape,
                args.max_calib_samples,
            )

        engine_path = out_dir / f"{model_path.stem}.{precision}.{'dynamic' if dynamic else 'static'}.plan"
        trt_artifacts = build_trt_engine(
            onnx_path=optimized_onnx_path,
            engine_path=engine_path,
            precision=precision,
            workspace_mb=workspace_mb,
            dynamic=dynamic,
            input_name=input_name,
            static_shape=static_input_shape,
            profile_min=profile_min,
            profile_opt=profile_opt,
            profile_max=profile_max,
            timing_cache_path=out_dir / args.timing_cache if args.timing_cache else None,
            calibration_samples=calibration_samples,
            calibration_cache_path=calibration_cache_path,
        )

        if args.benchmark:
            try:
                trt_artifacts.benchmark = benchmark_trt_engine(
                    engine_path=engine_path,
                    sample=sample,
                    warmup=args.warmup,
                    runs=args.runs,
                    dynamic=dynamic,
                )
                log(f"[TensorRT] Latency: {trt_artifacts.benchmark['latency_ms']:.4f} ms")
            except Exception as exc:
                trt_artifacts.benchmark = {"error": str(exc)}
                log(f"[TensorRT] Benchmark failed: {exc}")

        try:
            trt_outputs = run_trt_once(engine_path, sample, dynamic)

            if args.sample_output_index >= len(ref_outputs):
                raise IndexError(
                    f"sample_output_index={args.sample_output_index} out of range for ORT outputs={len(ref_outputs)}"
                )
            if args.sample_output_index >= len(trt_outputs):
                raise IndexError(
                    f"sample_output_index={args.sample_output_index} out of range for TRT outputs={len(trt_outputs)}"
                )

            validation["trt_vs_ort"] = compare_single_output(
                ref_outputs[args.sample_output_index],
                trt_outputs[args.sample_output_index],
                args.atol,
                args.rtol,
            )
            log(
                f"[VALIDATION] TRT vs ORT: "
                f"match={validation['trt_vs_ort']['matches']} "
                f"max_abs={validation['trt_vs_ort']['max_abs_error']:.6g} "
                f"mean_abs={validation['trt_vs_ort']['mean_abs_error']:.6g}"
            )
        except Exception as exc:
            validation["trt_vs_ort_error"] = str(exc)

    if args.sample_output:
        expected = load_sample_output(Path(args.sample_output))

        if args.sample_output_index >= len(ref_outputs):
            raise IndexError(
                f"sample_output_index={args.sample_output_index} out of range for ORT outputs={len(ref_outputs)}"
            )

        validation["sample_vs_ort"] = compare_single_output(
            expected,
            ref_outputs[args.sample_output_index],
            args.atol,
            args.rtol,
        )
        log(
            f"[VALIDATION] Sample vs ORT: "
            f"match={validation['sample_vs_ort']['matches']} "
            f"max_abs={validation['sample_vs_ort']['max_abs_error']:.6g} "
            f"mean_abs={validation['sample_vs_ort']['mean_abs_error']:.6g}"
        )

        if trt_artifacts is not None and trt_artifacts.engine_path is not None:
            try:
                trt_outputs = run_trt_once(Path(trt_artifacts.engine_path), sample, dynamic)

                if args.sample_output_index >= len(trt_outputs):
                    raise IndexError(
                        f"sample_output_index={args.sample_output_index} out of range for TRT outputs={len(trt_outputs)}"
                    )

                validation["sample_vs_trt"] = compare_single_output(
                    expected,
                    trt_outputs[args.sample_output_index],
                    args.atol,
                    args.rtol,
                )
                log(
                    f"[VALIDATION] Sample vs TRT: "
                    f"match={validation['sample_vs_trt']['matches']} "
                    f"max_abs={validation['sample_vs_trt']['max_abs_error']:.6g} "
                    f"mean_abs={validation['sample_vs_trt']['mean_abs_error']:.6g}"
                )
            except Exception as exc:
                validation["sample_vs_trt_error"] = str(exc)

    manifest = build_manifest(
        model_path=model_path,
        preset=args.preset,
        input_info={
            "name": input_name,
            "dtype": str(input_dtype),
            "inferred_shape": inferred_shape,
            "static_shape": static_input_shape,
            "dynamic": dynamic,
            "profile_min": profile_min,
            "profile_opt": profile_opt,
            "profile_max": profile_max,
            "sample_input": args.sample_input,
            "sample_output": args.sample_output,
            "sample_output_index": args.sample_output_index,
        },
        onnx_optimization=onnx_opt_info,
        optimized_onnx_path=optimized_onnx_path,
        ort_optimized_export_path=ort_optimized_export_path,
        trt_artifacts=trt_artifacts,
        ort_benchmarks=ort_benchmarks,
        accuracy=validation,
    )

    manifest_path = out_dir / DEFAULT_MANIFEST
    report_path = out_dir / DEFAULT_REPORT
    write_json(manifest_path, manifest)
    write_json(report_path, manifest["benchmarks"])

    log(f"[DONE] Manifest: {manifest_path}")
    log(f"[DONE] Benchmark report: {report_path}")
    return 0