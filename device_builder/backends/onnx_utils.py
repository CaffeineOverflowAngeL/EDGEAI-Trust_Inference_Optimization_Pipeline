from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import onnx
from onnx import shape_inference

try:
    import onnxsim  # type: ignore
    ONNXSIM_AVAILABLE = True
except Exception:
    ONNXSIM_AVAILABLE = False


def load_onnx_model(model_path: Path) -> onnx.ModelProto:
    model = onnx.load(str(model_path))
    onnx.checker.check_model(model)
    return model


def infer_primary_input_info(model: onnx.ModelProto) -> Tuple[str, List[int], np.dtype]:
    init_names = {init.name for init in model.graph.initializer}
    for value_info in model.graph.input:
        if value_info.name in init_names:
            continue
        tensor_type = value_info.type.tensor_type
        shape: List[int] = []
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                shape.append(int(dim.dim_value))
            else:
                shape.append(-1)
        np_type = onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type)
        return value_info.name, shape, np.dtype(np_type)
    raise RuntimeError("Could not find a non-initializer graph input")


def optimize_onnx_model(model: onnx.ModelProto, try_simplify: bool = True) -> Tuple[onnx.ModelProto, Dict[str, Any]]:
    info: Dict[str, Any] = {
        "shape_inference": False,
        "onnxsim": False,
        "onnxsim_error": None,
    }
    optimized = model
    try:
        optimized = shape_inference.infer_shapes(optimized)
        info["shape_inference"] = True
    except Exception as exc:
        info["shape_inference_error"] = str(exc)

    if try_simplify and ONNXSIM_AVAILABLE:
        try:
            optimized, ok = onnxsim.simplify(optimized)
            info["onnxsim"] = bool(ok)
        except Exception as exc:
            info["onnxsim_error"] = str(exc)
    elif try_simplify:
        info["onnxsim_error"] = "onnxsim not installed"

    onnx.checker.check_model(optimized)
    return optimized, info


def save_onnx(model: onnx.ModelProto, path: Path) -> None:
    onnx.save(model, str(path))