import tvm
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx
from tvm.runtime._tensor import tensor
import numpy as np
print(tvm.__version__)
print(tvm.target.Target.list_kinds())

ctx = tvm.device("cpu")  # or tvm.device("cuda")
data = np.random.rand(1,12,128,256).astype("float32")
tvm_data = tensor(data, ctx)
print(type(tvm_data))

print("TVM path:", tvm.__file__)
print("TVM version:", tvm.__version__)
print(relax.transform)
print("PlanDevices in relax.transform:", hasattr(relax.transform, "PlanDevices"))
print("BindTarget in relax.transform:", hasattr(relax.transform, "BindTarget"))