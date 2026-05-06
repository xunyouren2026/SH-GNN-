"""Model export to ONNX and TensorRT for production inference."""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
from .sh_gnn import SHGNN
from .config import Config


def export_to_onnx(
    model: SHGNN,
    input_example: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    onnx_path: str,
    opset_version: int = 14,
    input_names: List[str] = ["x", "edge_index", "edge_attr"],
    output_names: List[str] = ["task_out", "alms"],
    dynamic_axes: Optional[dict] = None,
):
    model.eval()
    x, edge_index, edge_attr = input_example
    edge_index = edge_index.long()

    with torch.no_grad():
        task_out, alms, _ = model.forward(
            type("Data", (), {"x": x, "edge_index": edge_index,
                 "edge_attr": edge_attr})(),
            return_phys_loss=False,
        )

    if dynamic_axes is None:
        dynamic_axes = {
            "x": {0: "num_nodes"},
            "edge_index": {1: "num_edges"},
            "edge_attr": {0: "num_edges"},
            "task_out": {0: "num_nodes"},
            "alms": {0: "num_nodes"},
        }

    torch.onnx.export(
        model,
        (x, edge_index, edge_attr),
        onnx_path,
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        verbose=False,
    )
    print(f"ONNX model exported to {onnx_path}")


def export_to_tensorrt(
    onnx_path: str,
    engine_path: str,
    precision: str = "fp16",
    max_workspace_size: int = 1 << 30,
    max_batch_size: int = 32,
):
    try:
        import tensorrt as trt
    except ImportError:
        raise ImportError("TensorRT not installed. Run: pip install tensorrt")

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX model")

    config = builder.create_builder_config()
    config.max_workspace_size = max_workspace_size
    if precision == "fp16" and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    profile.set_shape(input_tensor.name, (1,),
                      (max_batch_size,), (max_batch_size,))
    config.add_optimization_profile(profile)

    engine = builder.build_engine(network, config)
    with open(engine_path, "wb") as f:
        f.write(engine.serialize())
    print(f"TensorRT engine saved to {engine_path}")


def inference_onnx(onnx_path: str, x: np.ndarray, edge_index: np.ndarray, edge_attr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    import onnxruntime as ort
    session = ort.InferenceSession(onnx_path)
    inputs = {
        "x": x.astype(np.float32),
        "edge_index": edge_index.astype(np.int64),
        "edge_attr": edge_attr.astype(np.float32),
    }
    outputs = session.run(["task_out", "alms"], inputs)
    return outputs[0], outputs[1]
