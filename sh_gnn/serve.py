"""FastAPI server for SH-GNN inference."""

import torch
import numpy as np
import os
import time
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager

from .sh_gnn import SHGNN
from .config import Config

_model = None
_onnx_session = None
_cfg = None


class NodeFeatures(BaseModel):
    x: List[List[float]]
    edge_index: List[List[int]]
    edge_attr: List[List[float]]


class PredictRequest(BaseModel):
    nodes: NodeFeatures
    return_alms: bool = False


class PredictResponse(BaseModel):
    task_out: List[List[float]]
    alms: Optional[List[List[float]]] = None
    inference_time_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _onnx_session, _cfg
    _cfg = Config()
    onnx_path = os.getenv("SHGNN_ONNX_PATH", "")
    if onnx_path and os.path.exists(onnx_path):
        import onnxruntime as ort
        _onnx_session = ort.InferenceSession(onnx_path)
        print(f"Loaded ONNX model from {onnx_path}")
    else:
        _model = SHGNN(_cfg)
        ckpt = torch.load(os.getenv("SHGNN_CHECKPOINT", "checkpoints/best.pt"),
                          map_location=_cfg.device, weights_only=False)
        _model.load_state_dict(ckpt["model_state_dict"])
        _model.to(_cfg.device)
        _model.eval()
        print(f"Loaded PyTorch model from checkpoint")
    yield
    print("Shutting down...")

app = FastAPI(title="SH-GNN Inference API", lifespan=lifespan)


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    start = time.time()
    x = np.array(req.nodes.x, dtype=np.float32)
    edge_index = np.array(req.nodes.edge_index, dtype=np.int64)
    edge_attr = np.array(req.nodes.edge_attr, dtype=np.float32)

    if _onnx_session is not None:
        task_out, alms = _onnx_session.run(
            ["task_out", "alms"],
            {"x": x, "edge_index": edge_index, "edge_attr": edge_attr}
        )
    else:
        with torch.no_grad():
            data = type('Data', (), {
                'x': torch.tensor(x, device=_cfg.device),
                'edge_index': torch.tensor(edge_index, device=_cfg.device),
                'edge_attr': torch.tensor(edge_attr, device=_cfg.device),
            })()
            task_out_t, alms_t, _ = _model(data, return_phys_loss=False)
            task_out = task_out_t.cpu().numpy()
            alms = alms_t.cpu().numpy()

    elapsed = (time.time() - start) * 1000.0
    return PredictResponse(
        task_out=task_out.tolist(),
        alms=alms.tolist() if req.return_alms else None,
        inference_time_ms=elapsed,
    )


@app.get("/health")
async def health():
    return {"status": "ok", "onnx": _onnx_session is not None, "torch": _model is not None}
