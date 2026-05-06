"""Command-line interface for training, evaluation, export, serving."""

import argparse
import os
import sys
import torch
from torch_geometric.loader import DataLoader
from .config import Config
from .sh_gnn import SHGNN
from .trainer import Trainer
from .datasets import ModelNet40Spherical, QM9Spherical, CMBSpherical
from .export import export_to_onnx
from .utils import set_seed


def main():
    parser = argparse.ArgumentParser(description="SH-GNN CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train command
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument(
        "--dataset", type=str, default="modelnet40", choices=["modelnet40", "qm9", "cmb"])
    train_parser.add_argument("--epochs", type=int, default=None)
    train_parser.add_argument("--batch_size", type=int, default=None)
    train_parser.add_argument("--l_max", type=int, default=None)
    train_parser.add_argument("--seed", type=int, default=42)

    # Export command
    export_parser = subparsers.add_parser(
        "export", help="Export model to ONNX")
    export_parser.add_argument("--checkpoint", type=str, required=True)
    export_parser.add_argument("--output", type=str, required=True)
    export_parser.add_argument("--opset", type=int, default=14)

    # Serve command
    serve_parser = subparsers.add_parser(
        "serve", help="Start inference server")
    serve_parser.add_argument("--checkpoint", type=str,
                              default="checkpoints/best.pt")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--host", type=str, default="0.0.0.0")

    args = parser.parse_args()

    if args.command == "train":
        set_seed(args.seed)
        cfg = Config()
        if args.epochs:
            cfg.num_epochs = args.epochs
        if args.batch_size:
            cfg.batch_size = args.batch_size
        if args.l_max:
            cfg.l_max = args.l_max

        # Create dataset
        if args.dataset == "modelnet40":
            train_dataset = ModelNet40Spherical(
                root=cfg.data_root, train=True, l_max=cfg.l_max)
            val_dataset = ModelNet40Spherical(
                root=cfg.data_root, train=False, l_max=cfg.l_max)
        elif args.dataset == "qm9":
            train_dataset = QM9Spherical(
                root=cfg.data_root, train=True, l_max=cfg.l_max)
            val_dataset = QM9Spherical(
                root=cfg.data_root, train=False, l_max=cfg.l_max)
        elif args.dataset == "cmb":
            train_dataset = CMBSpherical(
                root=cfg.data_root, train=True, l_max=cfg.l_max)
            val_dataset = CMBSpherical(
                root=cfg.data_root, train=False, l_max=cfg.l_max)
        else:
            raise ValueError(f"Unknown dataset {args.dataset}")

        train_loader = DataLoader(
            train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
        val_loader = DataLoader(
            val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

        model = SHGNN(cfg).to(cfg.device)
        trainer = Trainer(model, train_loader, val_loader, cfg)
        trainer.run()

    elif args.command == "export":
        cfg = Config()
        model = SHGNN(cfg)
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        N = 100
        E = 500
        in_features = cfg.in_features
        x = torch.randn(N, in_features)
        edge_index = torch.randint(0, N, (2, E))
        edge_attr = torch.rand(E, 3)
        export_to_onnx(model, (x, edge_index, edge_attr),
                       args.output, opset_version=args.opset)

    elif args.command == "serve":
        os.environ["SHGNN_CHECKPOINT"] = args.checkpoint
        import uvicorn
        from .serve import app
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
