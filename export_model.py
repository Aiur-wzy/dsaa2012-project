"""Export and optionally quantize the FER EmotionCNN model.

This script traces a TorchScript module and exports an ONNX graph so the
lightweight classifier can be used in environments without PyTorch. It also
offers a simple dynamic quantization path for size/latency comparisons.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Tuple

import torch
from torch import nn

from fer.augment import get_eval_transform
from fer.data import FER2013Dataset
from fer.models import EmotionCNN
from fer.train import load_checkpoint


def _model_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def build_model(checkpoint: Path, device: torch.device, in_chans: int, num_classes: int, width_mult: float) -> nn.Module:
    model = EmotionCNN(in_chans=in_chans, num_classes=num_classes, width_mult=width_mult)
    load_checkpoint(model, checkpoint, device)
    model.to(device)
    model.eval()
    return model


def benchmark_model(model: nn.Module, device: torch.device, in_chans: int, runs: int = 50) -> float:
    dummy = torch.randn(1, in_chans, 48, 48, device=device)
    with torch.inference_mode():
        for _ in range(5):  # warmup
            _ = model(dummy)
        start = time.perf_counter()
        for _ in range(runs):
            _ = model(dummy)
        elapsed = time.perf_counter() - start
    return (elapsed / runs) * 1000  # ms per run


def export_torchscript(model: nn.Module, dummy: torch.Tensor, output: Path) -> None:
    scripted = torch.jit.trace(model, dummy)
    scripted.save(output)


def export_onnx(model: nn.Module, dummy: torch.Tensor, output: Path) -> None:
    torch.onnx.export(
        model,
        dummy,
        output,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=12,
    )


def evaluate_subset(
    model: nn.Module,
    csv_path: Path,
    in_chans: int,
    device: torch.device,
    samples: int = 256,
    batch_size: int = 32,
) -> Tuple[int, int]:
    transform = get_eval_transform(in_chans)
    dataset = FER2013Dataset(csv_path, usage="PrivateTest", transform=transform, in_chans=in_chans)
    subset_size = min(samples, len(dataset))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0
    with torch.inference_mode():
        for images, labels in loader:
            if total >= subset_size:
                break
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct, min(total, subset_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export EmotionCNN to TorchScript/ONNX and test quantization.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to trained checkpoint (e.g., runs/exp1/best.pt)")
    parser.add_argument("--output-dir", type=Path, default=Path("exports"), help="Destination directory")
    parser.add_argument("--in-chans", type=int, default=1, choices=[1, 3], help="Model input channels")
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--width-mult", type=float, default=1.0, help="Width multiplier used during training")
    parser.add_argument("--quantize", action="store_true", help="Also export a dynamically quantized TorchScript model")
    parser.add_argument("--evaluate-csv", type=Path, help="Optional FER2013 CSV to estimate accuracy drop")
    parser.add_argument("--device", default="cpu", help="Export device (default cpu; use cuda if available)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args.checkpoint, device, args.in_chans, args.num_classes, args.width_mult)
    dummy = torch.randn(1, args.in_chans, 48, 48, device=device)

    ts_path = args.output_dir / "model_scripted.pt"
    onnx_path = args.output_dir / "model.onnx"
    export_torchscript(model, dummy, ts_path)
    export_onnx(model, dummy, onnx_path)

    print(f"TorchScript saved to {ts_path} ({_model_size_mb(ts_path):.2f} MB)")
    print(f"ONNX saved to {onnx_path} ({_model_size_mb(onnx_path):.2f} MB)")
    base_latency = benchmark_model(model, device, args.in_chans)
    print(f"FP32 latency: {base_latency:.2f} ms per inference")

    if args.quantize:
        quantized = torch.quantization.quantize_dynamic(model.cpu(), {nn.Linear}, dtype=torch.qint8)
        q_dummy = dummy.cpu()
        q_path = args.output_dir / "model_quantized.pt"
        export_torchscript(quantized, q_dummy, q_path)
        print(f"Quantized TorchScript saved to {q_path} ({_model_size_mb(q_path):.2f} MB)")
        q_latency = benchmark_model(quantized, torch.device("cpu"), args.in_chans)
        print(f"Quantized latency: {q_latency:.2f} ms per inference")

        if args.evaluate_csv:
            correct_fp32, total = evaluate_subset(model, args.evaluate_csv, args.in_chans, device)
            correct_q, _ = evaluate_subset(quantized, args.evaluate_csv, args.in_chans, torch.device("cpu"))
            print(
                f"Evaluation on subset ({total} samples) -> FP32 acc: {correct_fp32 / total:.3f}, "
                f"Quantized acc: {correct_q / total:.3f}"
            )


if __name__ == "__main__":
    main()
