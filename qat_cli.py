"""CLI for performing dynamic quantization or quantization-aware training.

Argument parsing occurs in :func:`parse_args`, and execution is driven by
:func:`main`, which dispatches to either dynamic quantization or QAT fine-tune
flows using the functions in :mod:`fer.quantization`.
"""

import argparse
from pathlib import Path

import torch
from torch import nn

from fer import EmotionCNN, build_dataloaders, get_eval_transform, get_train_transform
from fer.quantization import convert_qat_model, prepare_qat_model, quantize_dynamic
from fer.train import evaluate, train_one_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantization utilities for FER models")
    parser.add_argument("--csv", required=True, help="Path to fer2013.csv")
    parser.add_argument("--ckpt", required=True, help="Base checkpoint to quantize")
    parser.add_argument("--mode", choices=["dynamic", "qat"], default="qat")
    parser.add_argument("--output", default=None, help="Path to save the quantized checkpoint")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs for QAT fine-tuning")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--in-chans", type=int, default=1, choices=[1, 3])
    parser.add_argument("--width-mult", type=float, default=1.0)
    parser.add_argument("--device", default=None, help="Override device string")
    return parser.parse_args()


def load_base_model(ckpt_path: str, in_chans: int, width_mult: float, device: torch.device) -> EmotionCNN:
    model = EmotionCNN(in_chans=in_chans, width_mult=width_mult)
    state = torch.load(ckpt_path, map_location=device)
    state_dict = state["state_dict"] if "state_dict" in state else state
    model.load_state_dict(state_dict)
    model.to(device)
    return model


def run_dynamic_quantization(model: nn.Module, output: Path) -> None:
    quantized = quantize_dynamic(model.cpu())
    payload = {"state_dict": quantized.state_dict()}
    torch.save(payload, output)
    print("Dynamic quantized model saved to", output)


def run_qat(model: EmotionCNN, args: argparse.Namespace, device: torch.device) -> None:
    train_tf = get_train_transform(args.in_chans)
    eval_tf = get_eval_transform(args.in_chans)
    train_loader, val_loader, _ = build_dataloaders(
        args.csv,
        batch_size=args.batch_size,
        num_workers=args.workers,
        in_chans=args.in_chans,
        train_transform=train_tf,
        eval_transform=eval_tf,
    )

    qat_model = prepare_qat_model(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(qat_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(qat_model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(qat_model, val_loader, criterion, device)
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f}"
        )

    int8_model = convert_qat_model(qat_model)
    payload = {"state_dict": int8_model.state_dict()}
    torch.save(payload, Path(args.output))
    print("QAT INT8 model saved to", args.output)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(args.output) if args.output else Path(args.ckpt).with_name(f"{Path(args.ckpt).stem}_{args.mode}.pt")
    args.output = str(output)

    base_model = load_base_model(args.ckpt, args.in_chans, args.width_mult, device)

    if args.mode == "dynamic":
        run_dynamic_quantization(base_model, output)
    else:
        run_qat(base_model, args, device)


if __name__ == "__main__":
    main()
