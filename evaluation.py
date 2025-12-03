import argparse
import json
from pathlib import Path

import torch

from fer import (
    EMOTION_LABELS,
    EmotionCNN,
    build_dataloaders,
    evaluate,
    evaluate_with_confusion,
    get_eval_transform,
    save_confusion_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained EmotionCNN checkpoint")
    parser.add_argument("--csv", default="fer2013.csv", help="Path to fer2013.csv")
    parser.add_argument("--ckpt", default="runs/exp1/best.pt", help="Checkpoint path")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--in-chans", type=int, default=1, choices=[1, 3])
    parser.add_argument("--width-mult", type=float, default=1.0, help="Width multiplier for model ablations")
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
        help="Which split to evaluate (val=PublicTest, test=PrivateTest)",
    )
    parser.add_argument("--compute-confusion", action="store_true", help="Compute confusion matrix and per-class metrics")
    parser.add_argument(
        "--save-dir",
        default="runs/analysis",
        help="Directory to write confusion matrices, reports, and metric JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = EmotionCNN(in_chans=args.in_chans, width_mult=args.width_mult).to(device)
    state = torch.load(args.ckpt, map_location=device)
    state_dict = state["state_dict"] if "state_dict" in state else state
    model.load_state_dict(state_dict)

    eval_tf = get_eval_transform(args.in_chans)
    _train_loader, val_loader, test_loader = build_dataloaders(
        args.csv,
        batch_size=args.batch_size,
        num_workers=args.workers,
        in_chans=args.in_chans,
        train_transform=eval_tf,
        eval_transform=eval_tf,
    )

    loader = val_loader if args.split == "val" else test_loader
    criterion = torch.nn.CrossEntropyLoss()
    loss, acc = evaluate(model, loader, criterion, device)
    metrics = {"loss": loss, "accuracy": acc, "split": args.split}
    print("Evaluation:", metrics)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = save_dir / f"{args.split}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if args.compute_confusion:
        cm, report = evaluate_with_confusion(model, loader, device)
        label_names = [EMOTION_LABELS[idx] for idx in sorted(EMOTION_LABELS.keys())]
        outputs = save_confusion_outputs(cm, report, save_dir / f"{args.split}_confusion", labels=label_names)
        report_path = save_dir / f"{args.split}_classification_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print("Confusion matrix saved to", outputs)
        print("Classification report saved to", report_path)


if __name__ == "__main__":
    main()
