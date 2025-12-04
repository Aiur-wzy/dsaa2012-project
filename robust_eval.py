"""Evaluate robustness of a trained FER model under common corruptions.

The CLI builds dataloaders, applies configurable corruption functions, and
summarizes results through :func:`eval_under_corruption`. Argument parsing is
handled in :func:`parse_args` and execution starts in :func:`main`.
"""

import argparse
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch

from fer import (
    EmotionCNN,
    add_brightness_contrast,
    add_gaussian_blur,
    build_dataloaders,
    eval_under_corruption,
    get_eval_transform,
    jpeg_compress,
    random_rotate,
)


def load_model(ckpt: str, device: torch.device, in_chans: int, width_mult: float) -> EmotionCNN:
    model = EmotionCNN(in_chans=in_chans, width_mult=width_mult).to(device)
    try:
        state = torch.load(ckpt, map_location=device)
    except Exception as exc:  # pylint: disable=broad-except
        raise SystemExit(f"Failed to load checkpoint {ckpt}: {exc}") from exc

    if "state_dict" in state:
        state_dict = state["state_dict"]
    elif isinstance(state, dict):
        state_dict = state
    else:
        raise SystemExit(
            f"Checkpoint {ckpt} did not contain a state_dict."
            " Provide a file saved with torch.save(model.state_dict())."
        )

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise SystemExit(f"Checkpoint format is invalid: {exc}") from exc
    return model


def build_corruptions() -> Dict[str, List[Tuple[str, Callable]]]:
    """Return a mapping of corruption name -> list of (severity label, fn)."""

    brightness_levels = [
        ("alpha0.8_beta-20", lambda img: add_brightness_contrast(img, alpha=0.8, beta=-20)),
        ("alpha1.0_beta0", lambda img: add_brightness_contrast(img, alpha=1.0, beta=0)),
        ("alpha1.2_beta20", lambda img: add_brightness_contrast(img, alpha=1.2, beta=20)),
        ("alpha1.4_beta40", lambda img: add_brightness_contrast(img, alpha=1.4, beta=40)),
    ]

    blur_levels = [
        ("k3", lambda img: add_gaussian_blur(img, ksize=3)),
        ("k5", lambda img: add_gaussian_blur(img, ksize=5)),
        ("k7", lambda img: add_gaussian_blur(img, ksize=7)),
    ]

    jpeg_levels = [
        ("q90", lambda img: jpeg_compress(img, quality=90)),
        ("q70", lambda img: jpeg_compress(img, quality=70)),
        ("q50", lambda img: jpeg_compress(img, quality=50)),
        ("q30", lambda img: jpeg_compress(img, quality=30)),
    ]

    rotation_levels = [
        ("max5", lambda img: random_rotate(img, max_angle=5)),
        ("max15", lambda img: random_rotate(img, max_angle=15)),
        ("max30", lambda img: random_rotate(img, max_angle=30)),
    ]

    return {
        "brightness/contrast": brightness_levels,
        "blur": blur_levels,
        "jpeg": jpeg_levels,
        "rotation": rotation_levels,
    }


def evaluate_corruptions(model, loader, device: torch.device) -> pd.DataFrame:
    rows = []
    corruptions = build_corruptions()

    for name, levels in corruptions.items():
        for severity, fn in levels:
            acc = eval_under_corruption(model, loader, device, fn)
            rows.append({"corruption": name, "severity": severity, "accuracy": acc})
            print(f"{name} severity {severity}: acc={acc:.3f}")

    return pd.DataFrame(rows)


def plot_results(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, group in df.groupby("corruption"):
        ax.plot(group["severity"], group["accuracy"], marker="o", label=name)

    ax.set_xlabel("Severity")
    ax.set_ylabel("Accuracy")
    ax.set_title("Robustness under Corruptions")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    plot_path = output_dir / "robustness_plot.png"
    fig.savefig(plot_path)
    print("Saved plot to", plot_path)


def write_summary(clean_acc: float, df: pd.DataFrame, output_dir: Path) -> None:
    lines = ["# Robustness Summary", ""]
    lines.append(f"Clean accuracy: **{clean_acc:.3f}**.")

    worst = df.loc[df["accuracy"].idxmin()]
    lines.append(
        f"Worst corruption: **{worst['corruption']}** at severity **{worst['severity']}** "
        f"with accuracy {worst['accuracy']:.3f}."
    )

    rotation_df = df[df["corruption"] == "rotation"].copy()
    if not rotation_df.empty:
        rotation_df = rotation_df.sort_values("severity")
        drops = clean_acc - rotation_df["accuracy"]
        lines.append(
            f"Rotation robustness: drop of {drops.iloc[0]:.3f} at {rotation_df['severity'].iloc[0]}, "
            f"{drops.iloc[-1]:.3f} at {rotation_df['severity'].iloc[-1]}."
        )

    lines.append(
        "If you trained a MixUp variant, compare these numbers to the baseline run from Part 1 "
        "to quantify robustness gains."
    )

    summary_path = output_dir / "robustness_summary.md"
    summary_path.write_text("\n".join(lines))
    print("Wrote summary to", summary_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate corruption robustness of a trained FER model")
    parser.add_argument("--csv", default="fer2013.csv", help="Path to fer2013.csv")
    parser.add_argument("--ckpt", default="runs/exp1/best.pt", help="Checkpoint path")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--in-chans", type=int, default=1, choices=[1, 3])
    parser.add_argument("--width-mult", type=float, default=1.0, help="Width multiplier for model ablations")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output-dir", default="runs/robustness", help="Directory to save plots and tables")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without running evaluation")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise SystemExit(f"FER-2013 CSV not found: {csv_path}; expected columns: emotion, pixels, Usage")

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = load_model(args.ckpt, device, args.in_chans, args.width_mult)
    if args.dry_run:
        print("Dry run successful: inputs and checkpoint format look valid.")
        return
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

    # Clean accuracy for reference
    clean_acc = eval_under_corruption(model, loader, device, lambda img: img)
    print(f"Clean accuracy ({args.split}): {clean_acc:.3f}")

    df = evaluate_corruptions(model, loader, device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"robustness_{args.split}.csv"
    df.to_csv(csv_path, index=False)
    print("Saved table to", csv_path)

    plot_results(df, output_dir)
    write_summary(clean_acc, df, output_dir)


if __name__ == "__main__":
    main()
