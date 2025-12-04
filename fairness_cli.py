"""Run fairness-focused evaluation on FER-2013 models.

The CLI loads a checkpoint with :func:`build_model`, creates evaluation splits
with synthetic demographic groups, and renders group-level metrics through
:func:`plot_group_metrics`. Argument handling is defined in :func:`parse_args`.
"""

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import torch

from fer import EmotionCNN, FER2013Dataset, get_eval_transform
from fer.fairness_analysis import compute_age_groups, compute_confidence_groups, plot_group_metrics


def build_model(ckpt_path: str, in_chans: int, device: torch.device) -> EmotionCNN:
    model = EmotionCNN(in_chans=in_chans)
    try:
        state = torch.load(ckpt_path, map_location=device)
    except Exception as exc:  # pylint: disable=broad-except
        raise SystemExit(f"Failed to load checkpoint {ckpt_path}: {exc}") from exc

    if "state_dict" in state:
        state_dict = state["state_dict"]
    elif isinstance(state, dict):
        state_dict = state
    else:
        raise SystemExit(
            f"Checkpoint {ckpt_path} did not contain a state_dict."
            " Provide a file saved with torch.save(model.state_dict())."
        )

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise SystemExit(f"Checkpoint format is invalid: {exc}") from exc
    model.to(device)
    model.eval()
    return model


def build_loader(
    csv_path: str,
    usage: Optional[str],
    batch_size: int,
    num_workers: int,
    in_chans: int,
    return_group: bool = False,
):
    dataset = FER2013Dataset(
        csv_path,
        usage=usage,
        transform=get_eval_transform(in_chans),
        in_chans=in_chans,
        return_group=return_group,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def write_summary(conf_df: pd.DataFrame, age_df: pd.DataFrame, output_dir: Path) -> Path:
    lines = ["# Fairness summary", ""]

    for name, df in [("Confidence", conf_df), ("Age", age_df)]:
        if df.empty:
            lines.append(f"## {name}\nNo records available.\n")
            continue

        acc_gap = df["accuracy"].max() - df["accuracy"].min()
        f1_gap = df["f1"].max() - df["f1"].min()
        worst_acc_group = df.loc[df["accuracy"].idxmin(), "group"]
        best_acc_group = df.loc[df["accuracy"].idxmax(), "group"]
        lines.append(
            f"## {name}\n"
            f"Max accuracy gap: {acc_gap:.3f} ({best_acc_group} vs {worst_acc_group})\n"
            f"Max F1 gap: {f1_gap:.3f}\n"
        )

    output_path = output_dir / "fairness_summary.md"
    output_path.write_text("\n".join(lines))
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute fairness-oriented metrics and plots.")
    parser.add_argument("--csv", required=True, help="Path to fer2013.csv")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    parser.add_argument("--output-dir", required=True, help="Directory to store outputs")
    parser.add_argument("--usage", choices=["Training", "PublicTest", "PrivateTest", "all"], default="PrivateTest")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--in-chans", type=int, default=1, choices=[1, 3])
    parser.add_argument("--device", default=None, help="Override device string (e.g., cuda or cpu)")
    parser.add_argument("--no-summary", action="store_true", help="Skip writing the markdown summary")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without running analysis")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise SystemExit(f"FER-2013 CSV not found: {csv_path}; expected columns: emotion, pixels, Usage")

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    usage = None if args.usage == "all" else args.usage
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(str(ckpt_path), args.in_chans, device)
    if args.dry_run:
        print("Dry run successful: inputs and checkpoint format look valid.")
        return

    conf_loader = build_loader(
        args.csv,
        usage,
        batch_size=args.batch_size,
        num_workers=args.workers,
        in_chans=args.in_chans,
        return_group=False,
    )
    conf_df = compute_confidence_groups(model, conf_loader, device)
    conf_csv = output_dir / "confidence_metrics.csv"
    conf_df.to_csv(conf_csv, index=False)
    plot_group_metrics(conf_df, "accuracy", output_dir / "confidence_accuracy.png")

    age_loader = build_loader(
        args.csv,
        usage,
        batch_size=args.batch_size,
        num_workers=args.workers,
        in_chans=args.in_chans,
        return_group=True,
    )
    age_df = compute_age_groups(model, age_loader, device)
    age_csv = output_dir / "age_metrics.csv"
    age_df.to_csv(age_csv, index=False)
    plot_group_metrics(age_df, "accuracy", output_dir / "age_accuracy.png")

    if not args.no_summary:
        write_summary(conf_df, age_df, output_dir)


if __name__ == "__main__":
    main()
