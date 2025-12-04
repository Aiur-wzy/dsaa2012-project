"""Batch prediction entry point for FER-2013 CSV files.

This script loads a saved :class:`EmotionCNN` checkpoint, constructs the
corresponding model with the desired input channels, parses CLI arguments, and
delegates batched inference to :func:`fer.predict_from_fer2013_csv`.
"""
import argparse
from pathlib import Path
import torch

from fer import EmotionCNN, predict_from_fer2013_csv


def build_model(ckpt_path: str, in_chans: int) -> EmotionCNN:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch predictions from a FER-2013 CSV.")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint.")
    parser.add_argument("--csv", required=True, help="Path to FER-2013 CSV file.")
    parser.add_argument("--usage", choices=["Training", "PublicTest", "PrivateTest", "all"], default="all", help="Usage split to filter or 'all'.")
    parser.add_argument("--in-chans", type=int, default=1, help="Input channels used during training (1 for grayscale).")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without running predictions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise SystemExit(f"FER-2013 CSV not found: {csv_path}; expected columns: emotion, pixels, Usage")

    model = build_model(args.ckpt, args.in_chans)
    if args.dry_run:
        print("Dry run successful: inputs and checkpoint format look valid.")
        return

    usage = None if args.usage == "all" else args.usage
    preds = predict_from_fer2013_csv(
        model,
        args.csv,
        device=next(model.parameters()).device,
        in_chans=args.in_chans,
        usage=usage,
    )
    for i, pred in enumerate(preds):
        print(f"Row {i}: {pred['label_name']} (p={pred['confidence']:.3f})")


if __name__ == "__main__":
    main()
