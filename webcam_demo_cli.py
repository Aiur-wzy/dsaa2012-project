"""Command-line entry point for the real-time webcam demo."""
import argparse
from pathlib import Path

import torch

from fer import EmotionCNN, FaceDetector
from fer.inference import run_realtime_demo


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
    parser = argparse.ArgumentParser(description="Run the FER-2013 webcam demo.")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint.")
    parser.add_argument("--in-chans", type=int, default=1, help="Input channels used during training (1 for grayscale).")
    parser.add_argument("--detector", choices=["haar", "dnn"], default="haar", help="Face detector backend.")
    parser.add_argument("--dry-run", action="store_true", help="Validate checkpoint and exit without running the demo.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(str(ckpt_path), args.in_chans)
    if args.dry_run:
        print("Dry run successful: checkpoint format looks valid.")
        return
    detector = FaceDetector(detector_type=args.detector)
    run_realtime_demo(model, detector, device=device, in_chans=args.in_chans)


if __name__ == "__main__":
    main()
