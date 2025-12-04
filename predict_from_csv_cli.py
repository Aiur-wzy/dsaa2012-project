"""Batch predictions from a FER-2013 CSV file."""
import argparse
import torch

from fer import EmotionCNN, predict_from_fer2013_csv


def build_model(ckpt_path: str, in_chans: int) -> EmotionCNN:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionCNN(in_chans=in_chans)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["state_dict"])
    model.to(device)
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch predictions from a FER-2013 CSV.")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint.")
    parser.add_argument("--csv", required=True, help="Path to FER-2013 CSV file.")
    parser.add_argument("--usage", choices=["Training", "PublicTest", "PrivateTest", "all"], default="all", help="Usage split to filter or 'all'.")
    parser.add_argument("--in-chans", type=int, default=1, help="Input channels used during training (1 for grayscale).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = build_model(args.ckpt, args.in_chans)
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
