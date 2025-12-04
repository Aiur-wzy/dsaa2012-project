import argparse
from pathlib import Path

import torch

from fer import EmotionCNN, build_dataloaders, get_eval_transform, get_train_transform
from fer.pruning import prune_and_finetune


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Structured pruning followed by fine-tuning")
    parser.add_argument("--csv", required=True, help="Path to fer2013.csv")
    parser.add_argument("--ckpt", required=True, help="Checkpoint to prune")
    parser.add_argument("--output-dir", default=None, help="Directory to save pruned checkpoints")
    parser.add_argument("--prune-amount", type=float, default=0.3, help="Fraction of channels to prune")
    parser.add_argument("--finetune-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--in-chans", type=int, default=1, choices=[1, 3])
    parser.add_argument("--width-mult", type=float, default=1.0)
    parser.add_argument("--device", default=None, help="Override device string")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    model = EmotionCNN(in_chans=args.in_chans, width_mult=args.width_mult)
    output_dir = args.output_dir or (Path(args.ckpt).parent / "pruned")

    stats = prune_and_finetune(
        model,
        args.ckpt,
        train_loader,
        val_loader,
        device,
        amount=args.prune_amount,
        finetune_epochs=args.finetune_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        ckpt_dir=output_dir,
    )

    print("Pruning complete. Best val acc:", stats["best_val_acc"])
    print("Final val acc:", stats["final_val_acc"])
    print("Model size (MB):", f"{stats['model_size_mb']:.2f}")
    print("Checkpoints saved to:", stats["checkpoint_dir"])


if __name__ == "__main__":
    main()
