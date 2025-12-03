from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .augment import mixup_criterion, mixup_data
from .losses import LabelSmoothingCE


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    use_mixup: bool = False,
    mixup_alpha: float = 0.2,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        if use_mixup:
            images, targets_a, targets_b, lam = mixup_data(images, targets, alpha=mixup_alpha)
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            preds = outputs.argmax(dim=1)
            correct += (lam * preds.eq(targets_a).sum() + (1 - lam) * preds.eq(targets_b).sum()).item()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            preds = outputs.argmax(dim=1)
            correct += preds.eq(targets).sum().item()

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total += images.size(0)

    return running_loss / total, correct / total


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="eval", leave=False):
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            preds = outputs.argmax(dim=1)
            running_loss += loss.item() * images.size(0)
            correct += preds.eq(targets).sum().item()
            total += images.size(0)

    return running_loss / total, correct / total


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    device: torch.device,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    use_mixup: bool = False,
    mixup_alpha: float = 0.2,
    ckpt_dir: str | Path = "runs",
    history_path: str | Path | None = None,
    loss: str = "ce",
    label_smoothing_eps: float = 0.1,
    num_classes: int = 7,
) -> Dict[str, float]:
    if loss == "ce":
        criterion: nn.Module = nn.CrossEntropyLoss()
    elif loss == "label_smoothing":
        criterion = LabelSmoothingCE(num_classes=num_classes, eps=label_smoothing_eps)
    else:
        raise ValueError(f"Unsupported loss type: {loss}")
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history_path = Path(history_path) if history_path is not None else ckpt_dir / "history.csv"

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, use_mixup=use_mixup, mixup_alpha=mixup_alpha
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        save_checkpoint(model, optimizer, epoch, ckpt_dir / "latest.pt", val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, ckpt_dir / "best.pt", val_acc)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f}"
        )

    history_df = pd.DataFrame(
        {
            "epoch": list(range(1, epochs + 1)),
            "train_loss": history["train_loss"],
            "train_acc": history["train_acc"],
            "val_loss": history["val_loss"],
            "val_acc": history["val_acc"],
        }
    )
    history_df.to_csv(history_path, index=False)

    return {"best_val_acc": best_val_acc, "history_path": str(history_path), **history}


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int, path: str | Path, metric: float) -> None:
    payload = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metric": metric,
    }
    torch.save(payload, path)


def load_checkpoint(model: nn.Module, path: str | Path, device: torch.device) -> Dict:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    return checkpoint


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a lightweight FER-2013 model")
    parser.add_argument("--csv", required=True, help="Path to fer2013.csv")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mixup", action="store_true")
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--in-chans", type=int, default=1, choices=[1, 3])
    parser.add_argument("--width-mult", type=float, default=1.0, help="Width multiplier for model capacity ablations")
    parser.add_argument(
        "--augmentation",
        choices=["full", "baseline"],
        default="full",
        help="Choose augmentation strength for ablation studies",
    )
    parser.add_argument("--loss", choices=["ce", "label_smoothing"], default="ce", help="Training loss")
    parser.add_argument("--label-smoothing-eps", type=float, default=0.1, help="Smoothing factor for label smoothing")
    parser.add_argument("--num-classes", type=int, default=7, help="Number of target classes")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ckpt-dir", default="runs/exp1")
    args = parser.parse_args()

    from .augment import get_baseline_train_transform, get_eval_transform, get_train_transform
    from .data import build_dataloaders
    from .models import EmotionCNN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.augmentation == "baseline":
        train_tf = get_baseline_train_transform(args.in_chans)
    else:
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

    model = EmotionCNN(in_chans=args.in_chans, width_mult=args.width_mult).to(device)
    stats = train_model(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_mixup=args.mixup,
        mixup_alpha=args.mixup_alpha,
        ckpt_dir=args.ckpt_dir,
        loss=args.loss,
        label_smoothing_eps=args.label_smoothing_eps,
        num_classes=args.num_classes,
    )
    print("Training finished. Best val acc:", stats["best_val_acc"])
