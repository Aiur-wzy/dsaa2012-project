"""Training and evaluation loops for the FER models.

Key entry points include :func:`train_one_epoch`, :func:`evaluate`, and
:func:`train_model`, along with checkpoint utilities (:func:`save_checkpoint`,
:func:`load_checkpoint`) used by CLI scripts and notebooks.
"""

import json
from argparse import Namespace
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

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


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    return_details: bool = False,
    group_fn: Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], str]] = None,
) -> Tuple[float, float] | Tuple[float, float, Dict[str, Iterable[int] | List[str]]]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    collect_details = return_details or group_fn is not None
    all_preds: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    all_groups: List[str] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", leave=False):
            if len(batch) == 3:
                images, targets, batch_groups = batch
            else:
                images, targets = batch
                batch_groups = None

            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            loss = criterion(outputs, targets)
            preds = outputs.argmax(dim=1)
            running_loss += loss.item() * images.size(0)
            correct += preds.eq(targets).sum().item()
            total += images.size(0)

            if collect_details or batch_groups is not None:
                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())

                if group_fn is not None:
                    batch_group_labels = [
                        group_fn(prob_vec, pred_lbl, true_lbl)
                        for prob_vec, pred_lbl, true_lbl in zip(probs, preds, targets)
                    ]
                elif batch_groups is not None:
                    batch_group_labels = list(batch_groups)
                else:
                    batch_group_labels = []

                all_groups.extend(batch_group_labels)

    avg_loss = running_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0

    if not (collect_details or all_groups):
        return avg_loss, acc

    details: Dict[str, Iterable[int] | List[str]] = {
        "y_true": torch.cat(all_targets).tolist() if all_targets else [],
        "y_pred": torch.cat(all_preds).tolist() if all_preds else [],
        "groups": all_groups,
    }
    return avg_loss, acc, details


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


def save_process_file(
    ckpt_dir: Path, args: Dict | Namespace, stats: Dict[str, float], process_path: Path | None = None
) -> Path:
    """Persist a JSON summary of the training run for downstream analysis."""

    process_path = process_path or ckpt_dir / "process.json"
    if hasattr(args, "__dict__"):
        config = vars(args)
    else:
        config = dict(args)

    payload = {
        "config": config,
        "history_path": stats.get("history_path", str(ckpt_dir / "history.csv")),
        "best_checkpoint": str(ckpt_dir / "best.pt"),
        "latest_checkpoint": str(ckpt_dir / "latest.pt"),
        "metrics": {"best_val_acc": stats.get("best_val_acc", 0.0)},
    }

    process_path = Path(process_path)
    process_path.parent.mkdir(parents=True, exist_ok=True)
    process_path.write_text(json.dumps(payload, indent=2))
    return process_path


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int, path: str | Path, metric: float) -> None:
    payload = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metric": metric,
    }
    torch.save(payload, path)


def load_checkpoint(model: nn.Module, path: str | Path, device: torch.device) -> Dict:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as exc:  # pylint: disable=broad-except
        raise SystemExit(f"Failed to load checkpoint {checkpoint_path}: {exc}") from exc

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise SystemExit(
            f"Checkpoint {checkpoint_path} did not contain a state_dict."
            " Provide a file saved with torch.save(model.state_dict())."
        )

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise SystemExit(f"Checkpoint format is invalid: {exc}") from exc
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
    parser.add_argument("--history-path", default=None, help="Optional path for the per-epoch history CSV")
    parser.add_argument(
        "--process-path", default=None, help="Where to write the process summary JSON for downstream analysis"
    )
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
        history_path=args.history_path,
    )
    process_file = save_process_file(Path(args.ckpt_dir), args, stats, args.process_path)
    print("Training finished. Best val acc:", stats["best_val_acc"])
    print("Process summary written to:", process_file)
