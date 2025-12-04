"""Model pruning helpers for compressing the FER CNN.

Functions here apply structured pruning, fine-tune pruned checkpoints, and
export sparsified weights for later evaluation.
"""

import io
from pathlib import Path
from typing import Dict

import torch
from torch import nn, optim
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader

from .train import evaluate, load_checkpoint, save_checkpoint, train_one_epoch


def structured_prune_model(model: nn.Module, amount: float = 0.3) -> nn.Module:
    """Apply L1-norm structured pruning to all Conv2d layers."""

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name="weight", amount=amount, n=1, dim=0)
            prune.remove(module, "weight")
    return model


def _state_dict_size_mb(model: nn.Module) -> float:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell() / (1024 ** 2)


def prune_and_finetune(
    model: nn.Module,
    checkpoint_path: str | Path,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    amount: float = 0.3,
    finetune_epochs: int = 5,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    ckpt_dir: str | Path | None = None,
) -> Dict[str, float | str]:
    """
    Load a checkpoint, apply structured pruning, and fine-tune the model.
    Returns a summary containing size and validation accuracy.
    """

    ckpt_dir = Path(ckpt_dir) if ckpt_dir is not None else Path(checkpoint_path).parent / "pruned"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device)
    model.to(device)
    load_checkpoint(model, checkpoint_path, device)

    model = structured_prune_model(model, amount=amount)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = 0.0
    for epoch in range(1, finetune_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        save_checkpoint(model, optimizer, epoch, ckpt_dir / "latest.pt", val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, ckpt_dir / "best.pt", val_acc)

        print(
            f"Epoch {epoch}/{finetune_epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f}"
        )

    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    size_mb = _state_dict_size_mb(model)

    summary: Dict[str, float | str] = {
        "best_val_acc": best_val_acc,
        "final_val_acc": val_acc,
        "final_val_loss": val_loss,
        "model_size_mb": size_mb,
        "checkpoint_dir": str(ckpt_dir),
    }
    return summary
