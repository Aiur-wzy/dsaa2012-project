"""Knowledge distillation training routines for the FER model.

Includes the soft-label distillation loss, student/teacher training loop in
:func:`train_kd_model`, and logging helpers for experiment tracking.
"""

from argparse import Namespace
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .train import evaluate, save_checkpoint, save_process_file


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    T: float = 4.0,
    alpha: float = 0.5,
    ce_weight: float = 1.0,
) -> torch.Tensor:
    """
    Standard knowledge distillation loss that blends soft targets from the teacher
    with hard-label cross entropy.
    """

    ce = F.cross_entropy(student_logits, targets) * ce_weight
    log_p_s = F.log_softmax(student_logits / T, dim=1)
    p_t = F.softmax(teacher_logits / T, dim=1)
    kd = F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)
    return alpha * kd + (1 - alpha) * ce


def _extract_state_dict(state: Dict[str, torch.Tensor] | Dict[str, Dict]) -> Dict[str, torch.Tensor]:
    if "state_dict" in state:
        return state["state_dict"]
    return state


def train_kd_model(
    student: nn.Module,
    teacher: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    device: torch.device,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    T: float = 4.0,
    alpha: float = 0.5,
    ce_weight: float = 1.0,
    ckpt_dir: str | Path = "runs/kd",
    history_path: str | Path | None = None,
    process_path: str | Path | None = None,
    args: Optional[Dict | Namespace] = None,
    teacher_checkpoint: str | Path | None = None,
) -> Dict[str, float | str | Iterable[float]]:
    """
    Train a student model using knowledge distillation.

    The best checkpoint and a ``process.json`` summary are written to ``ckpt_dir``.
    """

    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history_path = Path(history_path) if history_path is not None else ckpt_dir / "history.csv"
    process_path = Path(process_path) if process_path is not None else ckpt_dir / "process.json"

    student.to(device)
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    if teacher_checkpoint is not None:
        state = torch.load(teacher_checkpoint, map_location=device)
        teacher.load_state_dict(_extract_state_dict(state))

    optimizer = optim.Adam(student.parameters(), lr=lr, weight_decay=weight_decay)
    ce_criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        student.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, targets in tqdm(train_loader, desc=f"kd train {epoch}", leave=False):
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            with torch.no_grad():
                teacher_logits = teacher(images)

            student_logits = student(images)
            loss = distillation_loss(student_logits, teacher_logits, targets, T=T, alpha=alpha, ce_weight=ce_weight)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = student_logits.argmax(dim=1)
            correct += preds.eq(targets).sum().item()
            total += images.size(0)

        train_loss = running_loss / total if total > 0 else 0.0
        train_acc = correct / total if total > 0 else 0.0

        val_loss, val_acc = evaluate(student, val_loader, ce_criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        save_checkpoint(student, optimizer, epoch, ckpt_dir / "latest.pt", val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(student, optimizer, epoch, ckpt_dir / "best.pt", val_acc)

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

    config = args if args is not None else {
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "T": T,
        "alpha": alpha,
        "ce_weight": ce_weight,
    }

    process_file = save_process_file(ckpt_dir, config, {"best_val_acc": best_val_acc, "history_path": str(history_path)}, process_path)

    return {
        "best_val_acc": best_val_acc,
        "history_path": str(history_path),
        "process_path": str(process_file),
        "train_loss": history["train_loss"],
        "train_acc": history["train_acc"],
        "val_loss": history["val_loss"],
        "val_acc": history["val_acc"],
    }
