"""Evaluation helpers for richer experiment analysis."""

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def evaluate_with_confusion(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
    """Return confusion matrix and classification report for a dataloader."""

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="analysis", leave=False):
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    preds_np = torch.cat(all_preds).numpy()
    targets_np = torch.cat(all_targets).numpy()

    cm = confusion_matrix(targets_np, preds_np)
    report = classification_report(targets_np, preds_np, output_dict=True)
    return cm, report


def _label_frame(matrix: np.ndarray, labels: Optional[Sequence[str]]) -> pd.DataFrame:
    if labels is None:
        labels = [str(i) for i in range(matrix.shape[0])]
    return pd.DataFrame(matrix, index=labels, columns=labels)


def save_confusion_outputs(
    cm: np.ndarray,
    report: Dict[str, Dict[str, float]],
    output_dir: str | Path,
    labels: Optional[Iterable[str]] = None,
) -> Dict[str, Path]:
    """Persist confusion matrix and classification report to disk.

    Returns a dictionary with paths to the saved files for convenient logging.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cm_path = output_dir / "confusion_matrix.npy"
    cm_csv_path = output_dir / "confusion_matrix.csv"
    report_path = output_dir / "classification_report.csv"

    np.save(cm_path, cm)
    cm_df = _label_frame(cm, labels)
    cm_df.to_csv(cm_csv_path)

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(report_path)

    return {"cm": cm_path, "cm_csv": cm_csv_path, "report_csv": report_path}
