"""Utilities for fairness-oriented evaluation and visualization.

The module defines helpers such as :func:`compute_age_groups`,
:func:`compute_confidence_groups`, and :func:`plot_group_metrics` to slice model
performance by demographic proxies or confidence bins and render summary
figures.
"""

from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


_CONFIDENCE_BINS = {
    "low_conf": (0.0, 0.5),
    "mid_conf": (0.5, 0.8),
    "high_conf": (0.8, 1.0),
}


def _aggregate_group_metrics(groups: Iterable[str], preds: Iterable[int], targets: Iterable[int]) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    group_to_indices: Dict[str, List[int]] = {}
    for idx, group in enumerate(groups):
        group_to_indices.setdefault(group, []).append(idx)

    preds_list = list(preds)
    targets_list = list(targets)

    for group, indices in group_to_indices.items():
        if not indices:
            continue
        group_preds = [preds_list[i] for i in indices]
        group_targets = [targets_list[i] for i in indices]
        accuracy = accuracy_score(group_targets, group_preds)
        f1 = f1_score(group_targets, group_preds, average="macro")
        records.append({"group": group, "accuracy": accuracy, "f1": f1, "count": len(indices)})

    return pd.DataFrame(records)


def compute_confidence_groups(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    """Return accuracy and F1 scores bucketed by prediction confidence."""

    model.eval()
    all_groups: List[str] = []
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="confidence", leave=False):
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            scores, preds = probs.max(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

            for score in scores.cpu().tolist():
                if score < _CONFIDENCE_BINS["mid_conf"][0]:
                    all_groups.append("low_conf")
                elif score < _CONFIDENCE_BINS["high_conf"][0]:
                    all_groups.append("mid_conf")
                else:
                    all_groups.append("high_conf")

    df = _aggregate_group_metrics(all_groups, all_preds, all_targets)
    if not df.empty:
        df["group"] = pd.Categorical(df["group"], ["low_conf", "mid_conf", "high_conf"], ordered=True)
        df = df.sort_values("group").reset_index(drop=True)
    return df


def compute_age_groups(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    """Compute metrics per synthetic age group from ``FER2013Dataset``."""

    model.eval()
    all_groups: List[str] = []
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for images, targets, groups in tqdm(loader, desc="age_groups", leave=False):
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())
            all_groups.extend(groups)

    return _aggregate_group_metrics(all_groups, all_preds, all_targets)


def plot_group_metrics(df: pd.DataFrame, metric: str, output_path: Path) -> None:
    """Save a bar plot of the given metric per group."""

    if metric not in {"accuracy", "f1"}:
        raise ValueError("metric must be 'accuracy' or 'f1'")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.bar(df["group"], df[metric], color="#4C72B0")
    plt.ylabel(metric.upper())
    plt.xlabel("Group")
    plt.ylim(0.0, 1.0)
    plt.title(f"{metric.upper()} by group")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
