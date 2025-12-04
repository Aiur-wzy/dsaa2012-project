#!/usr/bin/env python
"""Notebook-export script for label-noise and confusion experiments.

This file mirrors `experiments_noise.ipynb`, setting up label smoothing vs.
standard cross-entropy training, simulating noisy labels, and plotting class
confusion to probe robustness.
"""

from pathlib import Path

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch import nn

from fer import (
    EMOTION_LABELS,
    EmotionCNN,
    build_dataloaders,
    evaluate,
    evaluate_with_confusion,
    get_eval_transform,
    get_train_transform,
    load_checkpoint,
    train_model,
)
from fer.losses import LabelSmoothingCE
CSV_PATH = Path("fer2013.csv")
BATCH_SIZE = 128
EPOCHS = 20
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label_names = list(EMOTION_LABELS.values())

def prepare_loaders():
    train_tf = get_train_transform()
    eval_tf = get_eval_transform()
    return build_dataloaders(
        str(CSV_PATH),
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        in_chans=1,
        train_transform=train_tf,
        eval_transform=eval_tf,
    )


def run_experiment(run_dir: str, loss: str = "ce", smoothing_eps: float = 0.1):
    train_loader, val_loader, test_loader = prepare_loaders()
    model = EmotionCNN().to(DEVICE)
    history = train_model(
        model,
        train_loader,
        val_loader,
        epochs=EPOCHS,
        device=DEVICE,
        loss=loss,
        label_smoothing_eps=smoothing_eps,
        ckpt_dir=run_dir,
    )
    load_checkpoint(model, Path(run_dir) / "best.pt", DEVICE)

    ce_criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, ce_criterion, DEVICE)
    cm, report = evaluate_with_confusion(model, test_loader, DEVICE)
    return {
        "history": history,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "cm": cm,
        "report": report,
    }

def plot_confusion(cm, title):
    norm_cm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        norm_cm,
        xticklabels=label_names,
        yticklabels=label_names,
        cmap="mako",
        annot=False,
        cbar_kws={"label": "Normalized frequency"},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    return fig


def main():
    ce_results = run_experiment("runs/ce_baseline", loss="ce")
    ls_results = run_experiment(
        "runs/label_smoothing_eps01", loss="label_smoothing", smoothing_eps=0.1
    )

    comparison_df = pd.DataFrame(
        [
            {
                "experiment": "CrossEntropy",
                "test_acc": ce_results["test_acc"],
                "test_loss": ce_results["test_loss"],
            },
            {
                "experiment": "LabelSmoothing (eps=0.1)",
                "test_acc": ls_results["test_acc"],
                "test_loss": ls_results["test_loss"],
            },
        ]
    )
    print(comparison_df)

    plot_confusion(ce_results["cm"], "Baseline cross-entropy confusion")
    plot_confusion(ls_results["cm"], "Label smoothing confusion (eps=0.1)")


if __name__ == "__main__":
    main()


def inject_noise(df: pd.DataFrame, flip_prob: float = 0.1) -> pd.DataFrame:
    # Randomly flip labels to simulate symmetric noise.
    noisy = df.copy()
    rng = np.random.default_rng(42)
    mask = rng.random(len(noisy)) < flip_prob
    noisy.loc[mask, "emotion"] = rng.integers(0, len(EMOTION_LABELS), size=mask.sum())
    return noisy

