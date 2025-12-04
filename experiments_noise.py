#!/usr/bin/env python
"""Notebook-export script for label-noise and confusion experiments.

This file mirrors `experiments_noise.ipynb`, setting up label smoothing vs.
standard cross-entropy training, simulating noisy labels, and plotting class
confusion to probe robustness.
"""
# coding: utf-8
# Auto-generated from notebook: experiments_noise.ipynb

# %% [markdown] cell 0
# # Label Noise Mitigation & Class Confusion
# 
# This notebook compares standard cross-entropy training with a label-smoothing variant and visualizes class confusion for both models. It also includes a stub for simulating noisy labels to stress-test robustness strategies.

# %% [markdown] cell 1
# ## Setup
# Define the dataset path and common hyperparameters used across experiments. Training runs may take several minutes depending on hardware. Adjust batch size or epochs downward for quick sanity checks.

# %% cell 2
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

# %% cell 3
CSV_PATH = Path("path/to/fer2013.csv")
BATCH_SIZE = 128
EPOCHS = 20
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label_names = list(EMOTION_LABELS.values())

# %% [markdown] cell 4
# ## Helper functions
# A small wrapper wires up dataloaders, launches training, and then reloads the best checkpoint before computing accuracy and confusion matrices on the private test split.

# %% cell 5
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

# %% [markdown] cell 6
# ## Baseline vs. label smoothing
# Run two experiments: one with plain cross-entropy and one with label smoothing (`eps=0.1`). The summary table captures overall test accuracy and loss before diving into class-wise confusion.

# %% cell 7
ce_results = run_experiment("runs/ce_baseline", loss="ce")
ls_results = run_experiment("runs/label_smoothing_eps01", loss="label_smoothing", smoothing_eps=0.1)

comparison_df = pd.DataFrame(
    [
        {"experiment": "CrossEntropy", "test_acc": ce_results["test_acc"], "test_loss": ce_results["test_loss"]},
        {
            "experiment": "LabelSmoothing (eps=0.1)",
            "test_acc": ls_results["test_acc"],
            "test_loss": ls_results["test_loss"],
        },
    ]
)
comparison_df

# %% [markdown] cell 8
# ### Confusion matrices
# Visualize normalized confusion matrices to understand which emotions are most frequently mixed up. Focus especially on Angry/Fear/Sad interactions to see whether label smoothing reduces overconfidence-driven errors.

# %% cell 9
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

plot_confusion(ce_results["cm"], "Baseline cross-entropy confusion")
plot_confusion(ls_results["cm"], "Label smoothing confusion (eps=0.1)")

# %% [markdown] cell 10
# ## Optional: simulate label noise
# Use the helper below to flip a percentage of training labels at random, then rerun the experiments with CE, MixUp, and label smoothing to see how performance degrades under noisy supervision.

# %% cell 11
def inject_noise(df: pd.DataFrame, flip_prob: float = 0.1) -> pd.DataFrame:
    # Randomly flip labels to simulate symmetric noise.
    noisy = df.copy()
    rng = np.random.default_rng(42)
    mask = rng.random(len(noisy)) < flip_prob
    noisy.loc[mask, "emotion"] = rng.integers(0, len(EMOTION_LABELS), size=mask.sum())
    return noisy


# Example usage:
# noisy_df = inject_noise(pd.read_csv(CSV_PATH), flip_prob=0.2)
# noisy_df.to_csv("fer2013_noisy.csv", index=False)
# Then point CSV_PATH to the noisy copy and rerun the experiment cells.

