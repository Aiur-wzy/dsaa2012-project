#!/usr/bin/env python
"""Notebook-export script for fairness and group-wise metrics experiments.

This file mirrors `experiments_fairness.ipynb`, exploring proxy demographic
groupings, confidence buckets, and related evaluation utilities.
"""
# coding: utf-8
# Auto-generated from notebook: experiments_fairness.ipynb

# %% [markdown] cell 0
# # Fairness & Group-wise Metrics
# 
# This notebook explores group-aware evaluation without real demographic labels by using proxy groupings such as prediction confidence buckets and a synthetic age-like attribute.

# %% [markdown] cell 1
# ## Grouping strategies
# 
# * **Confidence buckets:** low / mid / high based on maximum softmax probability.
# * **Emotion buckets:** implicit through the labels when inspecting class-wise results.
# * **Synthetic age-like buckets:** deterministic random labels attached to each sample for stress-testing bias-aware metrics.

# %% cell 2
import pandas as pd
import torch
from torch.utils.data import DataLoader

from fer import EmotionCNN, build_dataloaders
from fer.augment import get_eval_transform
from fer.robustness import group_metrics
from fer.train import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
in_chans = 1
eval_tf = get_eval_transform(in_chans)
_, _, test_loader = build_dataloaders(
    "fer2013.csv", batch_size=128, num_workers=0, in_chans=in_chans, train_transform=eval_tf, eval_transform=eval_tf
 )

model = EmotionCNN(in_chans=in_chans).to(device)
state = torch.load("runs/exp1/best.pt", map_location=device)
state_dict = state["state_dict"] if "state_dict" in state else state
model.load_state_dict(state_dict)
criterion = torch.nn.CrossEntropyLoss()

def confidence_group(prob_vec, *_):
    max_conf = prob_vec.max().item()
    if max_conf < 0.5:
        return "low_conf"
    elif max_conf < 0.8:
        return "mid_conf"
    return "high_conf"

loss, acc, details = evaluate(
    model, test_loader, criterion, device, return_details=True, group_fn=confidence_group
 )
print(f"Overall accuracy: {acc:.3f}")
metrics = group_metrics(details["y_true"], details["y_pred"], details["groups"])
table = (
    pd.DataFrame.from_dict(metrics, orient="index")
    .rename(columns={"acc": "accuracy", "f1": "f1"})
)
table["count"] = pd.Series(details["groups"]).value_counts()
display(table)

# %% cell 3
from fer.data import FER2013Dataset

age_test_ds = FER2013Dataset("fer2013.csv", usage="PrivateTest", transform=eval_tf, in_chans=in_chans, return_group=True)
age_test_loader = DataLoader(age_test_ds, batch_size=128, shuffle=False)

_, _, age_details = evaluate(
    model, age_test_loader, criterion, device, return_details=True
 )
age_metrics = group_metrics(age_details["y_true"], age_details["y_pred"], age_details["groups"])
age_table = pd.DataFrame.from_dict(age_metrics, orient="index")
age_table["count"] = pd.Series(age_details["groups"]).value_counts()
display(age_table)

# %% [markdown] cell 4
# ## Notes for reports
# 
# * Proxy groups highlight where the model struggles (e.g., low confidence regions).
# * Synthetic groups do not reflect real demographics but expose sensitivity to distribution shifts.
# * Class-wise inspection can complement these aggregated views to reason about emotion-specific biases.

