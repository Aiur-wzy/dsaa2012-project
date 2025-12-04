"""Loss functions tailored for FER training experiments.

Currently exposes :class:`LabelSmoothingCE`, a configurable cross-entropy
variant that reduces overconfidence during optimization.
"""

import torch
from torch import nn


class LabelSmoothingCE(nn.Module):
    """Cross-entropy loss with label smoothing.

    This implementation mirrors the formulation used in many classification
    setups, distributing a small probability mass uniformly across all
    non-target classes to reduce overconfidence.
    """

    def __init__(self, num_classes: int, eps: float = 0.1) -> None:
        super().__init__()
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than 1 for label smoothing")
        if not 0 <= eps < 1:
            raise ValueError("eps should be in [0, 1)")
        self.eps = eps
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.eps / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1 - self.eps)
        return (-true_dist * log_probs).sum(dim=1).mean()
