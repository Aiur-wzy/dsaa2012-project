import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, p: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=k // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
        )

    def forward(self, x):  # pragma: no cover - simple wrapper
        return self.block(x)


class EmotionCNN(nn.Module):
    """A compact CNN tailored for 48x48 FER-2013 inputs."""

    def __init__(self, in_chans: int = 1, num_classes: int = 7, width_mult: float = 1.0):
        super().__init__()

        def _scale(channel: int) -> int:
            return max(1, int(round(channel * width_mult)))

        c1 = _scale(32)
        c2 = _scale(64)
        c3 = _scale(128)
        c4 = _scale(256)
        classifier_dim = _scale(128)

        self.features = nn.Sequential(
            ConvBlock(in_chans, c1, k=3, p=0.05),
            ConvBlock(c1, c1, k=3, p=0.05),
            nn.MaxPool2d(2),
            ConvBlock(c1, c2, k=3, p=0.1),
            ConvBlock(c2, c2, k=3, p=0.1),
            nn.MaxPool2d(2),
            ConvBlock(c2, c3, k=3, p=0.15),
            ConvBlock(c3, c3, k=3, p=0.15),
            nn.MaxPool2d(2),
            nn.Conv2d(c3, c4, kernel_size=3, padding=1),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(c4, classifier_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(classifier_dim, num_classes),
        )

    def forward(self, x):  # pragma: no cover
        x = self.features(x)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters for capacity studies."""

    return sum(p.numel() for p in model.parameters() if p.requires_grad)
