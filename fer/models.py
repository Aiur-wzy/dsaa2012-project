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

    def __init__(self, in_chans: int = 1, num_classes: int = 7):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_chans, 32, k=3, p=0.05),
            ConvBlock(32, 32, k=3, p=0.05),
            nn.MaxPool2d(2),
            ConvBlock(32, 64, k=3, p=0.1),
            ConvBlock(64, 64, k=3, p=0.1),
            nn.MaxPool2d(2),
            ConvBlock(64, 128, k=3, p=0.15),
            ConvBlock(128, 128, k=3, p=0.15),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):  # pragma: no cover
        x = self.features(x)
        return self.classifier(x)
