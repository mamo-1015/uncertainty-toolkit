"""
model.py
--------
A compact ResNet-style classifier for Fashion-MNIST.
Swap in any larger model and EpistemicEstimator still works.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Basic residual block with BatchNorm and optional dropout."""

    def __init__(self, channels: int, dropout_p: float = 0.3) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        return F.relu(out + residual)


class FashionMNISTResNet(nn.Module):
    """
    Compact ResNet classifier for Fashion-MNIST (1×28×28, 10 classes).

    Parameters
    ----------
    num_classes : int
        Output dimensionality (10 for Fashion-MNIST).
    dropout_p : float
        Dropout probability used in residual blocks and the classifier head.
        Controls the strength of stochastic regularisation; also the
        variance of MC Dropout samples at inference time.
    base_channels : int
        Number of feature maps in the first conv layer.  Controls capacity.
    """

    def __init__(
        self,
        num_classes: int = 10,
        dropout_p: float = 0.3,
        base_channels: int = 32,
    ) -> None:
        super().__init__()

        c = base_channels

        # Stem: grayscale → feature maps
        self.stem = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )

        # Stage 1 — 28×28
        self.stage1 = nn.Sequential(
            ResidualBlock(c, dropout_p),
            ResidualBlock(c, dropout_p),
        )
        self.pool1 = nn.MaxPool2d(2)  # → 14×14

        # Stage 2 — 14×14 with wider channels
        self.expand = nn.Sequential(
            nn.Conv2d(c, c * 2, 1, bias=False),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(c * 2, dropout_p),
            ResidualBlock(c * 2, dropout_p),
        )
        self.pool2 = nn.MaxPool2d(2)  # → 7×7

        # Classifier head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),       # <-- a Dropout (not Dropout2d) for variety
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.pool1(self.stage1(x))
        x = self.pool2(self.stage2(self.expand(x)))
        x = self.gap(x)
        return self.head(x)
