"""
ImprovedBackbone: shared feature extractor for all three models.

Architecture (thesis Table 3):
    Block1 (Conv+BN+ReLU, Conv+BN+ReLU, MaxPool, Dropout)  -> (B, 32, 24, 24)
    Block2 (Conv+BN+ReLU, ResBlock, MaxPool, Dropout)      -> (B, 64, 12, 12)
    Block3 (Conv+BN+ReLU, ResBlock, SEBlock, MaxPool)      -> (B, 128, 6, 6)

Total: 280,448 trainable parameters.

Included components:
    - Residual connections for gradient flow (He et al., 2016)
    - Squeeze-and-Excitation for channel attention (Hu et al., 2018)
    - Batch normalization for stable training (Ioffe & Szegedy, 2015)
    - Dropout2d for spatial regularization (Srivastava et al., 2014)
"""

import torch.nn as nn


class ResBlock(nn.Module):
    """Two 3x3 conv layers with a residual connection."""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel-wise recalibration."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.se(x).view(x.size(0), x.size(1), 1, 1)
        return x * scale


class ImprovedBackbone(nn.Module):
    """Shared backbone producing 128-channel 6x6 feature maps."""

    def __init__(self):
        super().__init__()

        # Block 1: (1, 48, 48) -> (32, 24, 24)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
        )

        # Block 2: (32, 24, 24) -> (64, 12, 12)
        self.block2_entry = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.block2_res = ResBlock(64)
        self.block2_pool = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
        )

        # Block 3: (64, 12, 12) -> (128, 6, 6)
        self.block3_entry = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.block3_res = ResBlock(128)
        self.block3_se = SEBlock(128)
        self.block3_pool = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2_entry(x)
        x = self.block2_res(x)
        x = self.block2_pool(x)
        x = self.block3_entry(x)
        x = self.block3_res(x)
        x = self.block3_se(x)
        x = self.block3_pool(x)
        return x  # (B, 128, 6, 6)
