"""
BaselineCNN: black-box reference model (thesis Section 3.2.2).

Uses the shared ImprovedBackbone followed by two fully-connected layers.
2.84M trainable parameters.
"""

import torch.nn as nn

from .backbone import ImprovedBackbone


class BaselineCNN(nn.Module):
    """Standard CNN classifier with fully-connected head."""

    def __init__(self, num_classes=7):
        super().__init__()
        self.features = ImprovedBackbone()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
