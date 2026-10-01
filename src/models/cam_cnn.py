"""
CAMCNN: standard CAM baseline (thesis Section 3.2.3).

Replaces the fully-connected head with a 1x1 conv layer + global average
pooling. This enables class activation map (CAM) visualization: the 1x1
conv output is the class-specific heatmap.

0.47M trainable parameters.
"""

import torch.nn as nn

from .backbone import ImprovedBackbone


class CAMCNN(nn.Module):
    """CAM-based classifier: 1x1 conv + global average pooling."""

    def __init__(self, num_classes=7):
        super().__init__()
        self.features = ImprovedBackbone()
        self.conv_class = nn.Conv2d(128, num_classes, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, return_cam=False):
        f = self.features(x)                # (B, 128, 6, 6)
        s = self.conv_class(f)              # (B, num_classes, 6, 6)
        logits = self.gap(s).squeeze(-1).squeeze(-1)

        if return_cam:
            return logits, s

        return logits
