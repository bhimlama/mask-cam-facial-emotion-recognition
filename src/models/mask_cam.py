"""
Mask-CAM: the proposed model (thesis Section 3.2.4).

Extends CAMCNN with a self-supervised soft mask generator. The mask is
applied element-wise to the backbone's feature maps before the classifier,
forcing the model to focus on discriminative facial regions.

The mask is learned via a composite loss (see src/losses.py):
    L_total = L_CE + lambda * mean(mask) + beta * TV(mask)

The mask generator is a single 3x3 conv + sigmoid. No external mask labels
are needed.

0.48M trainable parameters (0.01M more than CAMCNN, due to the mask conv).
"""

import torch.nn as nn

from .backbone import ImprovedBackbone


class MaskCAM(nn.Module):
    """Masked Class Activation Mapping.

    Args:
        num_classes: number of emotion classes (7)
        lambda_sparsity: initial sparsity weight (used in loss, stored for
                         reference; the training loop reads model.lambda_sparsity)
    """

    def __init__(self, num_classes=7, lambda_sparsity=0.001):
        super().__init__()
        self.features = ImprovedBackbone()

        # Self-supervised mask generator: 3x3 conv -> sigmoid
        self.mask_conv = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        # Classifier head (same as CAMCNN)
        self.conv_class = nn.Conv2d(128, num_classes, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Stored for the training loop to read
        self.lambda_sparsity = lambda_sparsity

    def forward(self, x, return_cam=False, return_mask=False):
        f = self.features(x)                    # (B, 128, 6, 6)
        m = self.mask_conv(f)                   # (B, 1, 6, 6)
        f_masked = f * m                        # element-wise broadcast
        s = self.conv_class(f_masked)           # (B, num_classes, 6, 6)
        logits = self.gap(s).squeeze(-1).squeeze(-1)

        if return_cam and return_mask:
            return logits, s, m
        elif return_cam:
            return logits, s
        elif return_mask:
            return logits, m
        return logits
