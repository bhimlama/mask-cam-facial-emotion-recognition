"""
MixUp utilities for training BaselineCNN and CAMCNN.

MixUp (Zhang et al., 2018) generates virtual training examples by linearly
interpolating pairs of images and their labels:

    x_mixed = lambda * x + (1 - lambda) * x[shuffled]
    y_mixed = lambda * y_a + (1 - lambda) * y_b

This regularization technique is applied to BaselineCNN and CAMCNN but
omitted from Mask-CAM, since MixUp's interpolated targets can destabilize
the self-supervised mask generator.
"""

import numpy as np
import torch


def mixup_data(x, y, alpha=0.2):
    """Generate a MixUp batch.

    Args:
        x: input tensor (B, C, H, W)
        y: label tensor (B,)
        alpha: Beta distribution parameter. If 0, no mixing.

    Returns:
        mixed_x: mixed input tensor
        y_a, y_b: the two label tensors being mixed
        lam: the mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute the MixUp loss for a batch.

    Args:
        criterion: the base loss function (e.g., nn.CrossEntropyLoss)
        pred: model predictions
        y_a, y_b: the two label sets
        lam: mixing coefficient from mixup_data

    Returns:
        Weighted loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
