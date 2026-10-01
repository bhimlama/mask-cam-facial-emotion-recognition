"""
Composite loss and warm-up schedule for Mask-CAM.

The composite loss (thesis Equation 1):

    L_total = L_CE + lambda * mean(mask) + beta * TV(mask)

where:
    L_CE     = cross-entropy with label smoothing 0.1
    mean(mask) = sparsity penalty (thesis Equation 2)
    TV(mask)  = total-variation smoothness penalty (thesis Equations 3 & 4)

These three terms compete:
    - CE keeps all information needed for classification
    - Sparsity pushes the mask to suppress as many regions as possible
    - Smoothness ensures the kept regions form a contiguous blob

The lambda warm-up (thesis Equation 5) gradually introduces the sparsity
pressure over the first T_warmup epochs so the mask doesn't collapse before
the classifier learns useful features.
"""

import torch
import torch.nn.functional as F


def composite_loss(logits, targets, mask, lambda_sparsity, beta_smoothness,
                   label_smoothing=0.1):
    """Composite loss for Mask-CAM (thesis Equation 1).

    Args:
        logits: (B, num_classes) model output
        targets: (B,) ground-truth labels
        mask: (B, 1, H, W) soft mask from the mask generator
        lambda_sparsity: weight for the sparsity penalty
        beta_smoothness: weight for the total-variation smoothness penalty
        label_smoothing: label smoothing value for cross-entropy

    Returns:
        total: the scalar total loss (for backpropagation)
        parts: dict with 'cls', 'sparsity', 'smoothness' — each a Python float,
               for logging
    """
    # Cross-entropy (with label smoothing)
    cls_loss = F.cross_entropy(
        logits, targets, label_smoothing=label_smoothing)

    # Sparsity: mean mask value (thesis Equation 2)
    sparsity_loss = mask.mean()

    # Smoothness: total variation (thesis Equations 3 & 4)
    tv_x = torch.abs(mask[:, :, 1:, :] - mask[:, :, :-1, :]).mean()
    tv_y = torch.abs(mask[:, :, :, 1:] - mask[:, :, :, :-1]).mean()
    smoothness_loss = tv_x + tv_y

    # Weighted total
    total = (cls_loss
             + lambda_sparsity * sparsity_loss
             + beta_smoothness * smoothness_loss)

    parts = {
        'cls': cls_loss.item(),
        'sparsity': sparsity_loss.item(),
        'smoothness': smoothness_loss.item(),
    }
    return total, parts


def lambda_warmup(epoch, lambda_max, warmup_epochs=10):
    """Linear warm-up for the sparsity weight (thesis Equation 5).

        lambda_eff(t) = lambda_max * t / T_warmup   if t <= T_warmup
                      = lambda_max                    otherwise

    Args:
        epoch: current epoch (0-indexed)
        lambda_max: final value of lambda (e.g., 0.0012)
        warmup_epochs: T_warmup, number of warm-up epochs

    Returns:
        The effective lambda for this epoch.
    """
    if epoch < warmup_epochs:
        return lambda_max * (epoch / warmup_epochs)
    return lambda_max
