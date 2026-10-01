"""
Model efficiency metrics (thesis Table 14).

Functions:
    - count_parameters:        number of trainable parameters
    - measure_flops:           multiply-accumulate operations (MACs)
    - measure_inference_time:  average inference time per image (ms)
"""

import time

import torch
from thop import profile


def count_parameters(model):
    """Return the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_flops(model, input_shape=(1, 1, 48, 48), device='cuda'):
    """Measure MACs (multiply-accumulate operations) using thop.

    Args:
        model: torch module
        input_shape: shape of a single input sample (default: (1, 1, 48, 48))
        device: torch device

    Returns:
        MACs as a float, or None if profiling fails.
    """
    inp = torch.randn(*input_shape).to(device)
    try:
        macs, _ = profile(model, inputs=(inp,), verbose=False)
        return macs
    except Exception as e:
        print(f"FLOP profiling failed: {e}")
        return None


def measure_inference_time(model, loader, device='cuda', num_batches=20):
    """Measure average inference time per image (in milliseconds).

    Warms up with one batch, then times `num_batches` batches and divides
    by the total number of images processed.

    Args:
        model: torch module (in eval mode ideally)
        loader: DataLoader
        device: torch device
        num_batches: number of batches to average over

    Returns:
        Average milliseconds per image.
    """
    model.eval()

    # Warm-up
    for images, _ in loader:
        _ = model(images.to(device))
        break

    start = time.time()
    batch_size = 1
    for i, (images, _) in enumerate(loader):
        if i >= num_batches:
            break
        images = images.to(device)
        _ = model(images)
        batch_size = images.shape[0]
    elapsed = time.time() - start

    return (elapsed / (num_batches * batch_size)) * 1000
