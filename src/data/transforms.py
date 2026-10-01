"""
Data augmentation transforms for FER datasets.

Two transforms are provided:
    - get_train_transform(): training with augmentation (thesis Section 3.3.3)
    - get_test_transform():  deterministic, used for val/test

Also includes RandomGridOcclusion, a custom augmentation that randomly
occludes a 16x16 block of the 48x48 image.
"""

import random

from torchvision import transforms


def get_train_transform():
    """Training transform with augmentation (thesis Section 3.3.3).

    Includes:
        - Random horizontal flip
        - Random rotation (±20°)
        - Random affine (translation ±15%, scale 85-115%)
        - Color jitter (brightness, contrast, saturation, hue)
        - Random grayscale (p=0.1)
        - RandomGridOcclusion (p=0.3)
        - Normalize to mean=0.5, std=0.5
    """
    return transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.15, 0.15),
            scale=(0.85, 1.15),
        ),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.05,
        ),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        RandomGridOcclusion(p=0.3, fill=0.0),
    ])


def get_test_transform():
    """Deterministic transform for validation and test sets (thesis Section 3.1.3)."""
    return transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


class RandomGridOcclusion:
    """Randomly occludes a 16x16 region of a 48x48 image.

    The image is treated as a 6x6 grid of 8x8 cells. A random 2x2 block
    of cells (16x16 pixels) is filled with `fill` with probability `p`.

    Used as a data augmentation during training only.

    Args:
        p: probability of applying the occlusion
        fill: value to fill the occluded region (0.0 after Normalize(0.5, 0.5)
              corresponds to mid-gray)
    """
    def __init__(self, p=0.3, fill=0.0):
        self.p = p
        self.fill = fill

    def __call__(self, tensor):
        if random.random() > self.p:
            return tensor

        # Random 2x2 block position on the 6x6 grid
        row = random.randint(0, 4)
        col = random.randint(0, 4)
        y1 = row * 8
        y2 = y1 + 16
        x1 = col * 8
        x2 = x1 + 16

        tensor[:, y1:y2, x1:x2] = self.fill
        return tensor
