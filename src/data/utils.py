"""
Small data utilities shared across the repo.
"""

from torch.utils.data import Dataset


class TensorDataset(Dataset):
    """Simple Dataset that wraps two tensors (images and labels).

    Used when we preload an entire dataset into RAM (thesis Section 3.5.2).
    This avoids re-running the transform pipeline on every epoch, which
    speeds up training significantly for small datasets like FERPlus.
    """

    def __init__(self, images, labels):
        """
        Args:
            images: torch.Tensor of shape (N, C, H, W)
            labels: torch.Tensor of shape (N,)
        """
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
