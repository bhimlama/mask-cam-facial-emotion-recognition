"""
Visualization utilities.

Functions:
    - generate_cam:             48x48 normalized CAM for a single image
    - generate_mask:            48x48 normalized mask for a single image
    - cam_attention:            CAM resized/normalized (used in overlay analysis)
    - mask_attention:           mask resized/normalized (used in overlay analysis)
    - render_cam_overlay:       blend image with a JET-colormapped CAM
    - render_masked_image:      image with suppressed regions filled with red
    - plot_cam_and_mask:        3- or 5-panel visualization for one image
    - plot_emotion_grid:        grid of mask overlays for one emotion (Fig 20)
    - plot_cam_vs_maskcam_grid: one sample per emotion, 3-column comparison (Fig 21)
"""

import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from .models import CAMCNN, MaskCAM


DEFAULT_SUPPRESSED_COLOR = np.array([0.9, 0.0, 0.0])  # red


# =============================================================================
# Basic CAM / mask extraction
# =============================================================================

def generate_cam(model, image_tensor, device='cuda'):
    """Generate a 48x48 normalized CAM for the predicted class.

    Works for CAMCNN and MaskCAM (both expose `return_cam=True`).

    Args:
        model: CAMCNN or MaskCAM instance (in eval mode)
        image_tensor: (C, H, W) tensor
        device: torch device

    Returns:
        (cam, pred_class) or (None, None) if the model doesn't produce a CAM.
    """
    model.eval()
    with torch.no_grad():
        if isinstance(model, (CAMCNN, MaskCAM)):
            logits, s = model(image_tensor.unsqueeze(0).to(device), return_cam=True)
        else:
            return None, None

        pred_class = logits.argmax(dim=1).item()
        cam = s[0, pred_class].cpu().numpy()

    cam = cv2.resize(cam, (48, 48), interpolation=cv2.INTER_LINEAR)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam, pred_class


def generate_mask(model, image_tensor, device='cuda'):
    """Generate a 48x48 normalized mask from MaskCAM.

    Returns:
        (48, 48) numpy array, or None if the model isn't MaskCAM.
    """
    model.eval()
    with torch.no_grad():
        if isinstance(model, MaskCAM):
            _, _, m = model(
                image_tensor.unsqueeze(0).to(device),
                return_cam=True, return_mask=True,
            )
            mask = m[0, 0].cpu().numpy()
        else:
            return None

    mask = cv2.resize(mask, (48, 48), interpolation=cv2.INTER_LINEAR)
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return mask


def cam_attention(model, img_tensor, label_idx, device='cuda'):
    """Extract a 48x48 normalized CAM for a specific label index."""
    with torch.no_grad():
        _, s = model(img_tensor.to(device), return_cam=True)
    cam = s[0, label_idx].cpu().numpy()
    cam = cv2.resize(cam, (48, 48), interpolation=cv2.INTER_LINEAR)
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


def mask_attention(model, img_tensor, device='cuda'):
    """Extract a 48x48 normalized mask from MaskCAM."""
    with torch.no_grad():
        _, mask = model(img_tensor.to(device), return_mask=True)
    mask = mask.squeeze().cpu().numpy()
    mask = cv2.resize(mask, (48, 48), interpolation=cv2.INTER_LINEAR)
    return (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)


# =============================================================================
# Rendering helpers
# =============================================================================

def _unnormalize_image(tensor):
    """Reverse Normalize(0.5, 0.5) and return a (H, W) numpy array."""
    img = tensor.cpu().clone() * 0.5 + 0.5
    img = img.squeeze().numpy()
    return np.clip(img, 0, 1)


def render_cam_overlay(image_tensor, cam, alpha=0.4):
    """Blend original grayscale image with a JET-colormapped CAM."""
    img = _unnormalize_image(image_tensor)
    cam_colored = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB) / 255.0
    overlay = (1 - alpha) * np.stack([img] * 3, axis=-1) + alpha * cam_colored
    return np.clip(overlay, 0, 1)


def render_masked_image(image_tensor, mask, bg_color=DEFAULT_SUPPRESSED_COLOR):
    """Render original image with suppressed regions (mask=0) filled with a color."""
    img = _unnormalize_image(image_tensor)
    mask_3c = np.stack([mask] * 3, axis=-1)
    img_3c = np.stack([img] * 3, axis=-1)
    out = mask_3c * img_3c + (1 - mask_3c) * bg_color
    return np.clip(out, 0, 1)


# =============================================================================
# Multi-panel visualizations
# =============================================================================

def plot_cam_and_mask(model, image_tensor, true_label, class_names,
                       device='cuda', bg_color=DEFAULT_SUPPRESSED_COLOR,
                       save_path=None):
    """Plot a 3- or 5-panel figure for a single image.

    3 panels: Original | CAM | CAM overlay (for CAMCNN)
    5 panels: + Learned mask | Masked original (for MaskCAM)

    Args:
        model: CAMCNN or MaskCAM instance
        image_tensor: (C, H, W)
        true_label: integer class index
        class_names: list of emotion strings
        device: torch device
        bg_color: color for suppressed regions
        save_path: optional filepath to save

    Returns:
        (cam, pred_class, mask) — mask is None for CAMCNN.
    """
    cam, pred_class = generate_cam(model, image_tensor, device)
    if cam is None:
        print("This model does not produce a CAM.")
        return None, None, None

    mask = generate_mask(model, image_tensor, device)
    has_mask = mask is not None
    ncols = 5 if has_mask else 3

    img_disp = _unnormalize_image(image_tensor)
    cam_overlay = render_cam_overlay(image_tensor, cam)

    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
    if ncols == 3:
        axes = [axes] if not isinstance(axes, np.ndarray) else axes
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    axes[0].imshow(img_disp, cmap='gray')
    axes[0].set_title(f"Original (True: {class_names[true_label]})")
    axes[0].axis('off')

    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title(f"CAM (Pred: {class_names[pred_class]})")
    axes[1].axis('off')

    axes[2].imshow(cam_overlay)
    axes[2].set_title("CAM Overlay")
    axes[2].axis('off')

    if has_mask:
        axes[3].imshow(mask, cmap='jet')
        axes[3].set_title("Learned Mask")
        axes[3].axis('off')

        axes[4].imshow(render_masked_image(image_tensor, mask, bg_color))
        axes[4].set_title("Masked Original\n(red = suppressed)")
        axes[4].axis('off')

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    return cam, pred_class, mask


def plot_emotion_grid(model, dataset, emotion_idx, class_names,
                       device='cuda', max_display=10, n_cols=5,
                       bg_color=DEFAULT_SUPPRESSED_COLOR, save_path=None):
    """Grid of mask overlays for random images of one emotion (thesis Figure 20).

    Args:
        model: MaskCAM instance (or any model; if not MaskCAM, shows originals)
        dataset: torch Dataset yielding (image_tensor, label)
        emotion_idx: int, index of the emotion to visualize
        class_names: list of emotion strings
        device: torch device
        max_display: max number of images to show
        n_cols: number of grid columns
        bg_color: color for suppressed regions
        save_path: optional filepath

    Returns:
        List of indices that were visualized.
    """
    emotion_name = class_names[emotion_idx]
    all_indices = [i for i, (_, lbl) in enumerate(dataset) if lbl == emotion_idx]
    total = len(all_indices)
    print(f"Total '{emotion_name}' images: {total}")

    if max_display and total > max_display:
        indices = random.sample(all_indices, max_display)
    else:
        indices = all_indices[:max_display] if max_display else all_indices

    if not indices:
        print("No images to display.")
        return []

    print(f"Displaying {len(indices)} images.")

    n_rows = (len(indices) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

    is_mask_model = isinstance(model, MaskCAM)
    model.eval()

    for ax, idx in zip(axes, indices):
        img_tensor, _ = dataset[idx]
        img_disp = _unnormalize_image(img_tensor)

        if is_mask_model:
            mask = generate_mask(model, img_tensor, device)
            display = render_masked_image(img_tensor, mask, bg_color)
        else:
            display = np.stack([img_disp] * 3, axis=-1)

        ax.imshow(display)
        ax.set_title(f"True: {emotion_name}", fontsize=8)
        ax.axis('off')

    for ax in axes[len(indices):]:
        ax.axis('off')

    plt.suptitle(
        f"Masked overlays for '{emotion_name}' images (red = suppressed)",
        fontsize=14,
    )
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    return indices


def plot_cam_vs_maskcam_grid(cam_model, mask_model, dataset, class_names,
                               device='cuda', save_path=None):
    """One sample per emotion: Original | CAM | Mask-CAM (thesis Figure 21).

    Args:
        cam_model: CAMCNN instance (in eval mode)
        mask_model: MaskCAM instance (in eval mode)
        dataset: torch Dataset yielding (image_tensor, label)
        class_names: list of emotion strings
        device: torch device
        save_path: optional filepath

    Returns:
        List of (index, emotion_name) pairs visualized.
    """
    selected = []
    for emotion_idx, emotion_name in enumerate(class_names):
        indices = [i for i, (_, lbl) in enumerate(dataset) if lbl == emotion_idx]
        if not indices:
            print(f"No samples for {emotion_name}")
            continue
        selected.append((random.choice(indices), emotion_name))

    n = len(selected)
    if n == 0:
        return []

    fig, axes = plt.subplots(n, 3, figsize=(9, 1.5 * n))
    if n == 1:
        axes = axes.reshape(1, 3)

    for col, header in enumerate(['Original', 'CAM', 'Masked-CAM']):
        axes[0, col].set_title(header, fontsize=12, pad=15)

    for i, (idx, emotion_name) in enumerate(selected):
        img_tensor, label = dataset[idx]
        img_t = img_tensor.unsqueeze(0).to(device)

        cam_att = cam_attention(cam_model, img_t, label, device)
        mask_att = mask_attention(mask_model, img_t, device)
        img_np = _unnormalize_image(img_tensor)

        axes[i, 0].imshow(img_np, cmap='gray')
        axes[i, 0].set_title(emotion_name, fontsize=10)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(img_np, cmap='gray')
        axes[i, 1].imshow(cam_att, cmap='jet', alpha=0.5)
        axes[i, 1].axis('off')

        axes[i, 2].imshow(img_np, cmap='gray')
        axes[i, 2].imshow(mask_att, cmap='jet', alpha=0.5)
        axes[i, 2].axis('off')

    plt.suptitle('Interpretability: One Sample per Emotion',
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    return selected