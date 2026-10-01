"""
Evaluation metrics and helpers.

Functions in this module:
    - get_predictions:           run a model over a loader, collect preds + labels
    - get_acc:                   accuracy (%) — thin wrapper over get_predictions
    - evaluate_tta:              accuracy with horizontal-flip test-time augmentation
    - predict_single:            predict the class of one image
    - mcnemar_test:              McNemar's test between two models
    - evaluate_overlay_accuracy: proportion of images where attention is on the face
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from statsmodels.stats.contingency_tables import mcnemar
from tqdm import tqdm


def get_predictions(model, loader, device):
    """Run a model over a loader and return (predictions, labels).

    Handles models that return either a tensor or a tuple (logits, ...).
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)


def get_acc(model, loader, device):
    """Return accuracy (%) on a loader."""
    preds, labels = get_predictions(model, loader, device)
    return accuracy_score(labels, preds) * 100


def evaluate_tta(model, loader, device):
    """Accuracy (%) with horizontal-flip TTA (thesis Section 3.3.5).

    Averages softmax probabilities from the original and flipped image:
        p_tta = 0.5 * (p(x) + p(x_flip))
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            out1 = model(images)
            if isinstance(out1, tuple):
                out1 = out1[0]

            out2 = model(torch.flip(images, dims=[3]))
            if isinstance(out2, tuple):
                out2 = out2[0]

            probs = (F.softmax(out1, dim=1) + F.softmax(out2, dim=1)) / 2
            _, preds = probs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return accuracy_score(all_labels, all_preds) * 100


def predict_single(model, img_tensor, device):
    """Predict the class of a single image tensor.

    Args:
        model: trained model
        img_tensor: shape (C, H, W)
        device: torch device

    Returns:
        Integer class index.
    """
    inp = img_tensor.float().unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(inp)
        if isinstance(out, tuple):
            out = out[0]
        return out.argmax(dim=1).item()


def mcnemar_test(model_a, model_b, loader, device):
    """McNemar's test between two models on the same data (thesis Table 16).

    Returns the p-value under the null hypothesis that both models have equal
    error rates.
    """
    preds_a, preds_b, labels = [], [], []

    with torch.no_grad():
        for images, lbls in loader:
            images = images.to(device)
            out_a = model_a(images)
            out_b = model_b(images)
            if isinstance(out_a, tuple):
                out_a = out_a[0]
            if isinstance(out_b, tuple):
                out_b = out_b[0]
            _, pa = out_a.max(1)
            _, pb = out_b.max(1)
            preds_a.extend(pa.cpu().numpy())
            preds_b.extend(pb.cpu().numpy())
            labels.extend(lbls.cpu().numpy())

    preds_a = np.array(preds_a)
    preds_b = np.array(preds_b)
    labels = np.array(labels)

    both_wrong = ((preds_a != labels) & (preds_b != labels)).sum()
    a_wrong_b_right = ((preds_a != labels) & (preds_b == labels)).sum()
    a_right_b_wrong = ((preds_a == labels) & (preds_b != labels)).sum()
    both_right = ((preds_a == labels) & (preds_b == labels)).sum()

    table = [[both_right, a_wrong_b_right],
             [a_right_b_wrong, both_wrong]]

    return mcnemar(table, exact=True).pvalue


def evaluate_overlay_accuracy(model, loader, device, attention_type='cam',
                               max_samples=None, size=48):
    """Overlay accuracy: proportion of images where attention energy inside
    the face region exceeds attention outside (thesis Table 12).

    Args:
        attention_type: 'cam' uses the CAM heatmap; 'mask' uses MaskCAM's mask.
        max_samples: optional cap on the number of images to process.

    Returns:
        (total_valid, num_correct) — total images with detected faces, and
        count where attention concentrated on the face.
    """
    from .landmarks import get_face_landmarks_mask
    from .visualization import cam_attention, mask_attention

    model.eval()
    total, correct, processed = 0, 0, 0

    for images, labels in tqdm(loader, desc=f'Overlay acc ({attention_type})'):
        images, labels = images.to(device), labels.to(device)
        for i in range(images.size(0)):
            if max_samples is not None and processed >= max_samples:
                break

            img_tensor = images[i].unsqueeze(0)
            lbl = labels[i].item()
            face_mask = get_face_landmarks_mask(img_tensor.cpu())
            if face_mask.sum() == 0:
                continue

            if attention_type == 'cam':
                att = cam_attention(model, img_tensor, lbl, device)
            else:
                att = mask_attention(model, img_tensor, device)

            inside = att[face_mask == 1].mean()
            outside = att[face_mask == 0].mean()
            if inside > outside:
                correct += 1
            total += 1
            processed += 1

        if max_samples is not None and processed >= max_samples:
            break

    return total, correct