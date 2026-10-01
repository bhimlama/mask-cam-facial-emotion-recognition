"""
Facial landmark utilities.

Two landmark libraries are used in this repo for different purposes:

1. `face_alignment` (FAN-based, 68 landmarks)
   Used by `create_masks_from_landmarks` to compute per-region masks
   (eyes, mouth, face) for the mask density analysis
   (thesis Table 13, Figure 19).

2. `face_recognition` (dlib-based, 68 landmarks)
   Used by `get_face_landmarks_mask` to compute a single convex-hull
   face mask for the overlay accuracy analysis
   (thesis Table 12) and the CAM vs Mask-CAM comparison (thesis Figure 21).

Both libraries detect the same 68 landmarks but with different APIs and
coordinate conventions, so they are kept separate.
"""

import cv2
import numpy as np
import torch
from scipy.spatial import ConvexHull
from skimage.draw import polygon


# Value used to fill occluded regions (after Normalize(0.5, 0.5))
FILL_BLACK = -1.0
H, W = 48, 48


# =============================================================================
# Heuristic fallback masks (used when landmark detection fails)
# =============================================================================

def heuristic_face_oval():
    """Simple elliptical face-region heuristic for 48x48 images."""
    y_grid, x_grid = np.ogrid[:H, :W]
    cy, cx = H / 2, W / 2
    mask = ((x_grid - cx) ** 2 / (W * 0.42) ** 2
            + (y_grid - cy) ** 2 / (H * 0.45) ** 2) <= 1
    return mask.astype(np.float32)


def heuristic_eyes_mouth():
    """Eye + mouth region heuristic for 48x48 images."""
    mask = np.zeros((H, W), dtype=np.float32)
    mask[:H // 3, :W // 3] = 1
    mask[:H // 3, 2 * W // 3:] = 1
    mask[2 * H // 3:, W // 4:3 * W // 4] = 1
    return mask


def heuristic_eyes_mouth_cheeks():
    """Eye + mouth + cheek region heuristic."""
    mask = heuristic_eyes_mouth().copy()
    mask[H // 3:2 * H // 3, int(W * 0.2):int(W * 0.8)] = 1
    return mask


# =============================================================================
# face_alignment-based mask creation (per-region masks)
# =============================================================================

def create_masks_from_landmarks(landmarks_224):
    """Convert 68 landmarks (in 224x224 coordinates) into three 48x48 masks.

    Args:
        landmarks_224: (68, 2) array of landmark coordinates in 224x224 space

    Returns:
        (face_mask, eyes_mouth_mask, eyes_mouth_cheeks_mask) — three (48,48)
        numpy arrays in [0, 1].
    """
    scale = 48.0 / 224.0
    lm = landmarks_224 * scale

    def poly_mask(indices):
        points = lm[indices].astype(int)
        points[:, 0] = np.clip(points[:, 0], 0, 47)
        points[:, 1] = np.clip(points[:, 1], 0, 47)
        rr, cc = polygon(points[:, 1], points[:, 0], shape=(48, 48))
        mask = np.zeros((48, 48), dtype=np.float32)
        mask[rr, cc] = 1.0
        return mask

    # Face mask (convex hull of all landmarks)
    try:
        hull = ConvexHull(lm)
        face_mask = poly_mask(hull.vertices)
    except Exception:
        face_mask = heuristic_face_oval()

    # Eyes + mouth mask
    try:
        left_eye = poly_mask(list(range(36, 42)))
        right_eye = poly_mask(list(range(42, 48)))
        mouth = poly_mask(list(range(48, 68)))
        eyes_mouth_mask = np.clip(left_eye + right_eye + mouth, 0, 1)
    except Exception:
        eyes_mouth_mask = heuristic_eyes_mouth()

    # Eyes + mouth + jaw/cheeks
    try:
        jaw = poly_mask(list(range(0, 17)))
        eyes_mouth_cheeks_mask = np.clip(jaw + eyes_mouth_mask, 0, 1)
    except Exception:
        eyes_mouth_cheeks_mask = heuristic_eyes_mouth_cheeks()

    return face_mask, eyes_mouth_mask, eyes_mouth_cheeks_mask


# =============================================================================
# face_recognition-based mask creation (single face mask)
# =============================================================================

def get_face_landmarks_mask(image_tensor, size=48):
    """Return a binary face mask using face_recognition's 68-point detector.

    Used by the overlay accuracy analysis (thesis Table 12).

    Args:
        image_tensor: normalized (1, H, W) tensor
        size: output mask size (default 48)

    Returns:
        (size, size) uint8 mask with 1s inside the convex-hull face region,
        or all zeros if no face was detected.
    """
    import face_recognition

    img_np = image_tensor.squeeze().cpu().numpy() * 0.5 + 0.5
    img_uint8 = (img_np * 255).astype(np.uint8)
    img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)

    landmarks_list = face_recognition.face_landmarks(img_rgb)
    if len(landmarks_list) == 0:
        return np.zeros((size, size), dtype=np.uint8)

    points = []
    for feature in landmarks_list[0].values():
        for point in feature:
            x = max(0, min(point[0], size - 1))
            y = max(0, min(point[1], size - 1))
            points.append((x, y))

    if len(points) == 0:
        return np.zeros((size, size), dtype=np.uint8)

    hull = cv2.convexHull(np.array(points))
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    return mask


# =============================================================================
# Mask application
# =============================================================================

def apply_mask(img_tensor, mask_2d, fill=FILL_BLACK):
    """Zero out (or fill with `fill`) regions outside the mask.

    Args:
        img_tensor: normalized (1, H, W) tensor
        mask_2d: (H, W) numpy mask in [0, 1]
        fill: value used for occluded regions

    Returns:
        Masked image tensor with the same shape.
    """
    img = img_tensor.clone()
    m = torch.from_numpy(mask_2d).float().unsqueeze(0)
    return img * m + (1 - m) * fill


# =============================================================================
# Per-emotion mask density statistics (thesis Table 13, Figure 19)
# =============================================================================

def compute_landmark_mask_stats(model, loader, class_names, device='cuda',
                                 threshold=0.25, fa=None, max_images=None):
    """Compute per-emotion mask statistics using landmarks (thesis Table 13).

    For each image, computes the average mask activation inside the eye region
    and inside the mouth region, using the 68 landmarks to define those regions.

    Args:
        model: MaskCAM instance
        loader: DataLoader
        class_names: list of emotion strings
        device: torch device
        threshold: threshold for "eyes preserved" binary decision
        fa: an initialized face_alignment.FaceAlignment instance (optional)
        max_images: cap on images processed (optional)

    Returns:
        (eye_means, mouth_means, eyes_kept_pct) — three dicts keyed by class index
    """
    from skimage.transform import resize
    from tqdm import tqdm

    if fa is None:
        import face_alignment
        fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            flip_input=False,
            device=device,
        )

    num_classes = len(class_names)
    eye_means = {c: [] for c in range(num_classes)}
    mouth_means = {c: [] for c in range(num_classes)}
    eye_kept = {c: [] for c in range(num_classes)}

    model.eval()
    processed = 0

    for images, labels in tqdm(loader, desc="Landmark mask stats"):
        for i in range(images.size(0)):
            if max_images is not None and processed >= max_images:
                break

            img_tensor = images[i].unsqueeze(0).to(device)
            label = labels[i].item()

            # Detect landmarks on a 224x224 upscaled version
            img_np = img_tensor.cpu().clone().squeeze().numpy()
            img_vis = (img_np * 0.5 + 0.5) * 255.0
            img_vis = np.clip(img_vis, 0, 255).astype(np.uint8)
            img_224 = resize(img_vis, (224, 224),
                             preserve_range=True, anti_aliasing=True).astype(np.uint8)
            img_224_rgb = np.stack([img_224] * 3, axis=-1)

            try:
                preds_lm = fa.get_landmarks(img_224_rgb)
                landmarks = preds_lm[0] if preds_lm else None
            except Exception:
                landmarks = None

            # Get the mask from the model
            with torch.no_grad():
                _, _, mask = model(img_tensor, return_cam=True, return_mask=True)
            mask_np = mask[0, 0].cpu().numpy()
            mask_up = cv2.resize(mask_np, (48, 48), interpolation=cv2.INTER_LINEAR)

            if landmarks is not None:
                scale = 48.0 / 224.0
                lm = landmarks * scale
                try:
                    left_eye = _landmark_to_mask(lm[36:42])
                    right_eye = _landmark_to_mask(lm[42:48])
                    eye_mask = np.clip(left_eye + right_eye, 0, 1)
                    mouth_mask = _landmark_to_mask(lm[48:68])

                    eye_pixels = eye_mask.sum()
                    mouth_pixels = mouth_mask.sum()
                    eye_mean = ((mask_up * eye_mask).sum() / eye_pixels
                                if eye_pixels > 0 else 0.0)
                    mouth_mean = ((mask_up * mouth_mask).sum() / mouth_pixels
                                  if mouth_pixels > 0 else 0.0)
                except Exception:
                    landmarks = None

            if landmarks is None:
                eye_mask = heuristic_eyes_mouth()
                mouth_mask = np.zeros((H, W), dtype=np.float32)
                mouth_mask[2 * H // 3:, W // 4:3 * W // 4] = 1
                eye_mean = ((mask_up * eye_mask).sum()
                            / max(eye_mask.sum(), 1))
                mouth_mean = ((mask_up * mouth_mask).sum()
                              / max(mouth_mask.sum(), 1))

            eye_means[label].append(eye_mean)
            mouth_means[label].append(mouth_mean)
            eye_kept[label].append(1.0 if eye_mean >= threshold else 0.0)
            processed += 1

        if max_images is not None and processed >= max_images:
            break

    eye_result = {c: float(np.mean(eye_means[c])) for c in eye_means if eye_means[c]}
    mouth_result = {c: float(np.mean(mouth_means[c])) for c in mouth_means if mouth_means[c]}
    kept_result = {
        class_names[c]: float(100.0 * np.mean(eye_kept[c]))
        for c in eye_kept if eye_kept[c]
    }
    return eye_result, mouth_result, kept_result


def _landmark_to_mask(points, shape=(48, 48)):
    """Small helper: convert landmark points to a binary mask."""
    mask = np.zeros(shape, dtype=np.float32)
    points = points.astype(int)
    points[:, 0] = np.clip(points[:, 0], 0, shape[1] - 1)
    points[:, 1] = np.clip(points[:, 1], 0, shape[0] - 1)
    rr, cc = polygon(points[:, 1], points[:, 0], shape=shape)
    mask[rr, cc] = 1.0
    return mask