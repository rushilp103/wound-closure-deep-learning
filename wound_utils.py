"""
Wound segmentation for wound-healing assays.
Uses the mask stack: background (0) = potential wound; largest central hole = wound.
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.measure import label


def smooth_wound_mask(wound_mask: np.ndarray, sigma_px: float = 2.0) -> np.ndarray:
    """
    Smooth the wound boundary by Gaussian blur and re-threshold.
    Reduces jaggedness so distance-from-edge layers have more consistent width.

    Args:
        wound_mask: Binary (H, W), 1 = wound, 0 = not wound.
        sigma_px: Gaussian sigma in pixels; larger = smoother boundary. Use 0 or skip to disable.

    Returns:
        Binary mask same shape, 1 = wound, 0 = not wound.
    """
    if sigma_px <= 0 or wound_mask.sum() == 0:
        return wound_mask
    blurred = gaussian_filter(wound_mask.astype(np.float64), sigma=sigma_px)
    return (blurred >= 0.5).astype(np.uint8)


def get_wound_mask_for_frame(frame_mask: np.ndarray) -> np.ndarray:
    """
    From a single-frame segmentation mask (0 = background, >0 = cell labels),
    identify the central wound as the largest connected component of background
    that overlaps the central region of the image.

    Returns:
        Binary mask (H, W), 1 = wound, 0 = not wound.
    """
    h, w = frame_mask.shape
    center_y, center_x = h / 2, w / 2

    # Background = pixels with no cell (0)
    background = (frame_mask == 0).astype(np.uint8)

    # Label connected components of background (4-connectivity to avoid diagonal bridges)
    labeled = label(background, connectivity=2)

    if labeled.max() == 0:
        return np.zeros_like(frame_mask, dtype=np.uint8)

    # Find the component that contains the image center (or is closest to it)
    center_r, center_c = int(np.round(center_y)), int(np.round(center_x))
    center_r = np.clip(center_r, 0, h - 1)
    center_c = np.clip(center_c, 0, w - 1)

    center_label = labeled[center_r, center_c]
    if center_label > 0:
        wound_mask = (labeled == center_label).astype(np.uint8)
        return wound_mask

    # Center is not in any background component (e.g. cell there); pick largest central hole
    # "Central" = centroid of component is within central 50% of image
    best_label = 0
    best_score = -1
    for idx in range(1, labeled.max() + 1):
        comp = (labeled == idx)
        if not comp.any():
            continue
        ys, xs = np.where(comp)
        cy, cx = ys.mean(), xs.mean()
        # Prefer component whose centroid is near image center
        dist_to_center = (cy - center_y) ** 2 + (cx - center_x) ** 2
        area = comp.sum()
        # Score: larger area and closer to center
        score = area - 0.001 * dist_to_center
        if score > best_score:
            best_score = score
            best_label = idx

    if best_label == 0:
        return np.zeros_like(frame_mask, dtype=np.uint8)

    wound_mask = (labeled == best_label).astype(np.uint8)
    return wound_mask


def get_wound_masks_from_stack(masks_stack: np.ndarray):
    """
    Compute wound binary mask for each frame in the mask stack.

    masks_stack: (T, H, W), integer labels per frame.

    Returns:
        List of T arrays, each (H, W) with 1 = wound, 0 = not wound.
    """
    return [get_wound_mask_for_frame(masks_stack[t]) for t in range(masks_stack.shape[0])]


def wound_centroid_and_radius(wound_mask: np.ndarray) -> tuple:
    """
    Compute wound centroid (y, x) in pixel coordinates and approximate
    radius (sqrt(area/pi)) for circular approximation.

    Returns:
        (cy, cx, radius) in pixels; (0, 0, 0) if wound is empty.
    """
    if wound_mask.sum() == 0:
        return 0.0, 0.0, 0.0
    ys, xs = np.where(wound_mask > 0)
    cy, cx = ys.mean(), xs.mean()
    area = len(ys)
    radius = np.sqrt(area / np.pi)
    return cy, cx, radius
