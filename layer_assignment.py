"""
Layer assignment for wound-healing assays.
Two methods: distance from wound edge (primary), distance from wound centroid.
"""
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

from wound_utils import smooth_wound_mask, wound_centroid_and_radius

# Default layer width in µm (as in plan)
LAYER_WIDTH_UM = 49.0
MAX_LAYER_ID = 50  # Cap far-away cells so we don't get huge layer IDs


def _sample_distance_at_point(distance_map: np.ndarray, y_px: float, x_px: float) -> float:
    """Sample distance map at (y, x) in pixel coords; uses bilinear-style rounding."""
    h, w = distance_map.shape
    r, c = int(np.round(y_px)), int(np.round(x_px))
    r = np.clip(r, 0, h - 1)
    c = np.clip(c, 0, w - 1)
    return float(distance_map[r, c])


def assign_layers_edge(
    objects_df: pd.DataFrame,
    wound_masks: list,
    um_per_pixel: float,
    layer_width_um: float = LAYER_WIDTH_UM,
    max_layer: int = MAX_LAYER_ID,
    smooth_wound_sigma_px: float | None = 2.0,
) -> pd.DataFrame:
    """
    Assign layer_id = floor(distance_from_wound_edge_um / layer_width_um).
    Distance is shortest distance from cell centroid to wound boundary.
    Cells inside the wound get layer_id = -1.

    objects_df must have columns: x, y, t (pixel coordinates).
    wound_masks: list of (H,W) binary arrays, 1 = wound, 0 = not wound.
    smooth_wound_sigma_px: If not None, smooth the wound mask with this Gaussian sigma (px) before distance transform.
    """
    out = objects_df.copy()
    dist_col = "distance_from_edge_um"
    layer_col = "layer_edge"
    out[dist_col] = np.nan
    out[layer_col] = -2  # sentinel for unassigned

    for t in objects_df["t"].unique():
        frame_df = objects_df[objects_df["t"] == t]
        if t >= len(wound_masks):
            continue
        wound = wound_masks[t]
        if wound.sum() == 0:
            continue
        if smooth_wound_sigma_px is not None and smooth_wound_sigma_px > 0:
            wound = smooth_wound_mask(wound, sigma_px=smooth_wound_sigma_px)
            if wound.sum() == 0:
                continue
        # Distance from each non-wound pixel to nearest wound pixel (edge)
        not_wound = (wound == 0).astype(np.float32)
        dist_to_edge_px = distance_transform_edt(not_wound)

        for idx in frame_df.index:
            x_px = objects_df.loc[idx, "x"]
            y_px = objects_df.loc[idx, "y"]
            r, c = int(np.round(y_px)), int(np.round(x_px))
            inside = (
                r >= 0
                and r < wound.shape[0]
                and c >= 0
                and c < wound.shape[1]
                and wound[r, c] > 0
            )
            if inside:
                out.loc[idx, dist_col] = 0.0
                out.loc[idx, layer_col] = -1
            else:
                d_px = _sample_distance_at_point(dist_to_edge_px, y_px, x_px)
                d_um = d_px * um_per_pixel
                out.loc[idx, dist_col] = d_um
                lid = int(np.floor(d_um / layer_width_um))
                out.loc[idx, layer_col] = min(lid, max_layer) if lid >= 0 else -1

    return out


def assign_layers_centroid(
    objects_df: pd.DataFrame,
    wound_masks: list,
    um_per_pixel: float,
    layer_width_um: float = LAYER_WIDTH_UM,
    max_layer: int = MAX_LAYER_ID,
    smooth_wound_sigma_px: float | None = 2.0,
) -> pd.DataFrame:
    """
    Assign layer_id = floor((distance_from_center_um - wound_radius_um) / layer_width_um).
    Cells inside the wound (dist < radius) get layer_id = -1.

    objects_df must have columns: x, y, t.
    wound_masks: list of (H,W) binary arrays.
    smooth_wound_sigma_px: If not None, smooth the wound mask before computing centroid/radius (consistency with edge method).
    """
    out = objects_df.copy()
    dist_col = "distance_from_center_um"
    layer_col = "layer_centroid"
    out[dist_col] = np.nan
    out[layer_col] = -2

    for t in objects_df["t"].unique():
        frame_df = objects_df[objects_df["t"] == t]
        if t >= len(wound_masks):
            continue
        wound = wound_masks[t]
        if smooth_wound_sigma_px is not None and smooth_wound_sigma_px > 0:
            wound = smooth_wound_mask(wound, sigma_px=smooth_wound_sigma_px)
        cy, cx, radius_px = wound_centroid_and_radius(wound)
        if radius_px <= 0:
            continue
        radius_um = radius_px * um_per_pixel

        for idx in frame_df.index:
            x_px = objects_df.loc[idx, "x"]
            y_px = objects_df.loc[idx, "y"]
            d_center_px = np.sqrt((x_px - cx) ** 2 + (y_px - cy) ** 2)
            d_center_um = d_center_px * um_per_pixel
            d_outside_um = d_center_um - radius_um
            out.loc[idx, dist_col] = d_center_um
            if d_outside_um < 0:
                out.loc[idx, layer_col] = -1
            else:
                lid = int(np.floor(d_outside_um / layer_width_um))
                out.loc[idx, layer_col] = min(lid, max_layer)

    return out


def assign_all_methods(
    objects_df: pd.DataFrame,
    wound_masks: list,
    um_per_pixel: float,
    layer_width_um: float = LAYER_WIDTH_UM,
    smooth_wound_sigma_px: float | None = 2.0,
) -> pd.DataFrame:
    """
    Run edge and centroid methods and return a single DataFrame with columns:
    layer_edge, layer_centroid, distance_from_edge_um, distance_from_center_um.
    smooth_wound_sigma_px: If not None, smooth wound masks before edge/centroid methods for consistent layer width.
    """
    df = assign_layers_edge(
        objects_df, wound_masks, um_per_pixel, layer_width_um,
        smooth_wound_sigma_px=smooth_wound_sigma_px,
    )
    df = assign_layers_centroid(
        df, wound_masks, um_per_pixel, layer_width_um,
        smooth_wound_sigma_px=smooth_wound_sigma_px,
    )
    return df
