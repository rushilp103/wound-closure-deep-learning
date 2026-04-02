"""
Layer assignment for wound-healing assays.
Layers from wound centroid: distance from cell to centroid minus equivalent wound radius.
"""
import numpy as np
import pandas as pd

from wound_utils import smooth_wound_mask, wound_centroid_and_radius

# Default layer width in µm (as in plan)
LAYER_WIDTH_UM = 49.0
MAX_LAYER_ID = 50  # Cap far-away cells so we don't get huge layer IDs


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
    smooth_wound_sigma_px: If not None, smooth the wound mask before computing centroid/radius.
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
