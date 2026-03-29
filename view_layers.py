"""
View layer-assigned detections in Napari.
Loads raw image stack, mask stack, and objects_with_layers.csv; displays each cell mask
colored by centroid-based layer. Run assign_layers.py first to generate the CSV.
"""
import argparse
import sys

import napari
import numpy as np
import pandas as pd
import tifffile
from skimage.io import imread

from wound_utils import get_wound_masks_from_stack, smooth_wound_mask
from pipeline_config import (
    input_tif_path,
    objects_with_layers_csv_path,
    masks_tracking_layers_path,
)

# --- CONFIGURATION ---
RAW_TIFF_PATH = input_tif_path
OBJECTS_WITH_LAYERS_CSV = objects_with_layers_csv_path
MASKS_PATH = masks_tracking_layers_path
LAYER_COLUMN = "layer_centroid"

LAYER_COLUMNS = ["layer_centroid"]


def _default_masks_path_from_csv(csv_path: str) -> str:
    """Derive masks path from objects_with_layers CSV path."""
    if "_objects_with_layers.csv" in csv_path:
        return csv_path.replace("_objects_with_layers.csv", "_masks_TRACKING.tif")
    return csv_path.replace(".csv", "_masks_TRACKING.tif")


def view_layers(
    raw_path: str,
    csv_path: str,
    masks_path: str,
    smooth_wound_sigma_px: float | None = None,
    closing_radius: int = 10,
    opening_radius: int = 5,
    erosion_radius: int = 3,
):
    print("Loading Napari viewer...")
    viewer = napari.Viewer()

    try:
        print(f"Loading raw image from {raw_path}...")
        raw_image = imread(raw_path)
        viewer.add_image(raw_image, name="Raw Image", blending="additive", colormap="gray")
    except Exception as e:
        print(f"Error loading raw image: {e}")
        raise

    print(f"Loading mask stack from {masks_path}...")
    masks = tifffile.imread(masks_path)
    if masks.ndim != 3:
        print(f"Error: Mask stack must be (T, Y, X), got shape {masks.shape}.", file=sys.stderr)
        sys.exit(1)
    print(f"  > Shape: {masks.shape}")

    print(f"Loading layer-assigned objects from {csv_path}...")
    df = pd.read_csv(csv_path)

    missing = [c for c in LAYER_COLUMNS if c not in df.columns]
    if missing:
        print(f"Error: CSV is missing layer columns: {missing}. Run assign_layers.py first.", file=sys.stderr)
        sys.exit(1)

    for c in ["x", "y", "t", "obj_id"]:
        if c not in df.columns:
            print(f"Error: CSV is missing required column '{c}'.", file=sys.stderr)
            sys.exit(1)

    # Map (t, obj_id) -> layer value
    t_obj_to_layer = df.set_index(["t", "obj_id"])[LAYER_COLUMN]

    # Remap layer values to positive integers for Napari labels (0 = background)
    unique_layers = sorted(t_obj_to_layer.unique())
    layer_to_label = {v: i + 1 for i, v in enumerate(unique_layers)}
    wound_label = max(layer_to_label.values()) + 1

    wound_masks = get_wound_masks_from_stack(
        masks,
        closing_radius=closing_radius,
        opening_radius=opening_radius,
        erosion_radius=erosion_radius,
    )
    if smooth_wound_sigma_px is not None and smooth_wound_sigma_px > 0:
        wound_masks = [smooth_wound_mask(w, sigma_px=smooth_wound_sigma_px) for w in wound_masks]

    print("Building per-cell layer mask...")
    T, H, W = masks.shape
    layer_mask = np.zeros((T, H, W), dtype=np.int32)
    for t in range(T):
        frame_mask = masks[t]
        frame_df = df[df["t"] == t]
        for _, row in frame_df.iterrows():
            obj_id = int(row["obj_id"])
            layer_val = row[LAYER_COLUMN]
            if np.isnan(layer_val):
                continue
            label = layer_to_label[layer_val]
            layer_mask[t][frame_mask == obj_id] = label
    for t in range(T):
        layer_mask[t][wound_masks[t] > 0] = wound_label
    print(f"  > Wound gap added to layer mask (label {wound_label})")

    viewer.add_labels(layer_mask, name="Layers (cell masks)")

    print("Viewer open! Each cell is colored by its layer (labels layer).")
    napari.run()


def main():
    parser = argparse.ArgumentParser(
        description="View layer-assigned cell masks in Napari. Run assign_layers.py first."
    )
    parser.add_argument("--raw", default=RAW_TIFF_PATH, help="Path to raw TIFF stack")
    parser.add_argument("--csv", default=OBJECTS_WITH_LAYERS_CSV, help="Path to objects_with_layers CSV")
    parser.add_argument(
        "--masks",
        default=None,
        help="Path to mask stack TIF (T,Y,X). Default: derived from --csv (e.g. ..._masks_TRACKING.tif)",
    )
    parser.add_argument(
        "--smooth-wound",
        type=float,
        default=None,
        metavar="SIGMA",
        help="Smooth wound mask with this Gaussian sigma (px); use same as assign_layers.py if desired",
    )
    parser.add_argument(
        "--closing-radius",
        type=int,
        default=10,
        help="Morphological closing radius (px) to fill intercellular gaps (default: 10)",
    )
    parser.add_argument(
        "--opening-radius",
        type=int,
        default=5,
        help="Morphological opening radius (px) to remove small debris from wound (default: 5)",
    )
    parser.add_argument(
        "--erosion-radius",
        type=int,
        default=3,
        help="Morphological erosion radius (px) to pull back wound boundary (default: 3)",
    )
    args = parser.parse_args()

    masks_path = args.masks if args.masks is not None else _default_masks_path_from_csv(args.csv)

    view_layers(
        raw_path=args.raw,
        csv_path=args.csv,
        masks_path=masks_path,
        smooth_wound_sigma_px=args.smooth_wound,
        closing_radius=args.closing_radius,
        opening_radius=args.opening_radius,
        erosion_radius=args.erosion_radius,
    )


if __name__ == "__main__":
    main()
