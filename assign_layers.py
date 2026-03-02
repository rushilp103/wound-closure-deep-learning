"""
Layer assignment script for wound-healing assay.
Reads objects CSV and mask stack, runs edge and centroid methods, writes CSV with layer IDs.
"""
import os
import argparse

import pandas as pd
import tifffile

from wound_utils import get_wound_masks_from_stack
from layer_assignment import assign_all_methods
from pipeline_config import (
    masks_tracking_layers_path,
    objects_csv_layers_path,
    objects_with_layers_csv_path,
)

# --- CONFIGURATION ---
# Micrometers per pixel (set from your microscope/calibration). If unknown, use 1.0 and layers are in "pixel-width" units.
UM_PER_PIXEL = 1.0

MASKS_PATH = masks_tracking_layers_path
OBJECTS_CSV = objects_csv_layers_path
OUTPUT_CSV = objects_with_layers_csv_path

DEFAULT_LAYER_WIDTH_UM = 49.0


def main():
    parser = argparse.ArgumentParser(description="Assign layer IDs (edge, centroid) to object detections.")
    parser.add_argument("--objects", default=OBJECTS_CSV, help="Path to objects CSV (x, y, t, ...)")
    parser.add_argument("--masks", default=MASKS_PATH, help="Path to masks stack TIF (T, Y, X)")
    parser.add_argument("--output", default=OUTPUT_CSV, help="Output CSV with layer columns")
    parser.add_argument("--um-per-pixel", type=float, default=UM_PER_PIXEL, help="Micrometers per pixel")
    parser.add_argument("--layer-width", type=float, default=DEFAULT_LAYER_WIDTH_UM, help="Layer width in µm")
    parser.add_argument("--smooth-wound", type=float, default=2.0, metavar="SIGMA", help="Gaussian sigma (px) to smooth wound boundary for consistent layer width (default: 2)")
    parser.add_argument("--no-smooth-wound", action="store_true", help="Disable wound smoothing (use raw wound mask)")
    parser.add_argument("--closing-radius", type=int, default=10, help="Morphological closing radius (px) to fill intercellular gaps (default: 10)")
    parser.add_argument("--opening-radius", type=int, default=5, help="Morphological opening radius (px) to remove small debris from wound (default: 5)")
    parser.add_argument("--erosion-radius", type=int, default=3, help="Morphological erosion radius (px) to pull back wound boundary (default: 3)")
    args = parser.parse_args()

    smooth_sigma = None if args.no_smooth_wound else args.smooth_wound

    print("Loading objects...")
    objects_df = pd.read_csv(args.objects)
    print(f"  > {len(objects_df)} rows, columns: {list(objects_df.columns)}")

    print("Loading mask stack...")
    masks_stack = tifffile.imread(args.masks)
    print(f"  > Shape: {masks_stack.shape}")

    print("Computing wound masks per frame...")
    print(f"  > Using morphological operations: closing={args.closing_radius}px, opening={args.opening_radius}px, erosion={args.erosion_radius}px")
    wound_masks = get_wound_masks_from_stack(
        masks_stack, 
        closing_radius=args.closing_radius,
        opening_radius=args.opening_radius,
        erosion_radius=args.erosion_radius,
    )
    print(f"  > {len(wound_masks)} wound masks")

    print("Assigning layers (edge, centroid)...")
    if smooth_sigma is not None:
        print(f"  > Smoothing wound boundary (sigma={smooth_sigma} px)")
    result = assign_all_methods(
        objects_df,
        wound_masks,
        um_per_pixel=args.um_per_pixel,
        layer_width_um=args.layer_width,
        smooth_wound_sigma_px=smooth_sigma,
    )

    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    result.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")

    # Summary
    for col in ["layer_edge", "layer_centroid"]:
        if col in result.columns:
            valid = result[result[col] >= 0]
            print(f"  {col}: {valid[col].nunique()} unique layers, range [{valid[col].min()}, {valid[col].max()}]")


if __name__ == "__main__":
    main()
