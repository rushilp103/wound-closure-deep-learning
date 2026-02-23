"""
View layer-assigned detections in Napari.
Loads raw image stack, mask stack, and objects_with_layers.csv; displays each cell mask
colored by its layer (edge or centroid). Run assign_layers.py first to generate the CSV.
Optionally draws 49 µm layer-boundary contours when using layer_edge.
"""
import argparse
import sys

import napari
import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import distance_transform_edt
from skimage.io import imread
from skimage.measure import find_contours

from wound_utils import get_wound_masks_from_stack, smooth_wound_mask

# --- CONFIGURATION ---
RAW_TIFF_PATH = "/Users/Rushilp/Projects/VSCode/Image-Analysis-and-Segmentation-of-Wound-Gap-Closure/Data Sets/ctrl-1.tif"
OBJECTS_WITH_LAYERS_CSV = "/Users/Rushilp/Projects/VSCode/Image-Analysis-and-Segmentation-of-Wound-Gap-Closure/Cellpose-SAM Results/ctrl-1_objects_with_layers.csv"
MASKS_PATH = "/Users/Rushilp/Projects/VSCode/Image-Analysis-and-Segmentation-of-Wound-Gap-Closure/Cellpose-SAM Results/ctrl-1_masks_TRACKING.tif"
LAYER_COLUMN = 'layer_edge'  # 'layer_edge' | 'layer_centroid'

LAYER_COLUMNS = ['layer_edge', 'layer_centroid']


def _default_masks_path_from_csv(csv_path: str) -> str:
    """Derive masks path from objects_with_layers CSV path."""
    if '_objects_with_layers.csv' in csv_path:
        return csv_path.replace('_objects_with_layers.csv', '_masks_TRACKING.tif')
    return csv_path.replace('.csv', '_masks_TRACKING.tif')


def _compute_layer_boundary_shapes(
    wound_masks: list,
    um_per_pixel: float,
    layer_width_um: float,
    num_boundaries: int = 5,
) -> list:
    """Build (t, y, x) contour shapes at 49, 98, ... µm for Napari."""
    shapes = []
    for t, wound in enumerate(wound_masks):
        if wound.sum() == 0:
            continue
        not_wound = (wound == 0).astype(np.float32)
        dist_px = distance_transform_edt(not_wound)
        dist_um = dist_px * um_per_pixel
        for k in range(1, num_boundaries + 1):
            boundary_um = k * layer_width_um
            try:
                contours = find_contours(dist_um, level=boundary_um)
            except Exception:
                continue
            for contour in contours:
                # contour is (N, 2) in (row, col) = (y, x)
                n = len(contour)
                t_col = np.full((n, 1), float(t))
                shape_3d = np.hstack([t_col, contour])
                shapes.append(shape_3d)
    return shapes


def view_layers(
    raw_path: str,
    csv_path: str,
    masks_path: str,
    layer_column: str = LAYER_COLUMN,
    um_per_pixel: float = 1.0,
    layer_width_um: float = 49.0,
    smooth_wound_sigma_px: float | None = None,
    show_boundaries: bool = True,
    max_boundaries: int = 5,
):
    print("Loading Napari viewer...")
    viewer = napari.Viewer()

    try:
        print(f"Loading raw image from {raw_path}...")
        raw_image = imread(raw_path)
        viewer.add_image(raw_image, name='Raw Image', blending='additive', colormap='gray')
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

    for c in ['x', 'y', 't', 'obj_id']:
        if c not in df.columns:
            print(f"Error: CSV is missing required column '{c}'.", file=sys.stderr)
            sys.exit(1)

    if layer_column not in df.columns:
        print(f"Error: Layer column '{layer_column}' not in CSV. Choose one of {LAYER_COLUMNS}.", file=sys.stderr)
        sys.exit(1)

    # Map (t, obj_id) -> layer value
    t_obj_to_layer = df.set_index(['t', 'obj_id'])[layer_column]

    # Remap layer values to positive integers for Napari labels (0 = background)
    unique_layers = sorted(t_obj_to_layer.unique())
    layer_to_label = {v: i + 1 for i, v in enumerate(unique_layers)}
    wound_label = max(layer_to_label.values()) + 1

    # Compute wound masks once (used for layer mask and optionally for boundary contours)
    wound_masks = get_wound_masks_from_stack(masks)
    if smooth_wound_sigma_px is not None and smooth_wound_sigma_px > 0:
        wound_masks = [smooth_wound_mask(w, sigma_px=smooth_wound_sigma_px) for w in wound_masks]

    print("Building per-cell layer mask...")
    T, H, W = masks.shape
    layer_mask = np.zeros((T, H, W), dtype=np.int32)
    for t in range(T):
        frame_mask = masks[t]
        frame_df = df[df['t'] == t]
        for _, row in frame_df.iterrows():
            obj_id = int(row['obj_id'])
            layer_val = row[layer_column]
            if np.isnan(layer_val):
                continue
            label = layer_to_label[layer_val]
            layer_mask[t][frame_mask == obj_id] = label
    for t in range(T):
        layer_mask[t][wound_masks[t] > 0] = wound_label
    print(f"  > Wound gap added to layer mask (label {wound_label})")

    viewer.add_labels(layer_mask, name='Layers (cell masks)')

    if show_boundaries and layer_column == 'layer_edge':
        print("Computing layer boundaries...")
        boundary_shapes = _compute_layer_boundary_shapes(
            wound_masks, um_per_pixel, layer_width_um, num_boundaries=max_boundaries
        )
        if boundary_shapes:
            viewer.add_shapes(
                boundary_shapes,
                shape_type='path',
                edge_color='cyan',
                edge_width=2,
                name=f'Layer boundaries ({layer_width_um:.0f} µm)',
            )
            print(f"  > Added {len(boundary_shapes)} boundary contours at {layer_width_um:.0f}, {2*layer_width_um:.0f}, ... µm")
        else:
            print("  > No boundary contours found (empty wound masks or no contours)")

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
        "--layer",
        default=LAYER_COLUMN,
        choices=LAYER_COLUMNS,
        help="Layer column to use for cell color (default: layer_edge)",
    )
    parser.add_argument("--um-per-pixel", type=float, default=1.0, help="µm per pixel (for boundary contours)")
    parser.add_argument("--layer-width", type=float, default=49.0, help="Layer width in µm (for boundary contours)")
    parser.add_argument(
        "--smooth-wound",
        type=float,
        default=None,
        metavar="SIGMA",
        help="Smooth wound mask with this Gaussian sigma (px) before drawing boundaries; use same as assign_layers.py",
    )
    parser.add_argument(
        "--no-boundaries",
        action="store_true",
        help="Do not draw 49 µm boundary contours (only when using layer_edge)",
    )
    parser.add_argument(
        "--max-boundaries",
        type=int,
        default=5,
        help="Number of layer boundaries to draw at 49, 98, ... µm (default: 5)",
    )
    args = parser.parse_args()

    masks_path = args.masks if args.masks is not None else _default_masks_path_from_csv(args.csv)

    view_layers(
        raw_path=args.raw,
        csv_path=args.csv,
        masks_path=masks_path,
        layer_column=args.layer,
        um_per_pixel=args.um_per_pixel,
        layer_width_um=args.layer_width,
        smooth_wound_sigma_px=args.smooth_wound,
        show_boundaries=not args.no_boundaries,
        max_boundaries=args.max_boundaries,
    )


if __name__ == "__main__":
    main()
