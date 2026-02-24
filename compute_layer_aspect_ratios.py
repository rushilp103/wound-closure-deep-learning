"""
Compute per-layer-per-frame average cell aspect ratio.
Reads objects_with_layers.csv, samples cells per (t, layer), and writes
layer_aspect_ratios.csv with mean_aspect_ratio, std_aspect_ratio, n_cells.
Run assign_layers.py first to generate the input CSV.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

LAYER_COLUMNS = ['layer_edge', 'layer_centroid']
REQUIRED_COLUMNS = ['t', 'obj_id', 'major_axis_length', 'minor_axis_length']
DEFAULT_CSV = "/Users/Rushilp/Projects/VSCode/Image-Analysis-and-Segmentation-of-Wound-Gap-Closure/Cellpose-SAM Results/ctrl-1_objects_with_layers.csv"


def _default_output_path(csv_path: str) -> str:
    """Derive default output path from input CSV path."""
    if '_objects_with_layers.csv' in csv_path:
        return csv_path.replace('_objects_with_layers.csv', '_layer_aspect_ratios.csv')
    base, ext = os.path.splitext(csv_path)
    return f"{base}_layer_aspect_ratios{ext}"


def compute_layer_aspect_ratios(
    csv_path: str,
    output_path: str,
    layer_column: str = 'layer_edge',
    max_sample: int = 200,
    min_cells: int = 3,
    seed: int = 42,
    include_wound: bool = False,
) -> None:
    """
    Read objects_with_layers CSV, compute per-cell aspect ratio, then for each
    (t, layer) sample up to max_sample cells and write mean_aspect_ratio, std, n_cells.
    """
    if layer_column not in LAYER_COLUMNS:
        print(f"Error: --layer must be one of {LAYER_COLUMNS}.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)

    missing = [c for c in REQUIRED_COLUMNS + [layer_column] if c not in df.columns]
    if missing:
        print(f"Error: CSV is missing columns: {missing}. Run assign_layers.py and ensure masks_to_objects exported major/minor axis.", file=sys.stderr)
        sys.exit(1)

    # Per-cell aspect ratio; drop invalid
    df = df.copy()
    df['aspect_ratio'] = df['major_axis_length'] / df['minor_axis_length']
    valid = (df['minor_axis_length'] > 0) & df['major_axis_length'].notna() & df['minor_axis_length'].notna()
    df = df.loc[valid].copy()

    if not include_wound:
        df = df[df[layer_column] != -1].copy()

    # Group by (t, layer), sample, aggregate
    rows = []

    for (t, layer), grp in df.groupby(['t', layer_column]):
        n = len(grp)
        if n < min_cells:
            continue
        if n > max_sample:
            grp = grp.sample(n=max_sample, random_state=seed)
        ar = grp['aspect_ratio']
        rows.append({
            't': t,
            layer_column: layer,
            'mean_aspect_ratio': float(ar.mean()),
            'std_aspect_ratio': float(ar.std()) if len(ar) > 1 else np.nan,
            'n_cells': len(grp),
        })

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        print("Warning: No (t, layer) groups had enough valid cells. Output CSV will be empty.")

    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_df.to_csv(output_path, index=False)
    print(f"Saved {len(out_df)} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-layer-per-frame average cell aspect ratio from objects_with_layers.csv."
    )
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to objects_with_layers CSV")
    parser.add_argument("--output", default=None, help="Output CSV path (default: derived from --csv)")
    parser.add_argument(
        "--layer",
        default="layer_edge",
        choices=LAYER_COLUMNS,
        help="Layer column to use (default: layer_edge)",
    )
    parser.add_argument(
        "--max-sample",
        type=int,
        default=200,
        help="Max cells to sample per (t, layer) (default: 200)",
    )
    parser.add_argument(
        "--min-cells",
        type=int,
        default=3,
        help="Min cells required to report mean (default: 3)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42)")
    parser.add_argument(
        "--include-wound",
        action="store_true",
        help="Include layer == -1 (wound) in layer aspect ratios",
    )
    args = parser.parse_args()

    output_path = args.output if args.output is not None else _default_output_path(args.csv)
    compute_layer_aspect_ratios(
        csv_path=args.csv,
        output_path=output_path,
        layer_column=args.layer,
        max_sample=args.max_sample,
        min_cells=args.min_cells,
        seed=args.seed,
        include_wound=args.include_wound,
    )


if __name__ == "__main__":
    main()
