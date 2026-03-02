"""
Generate visualizations for layer aspect ratios and cell size.
Requires objects_with_layers.csv (from assign_layers.py) with major_axis_length, minor_axis_length, area.
"""
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline_config import objects_with_layers_csv_path

LAYER_COLUMNS = ['layer_edge', 'layer_centroid']
REQUIRED_ASPECT = ['t', 'major_axis_length', 'minor_axis_length']
REQUIRED_SIZE = ['t', 'area']
DEFAULT_CSV = objects_with_layers_csv_path


def _prepare_size_df(
    csv_path: str,
    layer_column: str,
    frame: int,
    include_wound: bool,
) -> pd.DataFrame:
    """Load CSV, filter to one frame, optionally exclude layer == -1. Requires area, t, layer column."""
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_SIZE + [layer_column] if c not in df.columns]
    if missing:
        print(f"Error: CSV is missing columns: {missing}.", file=sys.stderr)
        sys.exit(1)
    df = df[df['t'] == frame].copy()
    if df.empty:
        print(f"Error: No rows for frame {frame}.", file=sys.stderr)
        sys.exit(1)
    if not include_wound:
        df = df[df[layer_column] != -1].copy()
    return df


def _prepare_aspect_ratio_df(
    csv_path: str,
    layer_column: str,
    include_wound: bool,
) -> pd.DataFrame:
    """Load CSV, compute per-cell aspect_ratio, drop invalid. Optionally exclude layer == -1."""
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_ASPECT + [layer_column] if c not in df.columns]
    if missing:
        print(f"Error: CSV is missing columns: {missing}.", file=sys.stderr)
        sys.exit(1)
    df = df.copy()
    df['aspect_ratio'] = df['major_axis_length'] / df['minor_axis_length']
    valid = (df['minor_axis_length'] > 0) & df['major_axis_length'].notna() & df['minor_axis_length'].notna()
    df = df.loc[valid].copy()
    if not include_wound:
        df = df[df[layer_column] != -1].copy()
    return df


def plot_size_vs_layer(
    csv_path: str,
    frame: int,
    layer_column: str = 'layer_edge',
    include_wound: bool = False,
    output_path: str | None = None,
    dpi: int = 150,
) -> None:
    """
    Plot cell size (area) vs layer for a single frame using box-and-whisker plots.
    X = layer ID, Y = area; one box per layer at the chosen frame.
    """
    print(f"Loading {csv_path}...")
    df = _prepare_size_df(csv_path, layer_column, frame, include_wound)
    if df.empty:
        print("No valid rows after filtering. Exiting.")
        return

    layers = sorted(df[layer_column].dropna().unique())
    if not layers:
        print("No layers found. Exiting.")
        return

    data_by_layer = [df[df[layer_column] == layer]['area'].values for layer in layers]
    positions = range(len(layers))
    fig, ax = plt.subplots(figsize=(max(6, len(layers) * 0.5), 5))
    bp = ax.boxplot(
        data_by_layer,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker='o', markersize=2, alpha=0.4),
    )
    for patch in bp['boxes']:
        patch.set_facecolor('lightsteelblue')
        patch.set_alpha(0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels([int(l) for l in layers])
    ax.set_xlabel('Layer')
    ax.set_ylabel('Size (area, px²)')
    ax.set_title(f'Size vs layer at frame {frame} ({layer_column})')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved figure to {output_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_aspect_ratio_over_time_one_layer(
    csv_path: str,
    layer_id: int,
    layer_column: str = 'layer_edge',
    include_wound: bool = False,
    output_path: str | None = None,
    dpi: int = 150,
) -> None:
    """
    Plot aspect ratio over time for a single layer using box-and-whisker plots.
    X = time (frame), Y = aspect ratio; one box per time point for the chosen layer.
    """
    print(f"Loading {csv_path}...")
    df = _prepare_aspect_ratio_df(csv_path, layer_column, include_wound)
    df = df[df[layer_column] == layer_id].copy()
    if df.empty:
        print(f"No data for layer {layer_id}. Exiting.")
        return

    times = sorted(df['t'].unique())
    if not times:
        print("No time points found. Exiting.")
        return

    data_by_t = [df[df['t'] == t]['aspect_ratio'].values for t in times]
    positions = range(len(times))
    fig, ax = plt.subplots(figsize=(max(6, len(times) * 0.4), 5))
    bp = ax.boxplot(
        data_by_t,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker='o', markersize=2, alpha=0.4),
    )
    for patch in bp['boxes']:
        patch.set_facecolor('lightsteelblue')
        patch.set_alpha(0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(times)
    ax.set_xlabel('Time (frame)')
    ax.set_ylabel('Aspect ratio')
    ax.set_title(f'Aspect ratio over time for layer {layer_id} ({layer_column})')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved figure to {output_path}")
    else:
        plt.show()
    plt.close(fig)


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to objects_with_layers CSV")
    parser.add_argument(
        "--layer",
        default="layer_edge",
        choices=LAYER_COLUMNS,
        help="Layer column (default: layer_edge)",
    )
    parser.add_argument("--include-wound", action="store_true", help="Include layer == -1 (wound)")
    parser.add_argument("--output", "-o", default=None, help="Output figure path (default: show only)")
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI when saving (default: 150)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate layer visualizations: size vs layer (per frame) or aspect ratio over time (per layer)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Plot mode")

    # size-vs-layer: one plot, frame required
    p_size = subparsers.add_parser("size-vs-layer", help="Size (area) vs layer at a chosen frame")
    _add_shared_args(p_size)
    p_size.add_argument("--frame", type=int, required=True, metavar="T", help="Frame index (e.g. 3 for 3rd frame)")
    p_size.set_defaults(func=_run_size_vs_layer)

    # aspect-over-time: one plot, layer-id required
    p_aspect = subparsers.add_parser("aspect-over-time", help="Aspect ratio over time for a chosen layer")
    _add_shared_args(p_aspect)
    p_aspect.add_argument("--layer-id", type=int, required=True, metavar="L", help="Layer ID to track (e.g. 2)")
    p_aspect.set_defaults(func=_run_aspect_over_time)

    args = parser.parse_args()
    args.func(args)


def _run_size_vs_layer(args) -> None:
    plot_size_vs_layer(
        csv_path=args.csv,
        frame=args.frame,
        layer_column=args.layer,
        include_wound=args.include_wound,
        output_path=args.output,
        dpi=args.dpi,
    )


def _run_aspect_over_time(args) -> None:
    plot_aspect_ratio_over_time_one_layer(
        csv_path=args.csv,
        layer_id=args.layer_id,
        layer_column=args.layer,
        include_wound=args.include_wound,
        output_path=args.output,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
