"""
Plot mean displacement over time for each layer.
Displacement = distance moved from previous frame (same obj_id).
Requires objects_with_layers.csv (from assign_layers.py) with x, y, t, obj_id and layer column.
Use --layers to show only specific layer IDs; omit to show all layers.
"""
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline_config import objects_with_layers_csv_path

LAYER_COLUMNS = ['layer_edge', 'layer_centroid']
REQUIRED_COLUMNS = ['t', 'obj_id', 'x', 'y']
DEFAULT_CSV = objects_with_layers_csv_path


def _compute_displacement_by_layer(
    csv_path: str,
    layer_column: str,
    include_wound: bool,
) -> pd.DataFrame:
    """
    Load CSV with x, y, t, obj_id and layer column. Compute per-cell displacement
    from previous frame (same obj_id). Return DataFrame with t, layer, displacement
    (one row per cell per time step where t > 0 and cell had position at t-1).
    """
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS + [layer_column] if c not in df.columns]
    if missing:
        print(f"Error: CSV is missing columns: {missing}.", file=sys.stderr)
        sys.exit(1)
    df = df.sort_values(['obj_id', 't']).copy()
    if not include_wound:
        df = df[df[layer_column] != -1].copy()

    # Displacement from previous frame: merge each row with same obj_id at t-1
    df_prev = df[['obj_id', 't', 'x', 'y']].copy()
    df_prev = df_prev.rename(columns={'t': 't_prev', 'x': 'x_prev', 'y': 'y_prev'})
    df_prev['t'] = df_prev['t_prev'] + 1
    merged = df.merge(df_prev, on=['obj_id', 't'], how='inner')
    merged['displacement'] = np.sqrt(
        (merged['x'] - merged['x_prev']) ** 2 + (merged['y'] - merged['y_prev']) ** 2
    )
    return merged[['t', layer_column, 'displacement']].rename(columns={layer_column: 'layer'})


def plot_displacement_over_time(
    csv_path: str,
    layer_column: str = 'layer_edge',
    include_wound: bool = False,
    layers_to_show: list[int] | None = None,
    output_path: str | None = None,
    dpi: int = 150,
) -> None:
    """
    Plot mean displacement over time for each layer.
    X = time (frame), Y = mean displacement (px). One line per layer.
    If layers_to_show is None, show all layers; otherwise only those layer IDs.
    """
    print(f"Loading {csv_path}...")
    df = _compute_displacement_by_layer(csv_path, layer_column, include_wound)
    if df.empty:
        print("No displacement data (need same obj_id across consecutive frames). Exiting.")
        return

    layers = sorted(df['layer'].dropna().unique())
    if layers_to_show is not None:
        layers = [l for l in layers if int(l) in layers_to_show]
    if not layers:
        print("No layers to plot (none match --layers filter). Exiting.")
        return

    times = sorted(df['t'].unique())
    if not times:
        print("No time points. Exiting.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(layers))))

    for i, layer in enumerate(layers):
        layer_df = df[df['layer'] == layer]
        mean_disp = [layer_df[layer_df['t'] == t]['displacement'].mean() for t in times]
        color = colors[i % len(colors)]
        ax.plot(times, mean_disp, 'o-', label=f'Layer {int(layer)}', color=color)

    ax.set_xlabel('Time (frame)')
    ax.set_ylabel('Mean displacement (px)')
    ax.set_title(f'Displacement over time by layer ({layer_column})')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved figure to {output_path}")
    else:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot mean displacement over time for each layer. Use --layers to restrict; omit to show all."
    )
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to objects_with_layers CSV")
    parser.add_argument(
        "--layer",
        default="layer_edge",
        choices=LAYER_COLUMNS,
        help="Layer column (default: layer_edge)",
    )
    parser.add_argument("--include-wound", action="store_true", help="Include layer == -1 (wound)")
    parser.add_argument(
        "--layers",
        nargs="*",
        type=int,
        default=None,
        metavar="L",
        help="Layer IDs to show (default: all layers)",
    )
    parser.add_argument("--output", "-o", default=None, help="Output figure path (default: show only)")
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI when saving (default: 150)")
    args = parser.parse_args()

    layers = args.layers
    if layers is not None and len(layers) == 0:
        layers = None

    plot_displacement_over_time(
        csv_path=args.csv,
        layer_column=args.layer,
        include_wound=args.include_wound,
        layers_to_show=layers,
        output_path=args.output,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
