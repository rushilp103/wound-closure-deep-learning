"""
Plot per-layer aspect ratio (boxplots over time) and mean track speed (lines over time).
Reads objects_with_layers.csv and optionally converted_tracks.csv. Layers 0-9 use fixed
tab10 colors; subset via --layers. Use --plot to show both plots, aspect only, or speed only.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial import cKDTree

from pipeline_config import converted_tracks_csv_path, objects_with_layers_csv_path

LAYER_COLUMN = "layer_centroid"
DEFAULT_LAYERS = list(range(10))


def layer_palette() -> dict[int, tuple]:
    """Fixed colors for layer IDs 0..9 (tab10)."""
    pal = sns.color_palette("tab10", 10)
    return {i: pal[i] for i in range(10)}


def parse_layers(s: str | None) -> list[int]:
    if s is None or not str(s).strip():
        return list(DEFAULT_LAYERS)
    parts = [int(x.strip()) for x in str(s).split(",") if x.strip()]
    for p in parts:
        if p < 0 or p > 9:
            print(f"Error: layer IDs must be in 0..9, got {p}.", file=sys.stderr)
            sys.exit(1)
    return sorted(set(parts))


def load_aspect_ratio_frame(csv_path: str, layers: list[int]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = ["t", "major_axis_length", "minor_axis_length", LAYER_COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Error: CSV missing columns: {missing}", file=sys.stderr)
        sys.exit(1)
    df = df.copy()
    df["aspect_ratio"] = df["major_axis_length"] / df["minor_axis_length"]
    valid = (
        (df["minor_axis_length"] > 0)
        & df["major_axis_length"].notna()
        & df["minor_axis_length"].notna()
    )
    df = df.loc[valid].copy()
    df = df[df[LAYER_COLUMN].isin(layers)].copy()
    df["layer"] = df[LAYER_COLUMN].astype(int)
    return df


def plot_aspect_ratio_boxplots(
    ax: plt.Axes,
    df: pd.DataFrame,
    layers: list[int],
    colors: dict[int, tuple],
) -> None:
    if df.empty:
        ax.set_title("Aspect ratio over time (no data)")
        return
    order = sorted(layers)
    palette = {str(l): colors[l] for l in order}
    plot_df = df.copy()
    plot_df["layer_str"] = plot_df["layer"].astype(str)
    hue_order = [str(l) for l in order]
    sns.boxplot(
        data=plot_df,
        x="t",
        y="aspect_ratio",
        hue="layer_str",
        hue_order=hue_order,
        order=sorted(plot_df["t"].unique()),
        palette=palette,
        ax=ax,
        linewidth=0.8,
        fliersize=1.5,
    )
    ax.set_xlabel("Frame (t)")
    ax.set_ylabel("Aspect ratio")
    ax.set_title("Aspect ratio over time")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    h, lab = ax.get_legend_handles_labels()
    ax.legend(h, [f"Layer {int(x)}" for x in lab], title="Layer", loc="best")


def merge_track_points_to_layers(
    tracks: pd.DataFrame,
    objects_df: pd.DataFrame,
    max_dist: float,
) -> pd.DataFrame:
    """Add `_layer` column from nearest object at same (t) within max_dist pixels."""
    layer_col = LAYER_COLUMN
    merged = tracks.copy()
    merged["_layer"] = np.nan

    for t in sorted(tracks["t"].unique()):
        mask_t = tracks["t"] == t
        tr = tracks.loc[mask_t]
        obj_t = objects_df[objects_df["t"] == t]
        if obj_t.empty or len(tr) == 0:
            continue
        tree = cKDTree(obj_t[["x", "y"]].values)
        dist, idx = tree.query(tr[["x", "y"]].values, k=1)
        dist = np.atleast_1d(np.asarray(dist))
        idx = np.atleast_1d(np.asarray(idx))
        valid = dist <= max_dist
        lays = np.full(len(tr), np.nan)
        lays[valid] = obj_t.iloc[idx[valid]][layer_col].to_numpy()
        merged.loc[mask_t, "_layer"] = lays

    return merged


def segment_speeds_end_layer(track_with_layers: pd.DataFrame) -> pd.DataFrame:
    """Instantaneous speed between consecutive track points; layer taken at end frame."""
    rows: list[dict] = []
    twl = track_with_layers.sort_values(["trackID", "t"])
    for _, grp in twl.groupby("trackID", sort=False):
        grp = grp.sort_values("t")
        for i in range(len(grp) - 1):
            r0 = grp.iloc[i]
            r1 = grp.iloc[i + 1]
            dt = float(r1["t"] - r0["t"])
            if dt <= 0:
                continue
            dx = float(r1["x"] - r0["x"])
            dy = float(r1["y"] - r0["y"])
            speed = float(np.hypot(dx, dy) / dt)
            lay = r1["_layer"]
            if np.isnan(lay):
                continue
            rows.append({"t": int(r1["t"]), "layer": int(lay), "speed": speed})
    return pd.DataFrame(rows)


def mean_speed_lines(
    ax: plt.Axes,
    seg_df: pd.DataFrame,
    layers: list[int],
    colors: dict[int, tuple],
    ylabel: str,
) -> None:
    seg_df = seg_df[seg_df["layer"].isin(layers)].copy()
    if seg_df.empty:
        ax.set_title("Mean speed over time (no data)")
        ax.set_ylabel(ylabel)
        return
    agg = seg_df.groupby(["t", "layer"], as_index=False)["speed"].mean()
    for layer in sorted(layers):
        sub = agg[agg["layer"] == layer].sort_values("t")
        if sub.empty:
            continue
        ax.plot(
            sub["t"],
            sub["speed"],
            color=colors[layer],
            label=f"Layer {layer}",
            linewidth=1.5,
        )
    ax.set_xlabel("Frame (t)")
    ax.set_ylabel(ylabel)
    ax.set_title("Mean speed over time")
    ax.legend(title="Layer", loc="best")


def apply_speed_units(speed: float, um_per_pixel: float | None, minutes_per_frame: float | None) -> float:
    """Convert px/frame to µm/min if both calibration args are set; else return px/frame."""
    if um_per_pixel is not None and minutes_per_frame is not None:
        return speed * um_per_pixel / minutes_per_frame
    return speed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot layer aspect ratio (boxplots) and/or mean track speed (lines) over time."
    )
    parser.add_argument("--csv", default=objects_with_layers_csv_path, help="objects_with_layers CSV")
    parser.add_argument(
        "--tracks",
        default=converted_tracks_csv_path,
        help="converted_tracks CSV (required for speed plot)",
    )
    parser.add_argument(
        "--plot",
        choices=("both", "aspect", "speed"),
        default="both",
        help="Which plot(s): both (default), aspect only, or speed only",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layer IDs 0-9 (default: all ten 0-9)",
    )
    parser.add_argument("-o", "--output", default=None, help="Save figure to this path (PNG/PDF/...)")
    parser.add_argument(
        "--nn-max-dist",
        type=float,
        default=8.0,
        help="Max distance (pixels) for matching a track point to an object (default: 8)",
    )
    parser.add_argument(
        "--um-per-pixel",
        type=float,
        default=None,
        metavar="UM",
        help="If set with --minutes-per-frame, speed axis is µm/min",
    )
    parser.add_argument(
        "--minutes-per-frame",
        type=float,
        default=None,
        metavar="MIN",
        help="Minutes per frame; use with --um-per-pixel for physical speed",
    )
    args = parser.parse_args()

    if args.um_per_pixel is not None and args.minutes_per_frame is None:
        print("Error: --minutes-per-frame is required when --um-per-pixel is set.", file=sys.stderr)
        sys.exit(1)
    if args.minutes_per_frame is not None and args.um_per_pixel is None:
        print("Error: --um-per-pixel is required when --minutes-per-frame is set.", file=sys.stderr)
        sys.exit(1)

    layers = parse_layers(args.layers)
    colors = layer_palette()
    want_aspect = args.plot in ("both", "aspect")
    want_speed = args.plot in ("both", "speed")

    objects_df: pd.DataFrame | None = None
    if want_aspect or want_speed:
        if not os.path.isfile(args.csv):
            print(f"Error: CSV not found: {args.csv}", file=sys.stderr)
            sys.exit(1)
        objects_df = pd.read_csv(args.csv)
        if want_speed:
            required = {"t", "x", "y", LAYER_COLUMN}
            missing = sorted(required - set(objects_df.columns))
            if missing:
                print(
                    f"Error: {args.csv} missing required columns for speed plot: {missing}",
                    file=sys.stderr,
                )
                sys.exit(1)

    ar_df = pd.DataFrame()
    if want_aspect:
        assert objects_df is not None
        ar_df = load_aspect_ratio_frame(args.csv, layers)

    seg_df = pd.DataFrame()
    if want_speed:
        if not os.path.isfile(args.tracks):
            print(f"Error: tracks CSV not found: {args.tracks}", file=sys.stderr)
            sys.exit(1)
        assert objects_df is not None
        tracks = pd.read_csv(args.tracks)
        if "dummy" in tracks.columns:
            tracks = tracks[~tracks["dummy"].astype(bool)]
        req = {"trackID", "t", "x", "y"}
        miss = req - set(tracks.columns)
        if miss:
            print(f"Error: tracks CSV missing columns: {miss}", file=sys.stderr)
            sys.exit(1)
        twl = merge_track_points_to_layers(tracks, objects_df, args.nn_max_dist)
        seg_df = segment_speeds_end_layer(twl)
        if args.um_per_pixel is not None:
            seg_df = seg_df.copy()
            seg_df["speed"] = seg_df["speed"].apply(
                lambda s: apply_speed_units(s, args.um_per_pixel, args.minutes_per_frame)
            )

    speed_ylabel = (
        "Speed (µm/min)"
        if args.um_per_pixel is not None
        else "Speed (px / frame)"
    )

    if args.plot == "both":
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)
        plot_aspect_ratio_boxplots(axes[0], ar_df, layers, colors)
        mean_speed_lines(axes[1], seg_df, layers, colors, speed_ylabel)
    elif args.plot == "aspect":
        fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
        plot_aspect_ratio_boxplots(ax, ar_df, layers, colors)
    else:
        fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
        mean_speed_lines(ax, seg_df, layers, colors, speed_ylabel)

    if args.output:
        out_dir = os.path.dirname(args.output)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
