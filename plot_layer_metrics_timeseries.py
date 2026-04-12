"""
Plot per-layer aspect ratio (boxplots over time) and mean track speed (lines over time).
Reads objects_with_layers.csv and optionally converted_tracks.csv. Outside-wound layers are
1-based (1..10 by default, matching assign_layers.py); tab10 colors map layer N to palette N-1.
Subset via --layers. Use --plot to show both plots, aspect only, speed only, or cell-size zones only.
Speed subplot: mean per (frame, layer) with optional error bars (--speed-error sem|std|none).

Use --x-axis hours to show Time (h) on both plots (major ticks every 4 h, minor every 2 h).
When --x-axis hours and --minutes-per-frame is omitted, 20 minutes per frame is assumed.

With --x-axis hours, the speed plot uses 2-hour bins by default (pool segment speeds into [0,2), [2,4), ...
h; points at 2, 4, 6, ... h). Use --no-speed-bins for one point per frame, or --speed-bin-hours H to
use a different bin width.

Use --aspect-layout front_rear with --x-axis hours for two aspect panels: front (layers 1-5) and rear
(6-10), each with one boxplot per time bin pooling all objects in those layers and frames in the bin.
Bin width defaults to 2 h; override with --aspect-bin-hours.

Use --plot size with --x-axis hours for three side-by-side panels (Zone 1: layers 1-3, Zone 2: 4-6,
Zone 3: 7-10), boxplots of cell size at categorical times 0, 12, 24 h. Time bins default to
[0,6)->0 h, [6,18)->12 h, [18,inf)->24 h; override with --cell-size-bin-edges H0,H1. Cell area uses
column area if present, else ellipse pi*(a/2)*(b/2). With --um-per-pixel, Y is µm².
"""
from __future__ import annotations

import argparse
from typing import Literal
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from scipy.spatial import cKDTree

from pipeline_config import converted_tracks_csv_path, objects_with_layers_csv_path

LAYER_COLUMN = "layer_centroid"
# assign_layers.py writes 1-based outside-wound layers 1..num_layers (default num_layers=10).
DEFAULT_LAYERS = list(range(1, 11))
DEFAULT_MINUTES_PER_FRAME = 20.0
DEFAULT_SPEED_BIN_HOURS = 2.0
# Pooled aspect layout: outside-wound layers 1-5 (front) and 6-10 (rear).
FRONT_LAYERS = list(range(1, 6))
REAR_LAYERS = list(range(6, 11))
# Cell-size zone layout: three groups of outside-wound layers.
ZONE_LAYER_GROUPS: tuple[list[int], ...] = (
    list(range(1, 4)),
    list(range(4, 7)),
    list(range(7, 11)),
)
ZONE_TITLES = ("Zone 1", "Zone 2", "Zone 3")
# Face color for all boxes in that zone (grayscale).
ZONE_BOX_FACE = ("0.38", "0.58", "0.78")
# Default boundaries (hours) for labeling bins as 0, 12, 24: [0, e0)->0, [e0, e1)->12, [e1, inf)->24.
DEFAULT_CELL_SIZE_TIME_EDGES_H = (6.0, 18.0)


def frames_to_hours(t: pd.Series | np.ndarray, minutes_per_frame: float) -> np.ndarray:
    """Map frame index t to elapsed time in hours."""
    return np.asarray(t, dtype=float) * (float(minutes_per_frame) / 60.0)


def apply_time_axis_hours_ticks(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.xaxis.set_minor_locator(MultipleLocator(2))


def aggregate_speed_into_time_bins(
    seg_df: pd.DataFrame,
    layers: list[int],
    minutes_per_frame: float,
    bin_hours: float,
) -> pd.DataFrame:
    """
    Pool segment speeds into half-open time bins [0, H), [H, 2H), ... (hours).
    Returns columns x_time_h (right edge of bin = H, 2H, ...), layer, mean, sem, std.
    """
    d = seg_df[seg_df["layer"].isin(layers)].copy()
    if d.empty:
        return pd.DataFrame(columns=["x_time_h", "layer", "mean", "sem", "std"])
    bh = float(bin_hours)
    d["time_h"] = frames_to_hours(d["t"], minutes_per_frame)
    d["bin_index"] = np.floor(d["time_h"] / bh).astype(np.int64)
    d["x_time_h"] = (d["bin_index"] + 1) * bh
    return (
        d.groupby(["x_time_h", "layer"], as_index=False)["speed"]
        .agg(mean="mean", sem="sem", std="std")
    )


def layer_palette() -> dict[int, tuple]:
    """Fixed colors for layer IDs 1..10 (tab10 index layer-1)."""
    pal = sns.color_palette("tab10", 10)
    return {i: pal[i - 1] for i in range(1, 11)}


def parse_cell_size_bin_edges(s: str) -> tuple[float, float]:
    parts = [float(x.strip()) for x in str(s).split(",") if x.strip()]
    if len(parts) != 2:
        print(
            "Error: --cell-size-bin-edges requires two comma-separated values, e.g. 6,18",
            file=sys.stderr,
        )
        sys.exit(1)
    return parts[0], parts[1]


def parse_layers(s: str | None) -> list[int]:
    if s is None or not str(s).strip():
        return list(DEFAULT_LAYERS)
    parts = [int(x.strip()) for x in str(s).split(",") if x.strip()]
    for p in parts:
        if p < 1 or p > 10:
            print(f"Error: layer IDs must be in 1..10, got {p}.", file=sys.stderr)
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
    x_axis_hours: bool = False,
    minutes_per_frame: float = DEFAULT_MINUTES_PER_FRAME,
) -> None:
    if df.empty:
        ax.set_title("Aspect ratio over time (no data)")
        return
    order = sorted(layers)
    palette = {str(l): colors[l] for l in order}
    plot_df = df.copy()
    plot_df["layer_str"] = plot_df["layer"].astype(str)
    hue_order = [str(l) for l in order]
    if x_axis_hours:
        plot_df["time_h"] = frames_to_hours(plot_df["t"], minutes_per_frame)
        x_col = "time_h"
        x_order = sorted(plot_df["time_h"].unique())
    else:
        x_col = "t"
        x_order = sorted(plot_df["t"].unique())
    sns.boxplot(
        data=plot_df,
        x=x_col,
        y="aspect_ratio",
        hue="layer_str",
        hue_order=hue_order,
        order=x_order,
        palette=palette,
        ax=ax,
        linewidth=0.8,
        fliersize=1.5,
    )
    ax.set_xlabel("Time (h)" if x_axis_hours else "Frame (t)")
    ax.set_ylabel("Aspect ratio")
    ax.set_title("Aspect ratio over time")
    if x_axis_hours:
        apply_time_axis_hours_ticks(ax)
        ax.tick_params(axis="x", labelsize=8)
    else:
        ax.tick_params(axis="x", rotation=90, labelsize=7)
    h, lab = ax.get_legend_handles_labels()
    ax.legend(h, [f"Layer {int(x)}" for x in lab], title="Layer", loc="best")


def add_aspect_time_bins(
    df: pd.DataFrame,
    minutes_per_frame: float,
    bin_hours: float,
) -> pd.DataFrame:
    """Assign each row to x_time_h = (floor(time_h / H) + 1) * H (same convention as speed bins)."""
    d = df.copy()
    if d.empty:
        d["x_time_h"] = pd.Series(dtype=float)
        return d
    bh = float(bin_hours)
    d["time_h"] = frames_to_hours(d["t"], minutes_per_frame)
    d["bin_index"] = np.floor(d["time_h"] / bh).astype(np.int64)
    d["x_time_h"] = (d["bin_index"] + 1) * bh
    return d


def _aspect_binned_x_max(df: pd.DataFrame, minutes_per_frame: float, bin_hours: float) -> float:
    if df.empty:
        return float(bin_hours)
    d = add_aspect_time_bins(df, minutes_per_frame, bin_hours)
    return float(d["x_time_h"].max())


def plot_aspect_ratio_front_rear(
    axes: tuple[plt.Axes, plt.Axes],
    df: pd.DataFrame,
    minutes_per_frame: float,
    bin_hours: float,
) -> None:
    """Two panels: front (layers 1-5) and rear (6-10); one box per 2 h bin, all objects pooled."""
    ax_front, ax_rear = axes
    x_hi = _aspect_binned_x_max(df, minutes_per_frame, bin_hours)

    for ax, layer_ids, region in (
        (ax_front, FRONT_LAYERS, "front"),
        (ax_rear, REAR_LAYERS, "rear"),
    ):
        sub = df[df["layer"].isin(layer_ids)].copy()
        plot_df = add_aspect_time_bins(sub, minutes_per_frame, bin_hours)
        if plot_df.empty:
            ax.set_title(f"Aspect ratio — {region} (no data)")
            ax.set_xlabel("Time (h)")
            ax.set_ylabel("Aspect ratio")
            ax.set_xlim(0.0, x_hi)
            apply_time_axis_hours_ticks(ax)
            continue

        order = sorted(plot_df["x_time_h"].unique())
        pal = sns.color_palette("tab10", 10)
        palette_list = [pal[i % 10] for i in range(len(order))]
        sns.boxplot(
            data=plot_df,
            x="x_time_h",
            y="aspect_ratio",
            hue="x_time_h",
            order=order,
            hue_order=order,
            dodge=False,
            palette=palette_list,
            legend=False,
            ax=ax,
            linewidth=0.8,
            fliersize=1.5,
            native_scale=True,
        )
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Aspect ratio")
        ax.set_xlim(0.0, x_hi)
        apply_time_axis_hours_ticks(ax)
        ax.tick_params(axis="x", labelsize=8)
        ax.text(
            -0.11,
            0.5,
            region,
            transform=ax.transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=11,
        )


def add_three_hour_timepoints(
    df: pd.DataFrame,
    minutes_per_frame: float,
    edges_h: tuple[float, float],
) -> pd.DataFrame:
    """
    Map elapsed time_h to categorical 0, 12, 24 (int): [0, e0)->0, [e0, e1)->12, [e1, inf)->24.
    """
    d = df.copy()
    if d.empty:
        d["time_point"] = pd.Series(dtype=np.int64)
        return d
    e0, e1 = float(edges_h[0]), float(edges_h[1])
    d["time_h"] = frames_to_hours(d["t"], minutes_per_frame)
    th = d["time_h"].to_numpy(dtype=float)
    tp = np.full(len(d), 24, dtype=np.int64)
    tp[th < e1] = 12
    tp[th < e0] = 0
    d["time_point"] = tp
    return d


def load_cell_size_frame(
    csv_path: str,
    layers: list[int],
    um_per_pixel: float | None,
    objects_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, str]:
    """
    Per-object cell area: prefer CSV area (px²), else ellipse π(a/2)(b/2). Scale to µm² if calibrated.
    If objects_df is given, use it instead of reading csv_path again.
    """
    df = objects_df.copy() if objects_df is not None else pd.read_csv(csv_path)
    required = ["t", "major_axis_length", "minor_axis_length", LAYER_COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Error: CSV missing columns: {missing}", file=sys.stderr)
        sys.exit(1)
    df = df.copy()
    a = df["major_axis_length"].astype(float)
    b = df["minor_axis_length"].astype(float)
    valid = (b > 0) & a.notna() & b.notna()
    df = df.loc[valid].copy()
    df = df[df[LAYER_COLUMN].isin(layers)].copy()
    df["layer"] = df[LAYER_COLUMN].astype(int)
    ma = df["major_axis_length"].astype(float)
    mi = df["minor_axis_length"].astype(float)
    ell_area = np.pi * (ma / 2.0) * (mi / 2.0)
    if "area" in df.columns:
        area_px = df["area"].astype(float)
        area_px = area_px.where(area_px.notna() & (area_px > 0), ell_area)
    else:
        area_px = ell_area
    if um_per_pixel is not None:
        s = float(um_per_pixel) ** 2
        df["cell_size"] = area_px * s
        ylabel = r"Cell size ($\mu m^2$)"
    else:
        df["cell_size"] = area_px
        ylabel = r"Cell area (px$^2$)"
    return df, ylabel


def plot_cell_size_zones_three(
    axes: tuple[plt.Axes, plt.Axes, plt.Axes],
    df: pd.DataFrame,
    minutes_per_frame: float,
    time_edges_h: tuple[float, float],
    ylabel: str,
    sharey: bool = True,
) -> None:
    """Three panels: zones 1-3; boxplots at time points 0, 12, 24 h; mean as open square."""
    time_order = [0, 12, 24]
    ax_list = axes
    ref_ax = ax_list[0]
    for ax, layer_ids, title, face in zip(ax_list, ZONE_LAYER_GROUPS, ZONE_TITLES, ZONE_BOX_FACE):
        sub = df[df["layer"].isin(layer_ids)].copy()
        plot_df = add_three_hour_timepoints(sub, minutes_per_frame, time_edges_h)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Time (h)")
        if sharey and ax is not ref_ax:
            ax.set_ylabel("")
        else:
            ax.set_ylabel(ylabel)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
        if plot_df.empty:
            ax.set_xticks(range(len(time_order)))
            ax.set_xticklabels([str(x) for x in time_order])
            continue
        pal = {tp: face for tp in time_order}
        sns.boxplot(
            data=plot_df,
            x="time_point",
            y="cell_size",
            hue="time_point",
            order=time_order,
            hue_order=time_order,
            dodge=False,
            palette=pal,
            ax=ax,
            linewidth=0.8,
            fliersize=2.0,
            legend=False,
        )
        for i, tp in enumerate(time_order):
            vals = plot_df.loc[plot_df["time_point"] == tp, "cell_size"]
            if len(vals) == 0:
                continue
            ax.plot(
                float(i),
                float(vals.mean()),
                marker="s",
                mfc="none",
                mec="black",
                ms=4.5,
                zorder=5,
            )


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
    speed_error: Literal["sem", "std", "none"] = "sem",
    x_axis_hours: bool = False,
    minutes_per_frame: float = DEFAULT_MINUTES_PER_FRAME,
    speed_bin_hours: float | None = None,
) -> None:
    seg_df = seg_df[seg_df["layer"].isin(layers)].copy()
    if seg_df.empty:
        ax.set_title("Mean speed over time (no data)")
        ax.set_ylabel(ylabel)
        return
    binned = speed_bin_hours is not None and speed_bin_hours > 0
    if binned:
        agg = aggregate_speed_into_time_bins(
            seg_df, layers, minutes_per_frame, speed_bin_hours
        )
        sort_col = "x_time_h"
    else:
        agg = (
            seg_df.groupby(["t", "layer"], as_index=False)["speed"].agg(
                mean="mean", sem="sem", std="std"
            )
        )
        sort_col = "t"
    line_fmt = "-o" if binned else "-"
    ms = 4.0 if binned else None
    for layer in sorted(layers):
        sub = agg[agg["layer"] == layer].sort_values(sort_col)
        if sub.empty:
            continue
        if binned:
            x = sub["x_time_h"]
        elif x_axis_hours:
            x = frames_to_hours(sub["t"], minutes_per_frame)
        else:
            x = sub["t"]
        y = sub["mean"]
        if speed_error == "none":
            if ms is not None:
                ax.plot(x, y, line_fmt, color=colors[layer], label=f"Layer {layer}", linewidth=1.5, markersize=ms)
            else:
                ax.plot(x, y, color=colors[layer], label=f"Layer {layer}", linewidth=1.5)
            continue
        if speed_error == "sem":
            yerr = sub["sem"].fillna(0.0).to_numpy()
        else:
            yerr = sub["std"].fillna(0.0).to_numpy()
        eb_kwargs: dict = {
            "x": x,
            "y": y,
            "yerr": yerr,
            "fmt": line_fmt,
            "color": colors[layer],
            "label": f"Layer {layer}",
            "linewidth": 1.5,
            "capsize": 2.5,
            "elinewidth": 0.8,
        }
        if ms is not None:
            eb_kwargs["markersize"] = ms
        ax.errorbar(**eb_kwargs)
    if x_axis_hours:
        if binned and not agg.empty:
            x_hi = float(agg["x_time_h"].max())
            x_lo = 0.0
        elif not agg.empty:
            t_all = agg["t"].astype(float)
            x_hi = float(t_all.max() * minutes_per_frame / 60.0)
            x_lo = 0.0 if float(t_all.min()) == 0.0 else float(
                t_all.min() * minutes_per_frame / 60.0
            )
        else:
            x_lo, x_hi = 0.0, 1.0
        ax.set_xlim(x_lo, 30.0)
        apply_time_axis_hours_ticks(ax)
    ax.set_xlabel("Time (h)" if x_axis_hours else "Frame (t)")
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
        choices=("both", "aspect", "speed", "size"),
        default="both",
        help="Which plot(s): both (default), aspect, speed, or size (zone cell-size 0/12/24 h; needs --x-axis hours)",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layer IDs 1-10 (default: all ten 1-10; matches assign_layers.py)",
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
        "--x-axis",
        choices=("frame", "hours"),
        default="frame",
        help="X-axis: frame index (default) or Time (h) with 4 h major / 2 h minor ticks",
    )
    parser.add_argument(
        "--minutes-per-frame",
        type=float,
        default=None,
        metavar="MIN",
        help=(
            "Minutes per frame; required with --um-per-pixel for µm/min. "
            "With --x-axis hours, defaults to 20 if omitted."
        ),
    )
    parser.add_argument(
        "--speed-error",
        choices=("sem", "std", "none"),
        default="sem",
        help="Error bars on speed plot: sem (standard error of mean, default), std, or none",
    )
    parser.add_argument(
        "--no-speed-bins",
        action="store_true",
        help=(
            "With --x-axis hours, plot speed as one point per frame instead of the default 2 h bins."
        ),
    )
    parser.add_argument(
        "--speed-bin-hours",
        type=float,
        default=None,
        metavar="H",
        help=(
            "With --x-axis hours, override bin width (default: 2 h). Points at H, 2H, ...; "
            "each is the mean speed in [0,H), [H,2H), ... Incompatible with --no-speed-bins."
        ),
    )
    parser.add_argument(
        "--aspect-layout",
        choices=("layers", "front_rear"),
        default="layers",
        help=(
            "Aspect plot: hue by layer (layers), or two panels front (1-5) and rear (6-10) with "
            "one box per time bin (front_rear; requires --x-axis hours)."
        ),
    )
    parser.add_argument(
        "--aspect-bin-hours",
        type=float,
        default=None,
        metavar="H",
        help="Bin width in hours for --aspect-layout front_rear (default: 2). Requires --x-axis hours.",
    )
    parser.add_argument(
        "--cell-size-bin-edges",
        type=str,
        default=None,
        metavar="H0,H1",
        help=(
            "For --plot size: hours H0,H1 with [0,H0)->0, [H0,H1)->12, [H1,inf)->24 (default: 6,18)."
        ),
    )
    args = parser.parse_args()

    x_axis_hours = args.x_axis == "hours"
    minutes_per_frame = args.minutes_per_frame
    if x_axis_hours and minutes_per_frame is None:
        minutes_per_frame = DEFAULT_MINUTES_PER_FRAME
    if args.um_per_pixel is not None and minutes_per_frame is None:
        print("Error: --minutes-per-frame is required when --um-per-pixel is set.", file=sys.stderr)
        sys.exit(1)
    if args.speed_bin_hours is not None:
        if args.speed_bin_hours <= 0:
            print("Error: --speed-bin-hours must be positive.", file=sys.stderr)
            sys.exit(1)
        if not x_axis_hours:
            print("Error: --speed-bin-hours requires --x-axis hours.", file=sys.stderr)
            sys.exit(1)
    if args.no_speed_bins:
        if not x_axis_hours:
            print("Error: --no-speed-bins only applies with --x-axis hours.", file=sys.stderr)
            sys.exit(1)
        if args.speed_bin_hours is not None:
            print("Error: do not combine --no-speed-bins with --speed-bin-hours.", file=sys.stderr)
            sys.exit(1)
    if args.aspect_layout == "front_rear" and not x_axis_hours:
        print("Error: --aspect-layout front_rear requires --x-axis hours.", file=sys.stderr)
        sys.exit(1)
    if args.plot == "size" and not x_axis_hours:
        print("Error: --plot size requires --x-axis hours.", file=sys.stderr)
        sys.exit(1)
    cell_size_edges_h: tuple[float, float] = DEFAULT_CELL_SIZE_TIME_EDGES_H
    if args.cell_size_bin_edges is not None:
        if args.plot != "size":
            print("Error: --cell-size-bin-edges only applies with --plot size.", file=sys.stderr)
            sys.exit(1)
        e0, e1 = parse_cell_size_bin_edges(args.cell_size_bin_edges)
        if e0 <= 0 or e1 <= e0:
            print("Error: --cell-size-bin-edges require 0 < H0 < H1.", file=sys.stderr)
            sys.exit(1)
        cell_size_edges_h = (e0, e1)
    if args.aspect_bin_hours is not None:
        if args.aspect_layout != "front_rear":
            print("Error: --aspect-bin-hours requires --aspect-layout front_rear.", file=sys.stderr)
            sys.exit(1)
        if not x_axis_hours:
            print("Error: --aspect-bin-hours requires --x-axis hours.", file=sys.stderr)
            sys.exit(1)
        if args.aspect_bin_hours <= 0:
            print("Error: --aspect-bin-hours must be positive.", file=sys.stderr)
            sys.exit(1)

    layers = parse_layers(args.layers)
    colors = layer_palette()
    want_aspect = args.plot in ("both", "aspect")
    want_speed = args.plot in ("both", "speed")
    want_size = args.plot == "size"
    use_aspect_front_rear = want_aspect and args.aspect_layout == "front_rear"
    aspect_bin_hours: float | None = None
    if use_aspect_front_rear:
        aspect_bin_hours = (
            float(args.aspect_bin_hours)
            if args.aspect_bin_hours is not None
            else DEFAULT_SPEED_BIN_HOURS
        )

    speed_bin_hours: float | None = None
    if want_speed and x_axis_hours:
        if args.no_speed_bins:
            speed_bin_hours = None
        elif args.speed_bin_hours is not None:
            speed_bin_hours = args.speed_bin_hours
        else:
            speed_bin_hours = DEFAULT_SPEED_BIN_HOURS

    objects_df: pd.DataFrame | None = None
    if want_aspect or want_speed or want_size:
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
    cell_df = pd.DataFrame()
    cell_size_ylabel = ""
    if want_aspect:
        assert objects_df is not None
        aspect_layers = list(range(1, 11)) if use_aspect_front_rear else layers
        ar_df = load_aspect_ratio_frame(args.csv, aspect_layers)
    if want_size:
        assert objects_df is not None
        cell_df, cell_size_ylabel = load_cell_size_frame(
            args.csv, list(range(1, 11)), args.um_per_pixel, objects_df=objects_df
        )

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
                lambda s: apply_speed_units(s, args.um_per_pixel, minutes_per_frame)
            )

    speed_ylabel = (
        "Speed (µm/min)"
        if args.um_per_pixel is not None
        else "Speed (px / frame)"
    )

    plot_minutes_per_frame = (
        minutes_per_frame if x_axis_hours else DEFAULT_MINUTES_PER_FRAME
    )

    if args.plot == "both":
        if use_aspect_front_rear:
            assert aspect_bin_hours is not None
            fig, axes = plt.subplots(3, 1, figsize=(14, 12), constrained_layout=True)
            plot_aspect_ratio_front_rear(
                (axes[0], axes[1]),
                ar_df,
                plot_minutes_per_frame,
                aspect_bin_hours,
            )
            mean_speed_lines(
                axes[2],
                seg_df,
                layers,
                colors,
                speed_ylabel,
                speed_error=args.speed_error,
                x_axis_hours=x_axis_hours,
                minutes_per_frame=plot_minutes_per_frame,
                speed_bin_hours=speed_bin_hours,
            )
        else:
            fig, axes = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)
            plot_aspect_ratio_boxplots(
                axes[0],
                ar_df,
                layers,
                colors,
                x_axis_hours=x_axis_hours,
                minutes_per_frame=plot_minutes_per_frame,
            )
            mean_speed_lines(
                axes[1],
                seg_df,
                layers,
                colors,
                speed_ylabel,
                speed_error=args.speed_error,
                x_axis_hours=x_axis_hours,
                minutes_per_frame=plot_minutes_per_frame,
                speed_bin_hours=speed_bin_hours,
            )
    elif args.plot == "aspect":
        if use_aspect_front_rear:
            assert aspect_bin_hours is not None
            fig, axes = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)
            plot_aspect_ratio_front_rear(
                (axes[0], axes[1]),
                ar_df,
                plot_minutes_per_frame,
                aspect_bin_hours,
            )
        else:
            fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
            plot_aspect_ratio_boxplots(
                ax,
                ar_df,
                layers,
                colors,
                x_axis_hours=x_axis_hours,
                minutes_per_frame=plot_minutes_per_frame,
            )
    elif args.plot == "size":
        fig, ax_row = plt.subplots(
            1, 3, figsize=(12, 4.2), constrained_layout=True, sharey=True
        )
        plot_cell_size_zones_three(
            (ax_row[0], ax_row[1], ax_row[2]),
            cell_df,
            plot_minutes_per_frame,
            cell_size_edges_h,
            cell_size_ylabel,
            sharey=True,
        )
    else:
        fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
        mean_speed_lines(
            ax,
            seg_df,
            layers,
            colors,
            speed_ylabel,
            speed_error=args.speed_error,
            x_axis_hours=x_axis_hours,
            minutes_per_frame=plot_minutes_per_frame,
            speed_bin_hours=speed_bin_hours,
        )

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
