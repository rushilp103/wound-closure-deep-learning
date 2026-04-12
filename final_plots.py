from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from scipy.spatial import cKDTree

# Handle config imports or defaults
try:
    from pipeline_config import converted_tracks_csv_path, objects_with_layers_csv_path
except ImportError:
    converted_tracks_csv_path = "converted_tracks.csv"
    objects_with_layers_csv_path = "objects_with_layers.csv"

# Constants
LAYER_COLUMN = "layer_centroid"
DEFAULT_LAYERS = list(range(1, 11))
DEFAULT_MINUTES_PER_FRAME = 20.0
FRONT_LAYERS = list(range(1, 6))
REAR_LAYERS = list(range(6, 11))
ZONE_LAYER_GROUPS = (list(range(1, 4)), list(range(4, 7)), list(range(7, 11)))
ZONE_TITLES = ("Zone 1", "Zone 2", "Zone 3")

# --- UTILITY FUNCTIONS ---

def frames_to_hours(t: pd.Series | np.ndarray, minutes_per_frame: float) -> np.ndarray:
    return np.asarray(t, dtype=float) * (float(minutes_per_frame) / 60.0)

def apply_time_axis_hours_ticks(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.xaxis.set_minor_locator(MultipleLocator(2))

def add_three_hour_timepoints(df: pd.DataFrame, mpf: float, edges: tuple[float, float]) -> pd.DataFrame:
    d = df.copy()
    if d.empty: return d
    th = frames_to_hours(d["t"], mpf)
    tp = np.full(len(d), 24, dtype=np.int64)
    tp[th < edges[1]] = 12
    tp[th < edges[0]] = 0
    d["time_point"] = tp
    return d

# --- DATA PROCESSING LOGIC ---

def load_cell_size_frame(df: pd.DataFrame):
    d = df.copy()
    d["cell_size"] = (d["major_axis_length"] * 0.5) * (d["minor_axis_length"] * 0.5) * np.pi
    return d

def merge_track_points_to_layers(tracks: pd.DataFrame, objects_df: pd.DataFrame, max_dist: float) -> pd.DataFrame:
    merged = tracks.copy()
    merged["_layer"] = np.nan
    for t in sorted(tracks["t"].unique()):
        mask_t = tracks["t"] == t
        tr = tracks.loc[mask_t]
        obj_t = objects_df[objects_df["t"] == t]
        if obj_t.empty or len(tr) == 0: continue
        tree = cKDTree(obj_t[["x", "y"]].values)
        dist, idx = tree.query(tr[["x", "y"]].values, k=1)
        valid = dist <= max_dist
        lays = np.full(len(tr), np.nan)
        lays[valid] = obj_t.iloc[idx[valid]][LAYER_COLUMN].to_numpy()
        merged.loc[mask_t, "_layer"] = lays
    return merged

def segment_speeds_end_layer(track_with_layers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, grp in track_with_layers.groupby("trackID", sort=False):
        grp = grp.sort_values("t")
        for i in range(len(grp) - 1):
            r0, r1 = grp.iloc[i], grp.iloc[i + 1]
            dt = float(r1["t"] - r0["t"])
            if dt <= 0: continue
            speed = np.hypot(r1["x"] - r0["x"], r1["y"] - r0["y"]) / dt
            if not np.isnan(r1["_layer"]):
                rows.append({"t": int(r1["t"]), "layer": int(r1["_layer"]), "speed": speed})
    return pd.DataFrame(rows)

def aggregate_speed_into_time_bins(seg_df: pd.DataFrame, layers: list[int], mpf: float, bh: float, x_axis_hours: bool, um_per_pixel: float | None = None) -> pd.DataFrame:
    d = seg_df[seg_df["layer"].isin(layers)].copy()
    if d.empty: return pd.DataFrame()
    
    # Apply unit conversion if calibration is provided
    if um_per_pixel is not None:
        d["speed"] = d["speed"] * (um_per_pixel / mpf)
    
    if x_axis_hours:
        d["time_val"] = frames_to_hours(d["t"], mpf)
        d["bin_index"] = np.floor(d["time_val"] / bh).astype(np.int64)
        d["x_plot"] = (d["bin_index"] + 1) * bh
    else:
        d["x_plot"] = d["t"]
        
    return d.groupby(["x_plot", "layer"], as_index=False)["speed"].agg(mean="mean", sem="sem")

def add_aspect_time_bins(df: pd.DataFrame, mpf: float, bh: float, x_axis_hours: bool) -> pd.DataFrame:
    d = df.copy()
    if d.empty: return d
    if x_axis_hours:
        d["time_val"] = frames_to_hours(d["t"], mpf)
        d["bin_index"] = np.floor(d["time_val"] / bh).astype(np.int64)
        d["x_plot"] = (d["bin_index"] + 1) * bh
    else:
        d["x_plot"] = d["t"]
    return d

# --- PLOTTING FUNCTIONS ---

def plot_aspect_ratio_front_rear(axes: tuple[plt.Axes, plt.Axes], df: pd.DataFrame, mpf: float, bh: float, x_axis_hours: bool) -> None:
    sns.set_style("ticks")
    y_limit = (1, df["aspect_ratio"].quantile(0.99) * 1.1 if not df.empty else 7)
    regions = [(axes[0], FRONT_LAYERS, "Front Wound Margin"), (axes[1], REAR_LAYERS, "Rear Wound Margin")]
    
    for ax, layer_ids, title in regions:
        plot_df = add_aspect_time_bins(df[df["layer"].isin(layer_ids)], mpf, bh, x_axis_hours)
        if plot_df.empty: continue
        order = sorted(plot_df["x_plot"].unique())
        
        sns.boxplot(data=plot_df, x="x_plot", y="aspect_ratio", order=order, palette="crest", ax=ax,
                    linewidth=1.5, fliersize=1.2, flierprops={"alpha": 0.3, "marker": "o", "markeredgecolor": "none"})
        
        ax.set_title(title, fontweight='bold', loc='left', fontsize=14)
        ax.set_ylabel("Aspect Ratio ($L/W$)")
        ax.set_ylim(y_limit)
        
        if ax == axes[0]:
            ax.set_xlabel("")
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Time (h)" if x_axis_hours else "Frame (t)")
            if x_axis_hours:
                ax.set_xticklabels([f"{int(float(t.get_text()))}" for t in ax.get_xticklabels()])
    
    sns.despine(offset=10, trim=True)

def plot_speed_lines(ax: plt.Axes, seg_df: pd.DataFrame, layers: list[int], mpf: float, bh: float, x_axis_hours: bool, um_per_pixel: float | None = None) -> None:
    sns.set_style("ticks")
    palette = sns.color_palette("bright", n_colors=len(layers))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'X']
    
    agg = aggregate_speed_into_time_bins(seg_df, layers, mpf, bh, x_axis_hours, um_per_pixel)
    
    for i, layer in enumerate(sorted(layers)):
        sub = agg[agg["layer"] == layer].sort_values("x_plot")
        if sub.empty: continue
        
        ax.errorbar(
            sub["x_plot"], sub["mean"], yerr=sub["sem"],
            label=f"Layer {layer}", 
            color=palette[i], 
            marker=markers[i % len(markers)], 
            linewidth=2, 
            capsize=3, 
            markersize=5, 
            alpha=0.9
        )

    ax.set_title("Mean Cell Speed by Layer", fontweight='bold', loc='left', fontsize=14)
    ax.set_xlabel("Time (h)" if x_axis_hours else "Frame (t)")
    
    if um_per_pixel is not None:
        ax.set_ylabel("Speed ($\mu m/min$)")
    else:
        ax.set_ylabel("Speed (px/frame)")
    
    ax.legend(title="Spatial Layer", loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, ncol=1)
    
    if x_axis_hours:
        apply_time_axis_hours_ticks(ax)
        ax.set_xlim(0, 30)
        
    sns.despine(offset=10)

def plot_cell_size_zones_three(
    axes: tuple[plt.Axes, plt.Axes, plt.Axes],
    df: pd.DataFrame,
    minutes_per_frame: float,
    time_edges_h: tuple[float, float],
    um_per_pixel: float | None = None,
    sharey: bool = True,
) -> None:
    """Three panels: zones 1-3; boxplots at time points 0, 12, 24 h. (Removed mean square)."""
    time_order = [0, 12, 24]
    
    # Calibration Logic
    if um_per_pixel is not None:
        # Area scales by the square of the linear calibration
        df = df.copy()
        df["cell_size"] = df["cell_size"] * (um_per_pixel ** 2)
        ylabel = "Cell Area ($\mu m^2$)"
    else:
        ylabel = "Cell Area ($px^2$)"

    # Establish global Y-limit for comparability across zones
    # (Adjust quantile as needed for visibility, 0.98 is often a good poster default)
    y_max = df["cell_size"].quantile(0.995) * 1.1 if not df.empty else 5000
    
    palette = sns.color_palette("crest", n_colors=len(time_order))
    
    for i, (ax, layer_ids, title) in enumerate(zip(axes, ZONE_LAYER_GROUPS, ZONE_TITLES)):
        sub = df[df["layer"].isin(layer_ids)].copy()
        plot_df = add_three_hour_timepoints(sub, minutes_per_frame, time_edges_h)
        
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_xlabel("Time (h)")
        
        if i == 0:
            ax.set_ylabel(ylabel, fontsize=12)
        else:
            ax.set_ylabel("") # Hide Y-label for middle/right plots

        if plot_df.empty:
            continue
            
        # Boxplot Aesthetics
        # We removed the ax.plot loop that was here previously.
        sns.boxplot(
            data=plot_df, x="time_point", y="cell_size", order=time_order,
            palette=palette, dodge=False, ax=ax, linewidth=1.5,
            showfliers=True, fliersize=2.0,
            flierprops={"alpha": 0.15, "marker": "o", "markeredgecolor": "none", "markerfacecolor": "black"}
        )
        for j, tp in enumerate(time_order):
            vals = plot_df.loc[plot_df["time_point"] == tp, "cell_size"]
            if not vals.empty: ax.plot(j, vals.mean(), marker="s", mfc="white", mec="black", ms=6, zorder=10)
        ax.set_ylim(0, y_max)
        sns.despine(ax=ax, offset=5)

# --- MAIN ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=objects_with_layers_csv_path)
    parser.add_argument("--tracks", default=converted_tracks_csv_path)
    parser.add_argument("--plot", choices=("aspect", "speed", "size"), default="aspect")
    parser.add_argument("--x-axis", choices=("frame", "hours"), default="frame")
    parser.add_argument("--um-per-pixel", type=float, default=None, help="Micrometers per pixel calibration factor")
    parser.add_argument("--minutes-per-frame", type=float, default=DEFAULT_MINUTES_PER_FRAME, help="Minutes per frame calibration factor")
    parser.add_argument("--o", help="Output filename to save plot")
    args = parser.parse_args()

    use_hours = (args.x_axis == "hours")

    obj_df = pd.read_csv(args.csv)
    obj_df["aspect_ratio"] = obj_df["major_axis_length"] / obj_df["minor_axis_length"]
    obj_df["layer"] = obj_df[LAYER_COLUMN].astype(int)

    if args.plot == "aspect":
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        plot_aspect_ratio_front_rear((axes[0], axes[1]), obj_df, args.minutes_per_frame, 2.0, x_axis_hours=use_hours)
    elif args.plot == "speed":
        if not args.tracks:
            print("Error: --tracks is required for speed plot")
            return
        tracks = pd.read_csv(args.tracks)
        twl = merge_track_points_to_layers(tracks, obj_df, 8.0)
        seg_df = segment_speeds_end_layer(twl)
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_speed_lines(ax, seg_df, DEFAULT_LAYERS, args.minutes_per_frame, 2.0, x_axis_hours=use_hours, um_per_pixel=args.um_per_pixel)
    elif args.plot == "size":
        cell_df = load_cell_size_frame(obj_df)
        fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
        plot_cell_size_zones_three(axes, cell_df, args.minutes_per_frame, (6.0, 18.0), um_per_pixel=args.um_per_pixel)
    
    plt.tight_layout()
    if args.o:
        plt.savefig(args.o, dpi=300, bbox_inches="tight")
        print(f"Saved poster-ready plot to {args.o}")
    else:
        plt.show()

if __name__ == "__main__":
    main()