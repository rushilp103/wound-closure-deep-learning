"""
Single source of truth for pipeline input/output paths.
Change BASE_NAME (and optionally directory names) here to switch datasets;
all pipeline and visualization scripts import from this module.

Optional environment overrides (used by textual_app.py and similar launchers):
  WOUND_BASE_NAME        — dataset stem, e.g. ctrl-1 (default: ctrl-1)
  WOUND_DATA_DIR         — override Data Sets directory
  WOUND_RESULTS_DIR      — override Results directory
  WOUND_BTRACK_CONFIG    — override path to btrack_config.json
"""
import os

# Project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

_default_data = os.path.join(PROJECT_ROOT, "Data Sets")
_default_results = os.path.join(PROJECT_ROOT, "Results")
DATA_DIR = os.environ.get("WOUND_DATA_DIR") or _default_data
RESULTS_DIR = os.environ.get("WOUND_RESULTS_DIR") or _default_results

# Dataset switch: set to the base name of your input TIFF (without .tif).
# Example: "ctrl-1" for Data Sets/ctrl-1.tif, "treat-1" for Data Sets/treat-1.tif
BASE_NAME = (os.environ.get("WOUND_BASE_NAME") or "ctrl-1").strip() or "ctrl-1"

_default_btrack = os.path.join(PROJECT_ROOT, "btrack_config.json")
BTRACK_CONFIG_PATH = os.environ.get("WOUND_BTRACK_CONFIG") or _default_btrack

# --- Path helpers (all under RESULTS_DIR unless noted) ---
def _join(*parts):
    return os.path.join(*parts)


input_tif_path = _join(DATA_DIR, f"{BASE_NAME}.tif")
masks_tracking_path = _join(RESULTS_DIR, f"{BASE_NAME}_masks_TRACKING.tif")
objects_csv_path = _join(RESULTS_DIR, f"{BASE_NAME}_objects.csv")
tracks_h5_path = _join(RESULTS_DIR, f"{BASE_NAME}_tracks.h5")
converted_tracks_csv_path = _join(RESULTS_DIR, f"{BASE_NAME}_converted_tracks.csv")

masks_tracking_layers_path = _join(RESULTS_DIR, f"{BASE_NAME}_masks_TRACKING.tif")
objects_csv_layers_path = _join(RESULTS_DIR, f"{BASE_NAME}_objects.csv")
objects_with_layers_csv_path = _join(RESULTS_DIR, f"{BASE_NAME}_objects_with_layers.csv")
