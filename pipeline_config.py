"""
Single source of truth for pipeline input/output paths.
Change BASE_NAME (and optionally directory names) here to switch datasets;
all pipeline and visualization scripts import from this module.
"""
import os

# Project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Directory layout
DATA_DIR = os.path.join(PROJECT_ROOT, "Data Sets")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "Cellpose-SAM Results")

# Dataset switch: set to the base name of your input TIFF (without .tif).
# Example: "ctrl-1" for Data Sets/ctrl-1.tif, "treat-1" for Data Sets/treat-1.tif
BASE_NAME = "ctrl-1"

# Project-level config (not per-dataset)
BTRACK_CONFIG_PATH = os.path.join(PROJECT_ROOT, "btrack_config.json")

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
