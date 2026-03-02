import btrack
from btrack.btypes import PyTrackObject
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

from pipeline_config import (
    objects_csv_path,
    RESULTS_DIR,
    tracks_h5_path,
    BTRACK_CONFIG_PATH,
)

# --- CONFIGURATION ---
INPUT_CSV = objects_csv_path
OUTPUT_DIR = RESULTS_DIR
FILE_NAME = os.path.basename(tracks_h5_path)
CONFIG_PATH = BTRACK_CONFIG_PATH

# Tracking Parameters
MAX_SEARCH_RADIUS = 50.0
# ---------------------

def create_btrack_objects(df):
    print("Converting DataFrame to btrack objects...")
    objects = []

    # Iterate through the dataframe
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Creating objects"):
        obj = PyTrackObject()
        
        obj.ID = int(row.obj_id)
        obj.x = float(row.x)
        obj.y = float(row.y)
        obj.z = 0.0   # Explicitly 0.0 for 2D tracking
        obj.t = int(row.t)
        obj.dummy = False 
        
        if hasattr(row, 'area'):
            obj.properties = {'area': float(row.area)}
            
        objects.append(obj)
        
    return objects

def run_tracking():
    # 1. Load Data
    print(f"Loading objects from {INPUT_CSV}...")
    objects_df = pd.read_csv(INPUT_CSV)
    
    # Sort by time (required for tracking)
    if 't' in objects_df.columns:
        objects_df = objects_df.sort_values('t')

    # 2. Convert to Objects
    objects = create_btrack_objects(objects_df)
    print(f"  > Loaded {len(objects)} objects. Loading configuration...")

    # 3. Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    output_path = tracks_h5_path

    # 4. Run Tracker
    print("Starting tracking...")
    with btrack.BayesianTracker() as tracker:
        tracker.configure(CONFIG_PATH)
        tracker.max_search_radius = MAX_SEARCH_RADIUS
        tracker.append(objects)
        
        # Track
        tracker.track(step_size=100) 
        
        # Optimize
        # print("Optimizing tracks...")
        # tracker.optimize()
        
        tracks = tracker.tracks
        print(f"Tracking complete. Found {len(tracks)} unique trajectories.")

        # 5. Export Results using Built-in Function
        if len(tracks) > 0:
            print("Exporting to HDF5...")
            tracker.export(output_path, obj_type='obj_type_1')
            print(f"Exported tracks to {output_path}")
        else:
            print("No tracks found to export.")

if __name__ == "__main__":
    run_tracking()