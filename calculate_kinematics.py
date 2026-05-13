import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import tifffile
import os

from pipeline_config import (
    BASE_NAME,
    RESULTS_DIR,
    converted_tracks_csv_path,
    masks_tracking_path,
)
from wound_utils import get_wound_mask_for_frame, wound_centroid_and_radius

def get_wound_centroid_from_tracking_masks(
    tracking_masks_path: str,
    *,
    closing_radius: int = 10,
    opening_radius: int = 5,
    erosion_radius: int = 3,
) -> tuple[float, float]:
    """
    Derive wound centroid (cx, cy) from the first frame of the tracking label stack.
    Uses wound_utils.get_wound_mask_for_frame, where wound = central background hole.
    """
    print(f"Deriving wound centroid from tracking masks: {tracking_masks_path}")

    masks = tifffile.imread(tracking_masks_path)
    if getattr(masks, "ndim", None) != 3:
        raise ValueError(
            "Expected tracking masks as a 3D stack (T, H, W). "
            f"Got shape {getattr(masks, 'shape', None)}."
        )

    first_frame = masks[0]
    wound_mask = get_wound_mask_for_frame(
        first_frame,
        closing_radius=closing_radius,
        opening_radius=opening_radius,
        erosion_radius=erosion_radius,
    )

    cy, cx, _radius = wound_centroid_and_radius(wound_mask)
    if wound_mask.sum() == 0:
        raise ValueError("No wound region detected from tracking masks frame 0.")

    return float(cx), float(cy)

def assign_starting_layer_to_tracks(tracks_df, objects_df, max_distance=8.0):
    """
    Finds the first coordinate of each track and uses a KDTree to map it 
    to the corresponding spatial layer from the objects dataframe.
    """
    print("Mapping tracks to their starting spatial layers...")
    
    # Isolate the very first frame for each individual track
    first_frames = tracks_df.sort_values('t').groupby('trackID').first().reset_index()
    
    starting_layers = []
    
    # We group the objects by time to speed up the KDTree search
    objects_by_time = dict(tuple(objects_df.groupby('t')))
    
    for _, track_row in first_frames.iterrows():
        t_val = track_row['t']
        x_val, y_val = track_row['x'], track_row['y']
        
        # Check if we have layer objects for this specific frame
        if t_val not in objects_by_time:
            starting_layers.append(np.nan)
            continue
            
        frame_objects = objects_by_time[t_val]
        
        # Build KDTree for the objects in this specific frame
        tree = cKDTree(frame_objects[['x', 'y']].values)
        dist, idx = tree.query([x_val, y_val])
        
        # If the track point is within the 8.0 pixel tolerance, assign the layer
        if dist <= max_distance:
            starting_layers.append(frame_objects.iloc[idx]['layer_centroid'])
        else:
            starting_layers.append(np.nan)
            
    first_frames['Starting_Layer'] = starting_layers
    return first_frames[['trackID', 'Starting_Layer']]

def calculate_chemotactic_index(df, wound_center_x, wound_center_y):
    """
    Calculates the Chemotactic Index (CI) for each track.
    CI = Net Displacement projected toward the wound center / Total Accumulated Distance
    """
    ci_results = []
    
    # Group the dataframe by individual cells (trackID)
    grouped = df.groupby('trackID')
    
    for track_id, track_data in grouped:
        track_data = track_data.sort_values('t')
        
        if len(track_data) < 2:
            continue
            
        x_coords = track_data['x'].values
        y_coords = track_data['y'].values
        
        dx = np.diff(x_coords)
        dy = np.diff(y_coords)
        distances = np.sqrt(dx**2 + dy**2)
        d_total = np.sum(distances)
        
        if d_total == 0:
            continue 
            
        start_pos = np.array([x_coords[0], y_coords[0]])
        end_pos = np.array([x_coords[-1], y_coords[-1]])
        net_displacement_vec = end_pos - start_pos
        
        wound_center = np.array([wound_center_x, wound_center_y])
        wound_dir_vec = wound_center - start_pos
        
        wound_dir_norm = np.linalg.norm(wound_dir_vec)
        if wound_dir_norm == 0:
            continue
        wound_dir_unit = wound_dir_vec / wound_dir_norm
        
        d_net_projected = np.dot(net_displacement_vec, wound_dir_unit)
        ci = d_net_projected / d_total
        
        ci_results.append({
            'trackID': track_id,
            'D_total': d_total,
            'D_net_projected': d_net_projected,
            'Chemotactic_Index': ci
        })
        
    return pd.DataFrame(ci_results)

def calculate_neighbor_velocity_difference(df, spatial_radius_pixels, um_per_pixel, minutes_per_frame):
    """
    Calculates the alignment of velocity vectors between neighboring cells.
    Returns the average velocity difference for each cell at each time step in µm/min.
    """
    df = df.sort_values(['trackID', 't'])
    
    # Calculate time delta in actual minutes
    dt_frames = df.groupby("trackID")["t"].diff()
    dt_minutes = dt_frames * minutes_per_frame

    # Calculate physical displacement (dx, dy in µm) and divide by minutes
    df["vx"] = (df.groupby("trackID")["x"].diff() * um_per_pixel) / dt_minutes
    df["vy"] = (df.groupby("trackID")["y"].diff() * um_per_pixel) / dt_minutes

    df_vel = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["vx", "vy"]).copy()
    
    correlation_results = []
    
    for current_time, frame_data in df_vel.groupby('t'):
        if len(frame_data) < 2:
            continue
            
        positions = frame_data[['x', 'y']].values
        velocities = frame_data[['vx', 'vy']].values
        track_ids = frame_data['trackID'].values
        
        tree = cKDTree(positions)
        pairs = tree.query_pairs(r=spatial_radius_pixels)
        
        vel_diff_sums = {tid: 0.0 for tid in track_ids}
        neighbor_counts = {tid: 0 for tid in track_ids}
        
        for i, j in pairs:
            v_i = velocities[i]
            v_j = velocities[j]
            
            vector_distance = np.linalg.norm(v_i - v_j)
            
            id_i = track_ids[i]
            id_j = track_ids[j]
            
            vel_diff_sums[id_i] += vector_distance
            vel_diff_sums[id_j] += vector_distance
            neighbor_counts[id_i] += 1
            neighbor_counts[id_j] += 1
            
        for tid in track_ids:
            if neighbor_counts[tid] > 0:
                avg_diff = vel_diff_sums[tid] / neighbor_counts[tid]
                correlation_results.append({
                    'trackID': tid,
                    't': current_time,
                    'neighbors_found': neighbor_counts[tid],
                    'avg_velocity_vector_distance': avg_diff
                })
                
    return pd.DataFrame(correlation_results)

if __name__ == "__main__":
    # --- 1. CONFIGURATION & DATA LOADING ---
    input_file = converted_tracks_csv_path
    tracks_df = pd.read_csv(input_file)
    
    objects_path = f"{RESULTS_DIR}/{BASE_NAME}_objects_with_layers.csv"
    try:
        objects_df = pd.read_csv(objects_path)
    except FileNotFoundError:
        print(f"Error: {objects_path} not found. Run the layer assignment script first.")
        exit(1)

    # Physical Conversions (Ideally load from environment or config, hardcoded here as requested)
    UM_PER_PIXEL = float(os.environ.get("WOUND_UM_PER_PIXEL", "1.0"))
    MINUTES_PER_FRAME = float(os.environ.get("WOUND_MINUTES_PER_FRAME", "20.0")) # Adjust to your actual interval
    
    PHYSICAL_RADIUS_UM = 50.0 
    NEIGHBOR_RADIUS_PIXELS = PHYSICAL_RADIUS_UM / UM_PER_PIXEL
    # ---------------------------------------

    # --- 2. DYNAMIC CENTROID EXTRACTION ---
    print("\n--- Initializing Kinematics Analysis ---")
    try:
        WOUND_CENTER_X, WOUND_CENTER_Y = get_wound_centroid_from_tracking_masks(masks_tracking_path)
        print(f"--> Extracted Wound Centroid: X={WOUND_CENTER_X:.1f}, Y={WOUND_CENTER_Y:.1f}")
    except Exception as e:
        print(f"Failed to extract wound centroid dynamically: {e}")
        print("Falling back to center of image...")
        WOUND_CENTER_X, WOUND_CENTER_Y = 500.0, 500.0

    # --- 3. EXECUTE CALCULATIONS ---
    print(f"\nCalculating Chemotactic Index...")
    ci_df = calculate_chemotactic_index(tracks_df, WOUND_CENTER_X, WOUND_CENTER_Y)
    
    # Map layers and merge into the CI dataframe
    layer_mapping_df = assign_starting_layer_to_tracks(tracks_df, objects_df)
    ci_df = pd.merge(ci_df, layer_mapping_df, on='trackID', how='left')
    
    # Drop tracks that couldn't be mapped or started inside the wound (layer -1)
    ci_df = ci_df[(ci_df['Starting_Layer'] > 0)].dropna(subset=['Starting_Layer'])

    mean_ci = ci_df['Chemotactic_Index'].mean()
    print(f"--> Average Chemotactic Index: {mean_ci:.4f}")
    
    print(f"\nCalculating Neighbor Correlation (Radius = {PHYSICAL_RADIUS_UM}µm)...")
    corr_df = calculate_neighbor_velocity_difference(
        tracks_df, NEIGHBOR_RADIUS_PIXELS, UM_PER_PIXEL, MINUTES_PER_FRAME
    )
    
    mean_vector_dist = corr_df['avg_velocity_vector_distance'].mean()
    print(f"--> Average Neighbor Velocity Difference: {mean_vector_dist:.4f} µm/min")

    # --- 4. SAVE OUTPUTS ---
    ci_path = f"{RESULTS_DIR}/{BASE_NAME}_chemotactic_index.csv"
    corr_path = f"{RESULTS_DIR}/{BASE_NAME}_neighbor_velocity_difference.csv"

    ci_df.to_csv(ci_path, index=False)
    corr_df.to_csv(corr_path, index=False)

    print(f"\nAnalysis complete. Results saved to '{ci_path}' and '{corr_path}'.")