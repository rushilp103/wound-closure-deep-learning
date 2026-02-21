import numpy as np
import pandas as pd
import tifffile
from skimage.measure import regionprops_table
import os
from tqdm import tqdm

# Configuration
input_masks_path = '/Users/Rushilp/Downloads/UMass/Honors Thesis/Data Sets/Cellpose-SAM Results/ctrl-1_masks_TRACKING.tif'
output_objects_dir = '/Users/Rushilp/Downloads/UMass/Honors Thesis/Data Sets/Cellpose-SAM Results'
file_name = 'ctrl-1_objects.csv'

# Pixel calibration
SCALE_X = 1.0
SCALE_Y = 1.0

def extract_objects(masks_path, output_path):
    print(f"Loading masks from: {masks_path}...")
    masks = tifffile.imread(masks_path)
    print(f"  > Loaded masks shape: {masks.shape}")

    all_data = []

    print("Extracting cell coordinates...")
    for t in tqdm(range(masks.shape[0]), desc="Procesing frames"):
        frame_mask = masks[t]

        props = regionprops_table(
            frame_mask,
            properties=('label', 'centroid', 'area', 'major_axis_length', 'minor_axis_length', 'orientation'),
        )

        df = pd.DataFrame(props)

        # Need to add time column for btrack
        df['t'] = t

        all_data.append(df)
    
    all_data_df = pd.concat(all_data, ignore_index=True)

    # Formatting for btrack
    all_data_df.rename(columns={
        'centroid-1': 'x',
        'centroid-0': 'y',
        'label': 'obj_id'
    }, inplace=True)

    # Apply pixel scaling
    all_data_df['x'] = all_data_df['x'] * SCALE_X
    all_data_df['y'] = all_data_df['y'] * SCALE_Y

    print(f'Found {len(all_data_df)} total cell observations across {masks.shape[0]} frames.')

    # Save to CSV
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    all_data_df.to_csv(output_path, index=False)
    print(f"Saved extracted object data to: {output_path}")

if __name__ == "__main__":
    output_full_path = os.path.join(output_objects_dir, file_name)
    extract_objects(input_masks_path, output_full_path)