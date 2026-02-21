import numpy as np
import torch
from cellpose import models
import tifffile
import os
import cv2
import time
from tqdm import tqdm
import logging
from skimage.measure import regionprops

logging.getLogger('cellpose').setLevel(logging.ERROR)

# Configuration
input_path = '/Users/Rushilp/Downloads/UMass/Honors Thesis/Data Sets/ctrl-1.tif'
output_dir = '/Users/Rushilp/Downloads/UMass/Honors Thesis/Data Sets/Cellpose-SAM Results'
file_name = 'ctrl-1'

# Device setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
    use_gpu = True
    print("Using Apple MPS for computation.")
else:
    device = torch.device("cpu")
    use_gpu = False
    print("Using CPU for computation.")

# Load Image Stack
imgs = tifffile.imread(input_path)
print(f"Loaded image stack shape: {imgs.shape}")

# --- SUBSET TOGGLE ---
# imgs_subset = imgs[:15] # Process the first 15 frames
imgs_subset = imgs # Production Mode: Process all frames
# ----------------------

print(f"Processing stack shape: {imgs_subset.shape}")

# Initialize Cellpose-SAM model
model = models.CellposeModel(gpu=use_gpu, device=device, model_type='cpsam')

print("-" * 40)
print("Auto-Calibrating Diameter...")

# Determine optimal diameter using first frame
seed_diameter = 30.0
print(f"  > Running initial pass with guess: {seed_diameter}px...")
seed_masks, _, _ = model.eval(
    imgs_subset[0], 
    diameter=seed_diameter, 
    channels=[0, 0],
    flow_threshold=0.4
)

regions = regionprops(seed_masks.astype(int))

if len(regions) > 0:
    # Calculate the average equivalent diameter of all found cells
    measured_diams = [prop.equivalent_diameter_area for prop in regions]

    # Filter out tiny specks (<10px) that might skew the average
    valid_diams = [d for d in measured_diams if d > 10]
    
    if len(valid_diams) > 0:
        real_diameter = np.mean(valid_diams)
        print(f"  > Detected {len(valid_diams)} cells.")
        print(f"  > Measured Average Diameter: {real_diameter:.2f} pixels")
        estimated_diameter = real_diameter
    else:
        print("  > Warning: Only debris found. Defaulting to 30.0.")
        estimated_diameter = 30.0
else:
    print("  > Warning: No cells found in first frame. Defaulting to 30.0.")
    estimated_diameter = 30.0

print(f"  > Setting diameter to {estimated_diameter:.2f} for consistent tracking.")
print("-" * 40)

# --- MAIN LOOP ---
all_masks = []
total_start = time.time()

with tqdm(imgs_subset, desc="Segmenting Frames", unit="frame") as pbar:
    for i, frame in enumerate(pbar):
        frame_start = time.time()
        
        masks, flows, styles = model.eval(
            frame,
            diameter=estimated_diameter,
            channels=[0, 0],
            flow_threshold=0.4, 
            cellprob_threshold=0.0,
        )

        frame_duration = time.time() - frame_start
        all_masks.append(masks)

        pbar.set_postfix({"Last Frame": f"{frame_duration:.2f}s"})

total_end = time.time()
print("-" * 40)
print(f"Total segmentation time: {total_end - total_start:.2f} seconds")

# Convert to 3D stack (T, Y, X)
masks_stack = np.array(all_masks).astype(np.uint16)

# Brightness normalization
visual_stack = cv2.normalize(
    masks_stack,
    None,
    alpha=0,
    beta=255,
    norm_type=cv2.NORM_MINMAX,
    dtype=cv2.CV_8U
)

# Save the segmentation results
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

tracking_path = os.path.join(output_dir, f"{file_name}_masks_TRACKING.tif")
tifffile.imwrite(tracking_path, masks_stack, imagej=True, metadata={'axes': 'TYX'})

visual_path = os.path.join(output_dir, f"{file_name}_masks_VISUAL.tif")
tifffile.imwrite(visual_path, visual_stack, imagej=True, metadata={'axes': 'TYX'})

print(f"Segmentation completed. Saved {masks_stack.shape[0]} frames to {output_dir}.")