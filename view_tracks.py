import napari
import pandas as pd
import numpy as np
import btrack
from skimage.io import imread

from pipeline_config import input_tif_path, converted_tracks_csv_path

# --- CONFIGURATION ---
RAW_TIFF_PATH = input_tif_path
TRACKS_H5_PATH = converted_tracks_csv_path

def view_tracks():
    print(f"Loading napari viewer...")
    viewer = napari.Viewer()

    try:
        print(f"Loading raw image from {RAW_TIFF_PATH}...")
        raw_image = imread(RAW_TIFF_PATH)
        viewer.add_image(raw_image, name='Raw Image', blending='additive', colormap='gray')
    except Exception as e:
        print(f"Error loading raw image: {e}")
    
    print(f"Loading tracks from {TRACKS_H5_PATH}...")
    df = pd.read_csv(TRACKS_H5_PATH)

    tracks_data = df[['trackID', 't', 'y', 'x']].values

    properties = {'trackID': tracks_data[:, 0]}

    viewer.add_tracks(
        tracks_data,
        name='BTrack Results',
        properties=properties,
        color_by='trackID',
        tail_width=2,
        tail_length=30
    )

    print("Viewer open!")
    napari.run()

if __name__ == "__main__":
    view_tracks()