import btrack
import pandas as pd
import numpy as np
import os

from pipeline_config import tracks_h5_path, converted_tracks_csv_path

# --- CONFIGURATION ---
H5_FILE_PATH = tracks_h5_path
OUTPUT_CSV = converted_tracks_csv_path


def convert_h5_to_csv():
    print(f'Opening HDF5 file: {H5_FILE_PATH}')

    with btrack.io.HDF5FileHandler(H5_FILE_PATH, 'r', obj_type='obj_type_1') as reader:
        try:
            tracks = reader.tracks
        except IndexError as e:
            raise RuntimeError(
                "Failed to read tracks from HDF5: track references don't match stored objects. "
                "This usually happens when PyTrackObject.ID isn't 0..N-1 contiguous during export."
            ) from e
        print(f'    > Loaded {len(tracks)} tracks from HDF5.')

        if len(tracks) == 0:
            print("No tracks found in the HDF5 file.")
            return
        
        print('Converting tracks to DataFrame...')
        tracks_data = []

        for track in tracks:
            track_id = track.ID
            parent = track.parent
            root = track.root

            for i in range(len(track)):
                tracks_data.append({
                    'trackID': track_id,
                    't': track.t[i],
                    'x': track.x[i],
                    'y': track.y[i],
                    'z': 0,
                    'parent': parent,
                    'root': root,
                    'dummy': track.dummy[i]
                })
    df = pd.DataFrame(tracks_data)
    df = df.sort_values(['trackID', 't'])

    df.to_csv(OUTPUT_CSV, index=False)
    print(f'SUCCESS! Converted {len(df)} rows.')
    print(f'Saved to: {OUTPUT_CSV}')

if __name__ == "__main__":
    convert_h5_to_csv()