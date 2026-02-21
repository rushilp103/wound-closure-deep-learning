import btrack
import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
H5_FILE_PATH = '/Users/Rushilp/Downloads/UMass/Honors Thesis/Data Sets/Cellpose-SAM Results/ctrl-1_tracks.h5'
OUTPUT_CSV = '/Users/Rushilp/Downloads/UMass/Honors Thesis/Data Sets/Cellpose-SAM Results/ctrl-1_converted_tracks.csv'

def convert_h5_to_csv():
    print(f'Opening HDF5 file: {H5_FILE_PATH}')

    with btrack.io.HDF5FileHandler(H5_FILE_PATH, 'r', obj_type='obj_type_1') as reader:
        tracks = reader.tracks
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