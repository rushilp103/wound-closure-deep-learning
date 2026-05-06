# Image Analysis and Segmentation of Wound Gap Closure

Python pipeline for time-lapse microscopy of wound closure: **Cellpose-SAM** instance segmentation, **per-frame object measurements**, **centroid-based wound layers**, **Bayesian tracking (btrack)**, **track export to CSV**, and **summary plots** (aspect ratio, speed, cell size). Optional **Napari** viewers help inspect layers and tracks.

## Requirements

- **Python 3** with `pip` (use a virtual environment).
- **PyTorch**: installed via `requirements.txt` (used by Cellpose).
- **GPU (optional)** — segmentation uses **Apple MPS**, **CUDA**, or **CPU** automatically (`cellpose_inference.py`).
- **Display**: Napari needs a working GUI if you use the View layers / View tracks scripts.

## Installation

From the repository root:

```bash
conda create -n wound-closure
conda activate wound-closure
pip install -r requirements.txt
```

Major dependencies include Cellpose, btrack, napari, textual, numpy/pandas, scikit-image, tifffile, matplotlib, and seaborn.

## Data layout and configuration

Paths are centralized in `pipeline_config.py`.


| Item        | Default                      |
| ----------- | ---------------------------- |
| Input movie | `{Data Dir}/{BASE_NAME}.tif` |
| Outputs     | `{Results Dir}/`             |


`**BASE_NAME**` is the dataset stem (no extension), e.g. `ctrl-1` reads `ctrl-1.tif`.

You can override defaults **without editing Python** using environment variables (also used by the Textual app when it launches scripts):


| Variable              | Meaning                             |
| --------------------- | ----------------------------------- |
| `WOUND_BASE_NAME`     | Dataset stem (default `ctrl-1`)     |
| `WOUND_DATA_DIR`      | Folder containing `{BASE_NAME}.tif` |
| `WOUND_RESULTS_DIR`   | Folder for all pipeline outputs     |
| `WOUND_BTRACK_CONFIG` | Path to `btrack_config.json`        |


Tracking parameters for btrack live in `**btrack_config.json`** at the project root unless overridden.

## Running the pipeline

### Option A: Terminal UI (recommended)

From the **project root**:

```bash
python textual_app.py
```

Configure **Base Name**, **Data Dir**, **Results Dir**, and **BTrack Cfg** in the UI (absolute paths work; you can paste or drag paths into the terminal). Set **µm/px**, **layer width (µm)**, **number of layers**, and **minutes per frame** as needed.


| Control           | Action                                                                                                                                  |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| **1 Cellpose**    | Run Cellpose-SAM on the stack; write mask TIFFs under results.                                                                          |
| **2 Masks→obj**   | Extract per-frame object features → CSV for tracking.                                                                                   |
| **3 Layers**      | Assign wound-relative layers → `*_objects_with_layers.csv`.                                                                             |
| **4 Track**       | Run btrack → `*_tracks.h5`.                                                                                                             |
| **5 H5→CSV**      | Convert tracks HDF5 → `*_converted_tracks.csv`.                                                                                         |
| **Run pipeline**  | Runs steps **1 → 5** in order (stops on failure).                                                                                       |
| **6 Plot**        | Build one plot (**aspect** / **speed** / **size**); previews a temp PNG; use **Save to** + **Save plot** to copy into your chosen path. |
| **7 View layers** | Open Napari with layers (requires step 3).                                                                                              |
| **8 View tracks** | Open Napari with tracks (requires step 5).                                                                                              |


Press **q** to quit.

### Option B: Command line (same order)

Run from the project root so imports resolve. With defaults from `pipeline_config.py` (or env vars set):

```bash
python cellpose_inference.py
python masks_to_objects.py
python assign_layers.py --um-per-pixel 1.0 --layer-width 49.0 --num-layers 10   # adjust as needed
python run_tracking.py
python convert_h5_results.py
python final_plots.py --plot aspect --o Results/your_plot.png                     # or speed / size
```

`assign_layers.py` accepts additional morphology and smoothing flags; see `python assign_layers.py --help`.

`final_plots.py` supports `--plot {aspect,speed,size}`, `--csv`, `--tracks`, `--um-per-pixel`, `--minutes-per-frame`, `--x-axis {frame,hours}`, and `--o` for output. Speed and size plots use physical units when `--um-per-pixel` is set; use `0` or omit where you want pixel-only scaling (see script help).

## How the pipeline works

1. **Segmentation (`cellpose_inference.py`)**
  Loads the multi-frame TIFF, estimates cell diameter from the first frame, runs **Cellpose-SAM** on each frame, and saves a **uint16 label stack** (`{BASE_NAME}_masks_TRACKING.tif`) plus a normalized visualization stack (`*_masks_VISUAL.tif`).
2. **Objects (`masks_to_objects.py`)**
  For each frame, runs region properties on labels, builds columns (`x`, `y`, `t`, areas, axis lengths, etc.), and writes `**{BASE_NAME}_objects.csv`** formatted for btrack.
3. **Layers (`assign_layers.py`)**
  Derives a **wound region** from the mask stack (morphological cleanup), assigns **centroid-based layer IDs** relative to the wound edge, and writes `**{BASE_NAME}_objects_with_layers.csv`**. Layer geometry uses **µm/px** and **layer width (µm)** from your parameters.  
   **Tracking** still consumes the original `_objects.csv`; layers live in the separate `_objects_with_layers.csv` for plots and Napari.
4. **Tracking (`run_tracking.py`)**
  Loads objects, converts rows to btrack `PyTrackObject` instances (contiguous IDs), runs the Bayesian tracker using `**btrack_config.json`**, and writes `**{BASE_NAME}_tracks.h5**`.
5. **HDF5 → CSV (`convert_h5_results.py`)**
  Reads tracks from the HDF5 file and writes `**{BASE_NAME}_converted_tracks.csv`** for plotting and visualization.
6. **Plots (`final_plots.py`)**
  **Aspect**: layer-stratified aspect ratio over time. **Speed**: merges track points to nearest labeled cells per frame, segments motion, plots speeds (uses **µm/px** when set). **Size**: elliptical cell area by zone vs. time. Defaults for time grouping match the analysis (see `DEFAULT_MINUTES_PER_FRAME` in the script).

### Visualization helpers

- `**view_layers.py`** — Napari: raw stack, masks, objects colored by layer (run after assign layers).
- `**view_tracks.py**` — Napari: raw stack + tracks from the converted CSV (run after H5→CSV).

### Other scripts

- `**compute_layer_aspect_ratios.py**` — auxiliary metrics from layer assignments (run manually if needed).

## Output files (under your Results directory)


| File                                                   | Produced by     |
| ------------------------------------------------------ | --------------- |
| `{BASE_NAME}_masks_TRACKING.tif`, `*_masks_VISUAL.tif` | Cellpose        |
| `{BASE_NAME}_objects.csv`                              | Masks → objects |
| `{BASE_NAME}_objects_with_layers.csv`                  | Assign layers   |
| `{BASE_NAME}_tracks.h5`                                | Tracking        |
| `{BASE_NAME}_converted_tracks.csv`                     | Convert H5      |
| `{BASE_NAME}*plot*{aspect                              | speed           |


## Troubleshooting

- **Missing input** — Ensure `{Data Dir}/{BASE_NAME}.tif` exists and matches **Base Name** (case-sensitive on Linux/macOS).
- **Empty or wrong outputs** — Confirm **Results Dir** is writable and you are not mixing `BASE_NAME` between runs.
- **btrack / HDF5 errors** — If conversion fails with a message about object IDs, the tracker export expects contiguous object IDs (see `convert_h5_results.py`); rerun from masks → objects → track without ad hoc CSV edits that break ID order.

