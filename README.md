# Physical Intelligence Figure Code Export

This repository contains the cleaned code and compact data exports used to
reproduce the visualization notebooks for Figures 2-5.

## Contents

- `data/figure2_gaussian_impulse.npz`: Gaussian impulse simulation dataset,
  initial design, optimized design at index 95, loss record, and CNN prediction
  examples.
- `data/figure3_experimental_force.npz`: Experimental force dataset, simulation
  IMU data, initial design, minimum-loss optimized design, loss record, and CNN
  prediction examples.
- `data/figure4_sensor_count_losses.npz`: Loss summaries for 4- and 8-input
  source co-optimization runs across sensor counts.
- `data/figure5_structure_complexity_losses.npz`: Circular-structure complexity
  and prediction-loss summaries.
- `notebooks/figure2_gaussian_impulse.ipynb`: Figure 2 visualization notebook.
- `notebooks/figure3_experimental_force.ipynb`: Figure 3 visualization notebook.
- `notebooks/figure4_sensor_count_loss.ipynb`: Figure 4 visualization notebook.
- `notebooks/figure5_circular_structure_complexity.ipynb`: Figure 5 visualization notebook.
- `scripts/cooptimization_input_source_4_multiple_sensors.py`: 4-input-source
  co-optimization script.
- `scripts/cooptimization_input_source_8_multiple_sensors.py`: 8-input-source
  co-optimization script.
- `scripts/cooptimization_circular_robot_multiple_sensors.py`: Circular-structure
  co-optimization script.
- `scripts/export_figure_data.py`: Regenerates the compact `data/*.npz` files
  from the original `physical-intelligence-change` folder.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Regenerate Data

From this repository root:

```bash
python scripts/export_figure_data.py \
  --original-root "/home/kna35/GaTech Dropbox/Kyungmi Na/physical-intelligence-change"
```

The exporter only reads the original folder and writes new files under `data/`.

## Run Notebooks

Open the notebooks in `notebooks/` from the repository root, or start Jupyter:

```bash
jupyter lab
```
