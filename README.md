# Physical Intelligence: Sensing and Co-optimization Tutorial

This repository is a modular walkthrough of the sensing/co-optimization
pipeline behind the paper: forward simulation of each structure type, a
neural-network readout trained on that simulation data, a comparison of
readout performance before vs. after design optimization, and an
illustration of how the design gradient itself is computed.

## Contents

### Core library

- `blockymetamaterials/`: the block-metamaterial physics engine (geometry,
  energy, dynamics) used throughout.
- `cooptimization_real.py`, `neural_networks.py`: shared low-level dynamics-
  solver helpers and the `TimeSeriesCNN` / `minmax_scaler` definitions reused
  by every notebook and script below.
- `forward_simulation/`: modular forward-simulation package.
  - `solver.py`: `setup_dynamic_solver_multi` (a dynamics solver that reads
    out acceleration (x, y) and angular velocity at an explicit list of
    sensor blocks) and `build_forward_problem` (drives it with a real,
    interpolated force time series, vmap-able over samples).
  - `structures.py`: one config/builder function per structure —
    `build_input_source_2/4/8()` and `build_circular_structure()` — each
    returning a `StructureSetup` with that structure's geometry, initial
    design, mechanical parameters, and DOF setup.

### Data

- `data/forward_sim_{input_source_2,input_source_4,input_source_8,circular_structure}.npz`:
  a small sample of real co-optimization force data (`output_data`) paired
  with the originally recorded sensor response (`input_data`), used both to
  drive the forward-simulation notebooks and as ground truth for their
  sanity checks.
- `data/figure2_gaussian_impulse.npz`: co-optimization loss curve and design
  snapshots (initial + optimized) for the simulated Gaussian-impulse case.
- `data/gaussian_impulse_nn_comparison.npz`: the initial and optimized
  design's simulated sensor response to the *same* force samples, used to
  train and compare a CNN readout for each design on equal footing.

### Notebooks

1. **Forward simulation** — `forward_simulation_input_source_2.ipynb`,
   `_input_source_4.ipynb`, `_input_source_8.ipynb`, `_circular_structure.ipynb`.
   Each: visualizes the structure's initial design, runs the modular forward
   simulation driven by real co-optimization force data, and sanity-checks
   the result against the originally recorded response for that same force.
   All four reproduce the recorded response to near machine precision on
   every sample.
2. **Neural-network training** — `scripts/train_neural_network.py` (see
   below; not a notebook, per the "python code" ask) and
   `train_nn_gaussian_impulse.ipynb`, which trains a fresh CNN for the
   initial and optimized design (one shared train/test split and scaler,
   fit on the initial design) and plots test-loss history for both.
3. **Comparing initial vs. optimized design** — `compare_gaussian_impulse_nn.ipynb`.
   Loads the co-optimization loss curve, then trains a fresh CNN per design
   (same shared-split/shared-scaler approach) and compares test loss and
   example predictions before vs. after co-optimization.
4. **Computing the design gradient** — `gradient_computation_demo.ipynb`.
   A small, faithful illustration of the bilevel mechanism used for real
   co-optimization (inner CNN training via `jaxopt.OptaxSolver` with
   `implicit_diff=True`, outer design gradient via
   `jax.value_and_grad`/a second `OptaxSolver`), built from the same
   `forward_simulation` package and `neural_networks.py` as everything else
   — not the real production script.

### Scripts

- `scripts/train_neural_network.py`: trains a `TimeSeriesCNN` to predict
  force from sensor response, given one structure's `forward_sim_*.npz`.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
jupyter lab   # open any notebooks/*.ipynb from the repository root

python scripts/train_neural_network.py --structure input_source_2
python scripts/train_neural_network.py --structure input_source_4
python scripts/train_neural_network.py --structure input_source_8
python scripts/train_neural_network.py --structure circular_structure
```

Every notebook and script here runs out of the box using the bundled
`data/*.npz` files — no private data required.

## Sanity-Check Scripts

```bash
python -m py_compile forward_simulation/*.py scripts/*.py cooptimization_real.py neural_networks.py
```
