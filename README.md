# 3D Electronic Analysis for Site-Selectivity

![](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![](https://img.shields.io/badge/License-MIT-orange)

This repository contains the data-processing and modeling workflow used for
3D electronic descriptor analysis of ketone reductions and prediction of
site-/stereoselectivity.

The project is organized as a sequential pipeline:

1. `libs/dataset.py` + `libs/eda.py`
2. `libs/calc_mol.py`
3. `libs/calc_grid.py`
4. `libs/regression.py`
5. `libs/graph.py`

## What This Pipeline Does
- Cleans and standardizes experimental data from Excel.
- Runs molecule-level quantum chemistry jobs (Gaussian) for each substrate.
- Converts cube-file fields into aligned/folded 3D grid descriptors.
- Trains regression models (Lasso/Ridge/ElasticNet/PLS/OMP) with LOOCV.
- Generates publication figures (parity plots, contribution plots, 3D views).

## Repository Structure
- `libs/dataset.py`: dataset loading, cleaning, export, and baseline plots.
- `libs/eda.py`: exploratory and manuscript-supporting plots.
- `libs/calc_mol.py`: conformer generation + Gaussian workflow + cube generation.
- `libs/calc_grid.py`: grid descriptor extraction/aggregation from cube files.
- `libs/regression.py`: model training, LOOCV, prediction, coefficient export.
- `libs/graph.py`: evaluation, contribution analysis, and final visualization.
- `libs/render_molecule.py`: Py3Dmol helper utilities for molecular/grid rendering.
- `definition_grid.ipynb`: notebook for grid-definition/visual inspection.
- `run_pipeline.ipynb`: step-by-step execution notebook for the full workflow.
- `data/`: input datasets and generated analysis/model outputs.

## Environment Setup
A Conda environment file is provided:

```bash
conda env create -f environment.yml
conda activate 3D_electronic_analysis_for_site-selecticity
```

Notes:
- The environment name follows `environment.yml` as-is.
- Full pipeline execution requires Gaussian utilities (`g16`, `formchk`, `cubegen`) to be available in your shell environment.

## Run the Pipeline
### Option A: Notebook (recommended for stepwise execution)
Open and run:
- `run_pipeline.ipynb`

This notebook already executes scripts in the required order with explanatory cells.

### Option B: Command line
Run from repository root:

```bash
python libs/dataset.py
python libs/eda.py
python libs/calc_mol.py
python libs/calc_grid.py
python libs/regression.py
python libs/graph.py
```

## Runtime Configuration
### `libs/calc_mol.py`
Defaults:
- `NUM_THREADS`: detected CPU core count
- `MEMORY_GB`: half of detected RAM
- `OUTPUT_ROOT`: `~/molecules`
- `GAUSSIAN_RUN_COMMAND`: `source ~/.bash_profile && g16`

Environment-variable overrides:
- `GAUSSIAN_NUM_THREADS`
- `GAUSSIAN_MEMORY_GB`
- `GAUSSIAN_RUN_COMMAND`

### `libs/calc_grid.py`
Defaults:
- `OUTPUT_ROOT`: `~/molecules`
- `NUM_WORKERS`: detected CPU core count

Override:
- `GRID_NUM_WORKERS`

### `libs/regression.py`
Defaults:
- `INPUT_DATA_PATH`: `data/data.pkl`
- `NUM_WORKERS`: detected CPU core count

Override:
- `REGRESSION_NUM_WORKERS`

## Main Input/Output Flow
- Input raw table: `data/all_experimental_data.xlsx`
- After `dataset.py`: `data/data.xlsx`
- After `calc_grid.py`: `data/data.pkl`, `data/datafeat.csv`
- After `regression.py`: regression summary files with suffix
  `_electronic_electrostatic_lumo_regression.*` and `_results.csv`
- After `graph.py`: validation/contribution figures under `data/validation/` and `data/test/`

## Computational Notes
- `calc_mol.py` can be the most expensive step (conformer generation + QC jobs).
- For large batches, set worker/thread counts explicitly to match available hardware.
- If running remotely, set `GAUSSIAN_RUN_COMMAND` to your SSH submission command.

## Original Article
Daimon Sakaguchi, Taisei Kawasaki, Mayu Itakura, Chihiro Tada, and Hiroaki Gotoh,
*Competition Experiment-Based Kinetic Analysis of Ketone Reductions and Prediction of Site- and Stereoselectivity*, 2026 (submitted).

## License
This project is available under the [MIT License](LICENSE.txt).
