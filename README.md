# 3D Electronic Analysis for Ketone-Reduction Selectivity

![Python 3.10](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![License: MIT](https://img.shields.io/badge/License-MIT-orange)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20060463.svg)](https://doi.org/10.5281/zenodo.20060463)

This repository accompanies the manuscript *Competition-Derived Relative
Reactivity and 3D Electronic-State Analysis of Site- and Facial Selectivity in
NaBH4/MeOH Ketone Reductions*. It provides the
code, immutable descriptor inputs, reference predictions, and validation
artifacts required to reproduce the accepted three-dimensional electronic
model without rerunning the underlying quantum-chemical calculations.

The canonical reproduction entry point is `libs/current_model.py`. The frozen
package in `data/current_model/inputs/` is the authoritative computational
input. Historical model-search code is retained separately in `libs/legacy/`
and does not define any reported result.

The compact transition-state coordinate and energy workbook under
`data/transition_states/` supports the quantum-chemical mechanistic analysis.
It is not an input to the machine-learning model.

## Scientific scope

The accepted model combines three aligned spatial descriptor blocks:

- electronic density;
- electrostatic potential; and
- a HOMO-gap-damped projected C=O pi-star orbital descriptor.

Each block begins with 4,927 frozen grid coordinates. Within every training
fold, 105 coordinates per block are selected from the predefined compact
domain and augmented by block-specific maximum and minimum summaries. The
resulting 321 features are fitted with Lasso regression. Hyperparameter
selection is repeated independently within each outer leave-one-out fold, so
the held-out observation is excluded from scaling, spatial selection, and
alpha selection.

The reported diketone evaluation covers series a--f. No external prediction
series is appended to the accepted-model matrix.

## Repository contents

```text
data/
  Details_of_experimental_results.xlsx   Experimental source workbook
  current_model/
    inputs/                              Immutable, checksummed inputs
    results/model/                       Model fit and nested-LOOCV results
    results/diketones/                   Diketone predictions and evaluation
    results/publication_tables/          Compact tables used in figures
    audits/                              Input-alignment and workbook audits
    comparators/                         Reviewed comparator summaries
    spatial_analysis/                    Spatial coefficients and effects
  transition_states/                     TS coordinates and energies; not ML input
  validation/current_model/              Reference figures by analysis type
libs/
  current_model.py                       Canonical model runner
  analyze_current_model_spatial_contributions.py
  export_current_model_contribution_cubes.py
  diketone_metrics.py
  current_model_support/                 Accepted-model support modules
  legacy/                                Superseded and withdrawn workflows
scripts/
  verify_reproduction.py                 Independent repository verification
  maintenance/                           One-time data-maintenance utilities
  viewers/                               Optional GaussView helpers
tests/
  test_reproducibility.py                Lightweight regression tests
```

Large Gaussian outputs, checkpoint files, cube files, scratch directories,
exploratory runs, and editable manuscript files are intentionally excluded.
The inclusion policy is documented in `docs/DATA_POLICY.md`.

## Environment

Create the versioned Conda environment from the repository root:

```bash
conda env create -f environment.yml
conda activate 3d-electronic-analysis
```

The canonical reproduction uses frozen NPZ, CSV, and JSON inputs and does not
require Gaussian, GaussView, or Multiwfn.

## Canonical reproduction

### 1. Verify immutable inputs

```bash
python libs/current_model.py --verify-inputs-only
```

This command checks every file declared in
`data/current_model/inputs/input_manifest.csv`, including byte size, SHA-256,
array schema, coordinate alignment, and training identities.

### 2. Refit the model and rerun strict nested LOOCV

```bash
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
python libs/current_model.py \
  --workers 4 \
  --no-excel-refresh \
  --skip-contribution-cubes
```

`--workers` may be adjusted from 1 to 20. `--no-excel-refresh` fixes the run
to the versioned metadata and responses. `--skip-contribution-cubes` omits
only optional display cubes; it does not change model fitting, predictions,
metrics, compact result tables, or reference PNGs.

Do not use `--skip-nested` for a full reproduction. That option reuses saved
outer predictions and is intended only for rapid output checks.

### 3. Recompute the spatial analysis

```bash
python libs/analyze_current_model_spatial_contributions.py \
  --workers 4 \
  --no-excel-refresh
```

### 4. Verify saved results independently

```bash
python scripts/verify_reproduction.py
python -m unittest discover -s tests -p 'test_*.py'
```

The verifier rebuilds the portable 161-by-321 feature matrix, recalculates
metrics from all 83 outer predictions, and
validates the spatial coefficient and effect matrices. `run_pipeline.ipynb`
provides a small notebook interface to the same command-line workflow.

## Reference results

Machine-readable values and tolerances are versioned in
`data/current_model/expected_metrics.json`.

| Quantity | Reference value |
| --- | ---: |
| Frozen metadata rows | 161 |
| Training observations | 83 |
| Prediction rows | 161 |
| Full-grid coordinates per block | 4,927 |
| Selected coordinates per block | 105 |
| Total features | 321 |
| Full-training Lasso alpha | 0.01 |
| Strict nested outer-LOOCV R2 | 0.8037754478 |
| Strict nested outer-LOOCV RMSE | 0.5811724454 kcal/mol |
| Strict nested outer-LOOCV MAE | 0.4351459951 kcal/mol |
| Diketone a-f semiquantitative RMSE | 13.12256473 percentage points |

## Generating descriptors for new molecules

Gaussian 16, `formchk`, `cubegen`, and Multiwfn are required only when new
molecules are introduced. The reference protocol uses RDKit ETKDG/MMFF94
conformer generation, B3LYP-D3(BJ)/def2-SVP optimization and frequency
analysis, and a `wB97XD/def2-TZVP` SMD(methanol) single point. A minimal,
inspectable example is provided in `examples/gaussian/`.

Production quantum-chemical files must be stored outside the repository. The
withdrawn x/y generator is retained under `libs/legacy/` solely for provenance
and is not an active supported command. Its frozen inputs and last result
snapshot are documented under `data/archive/xy_diketones/`.

The historical Multiwfn version used to create the frozen descriptors was not
recorded; the checksummed accepted inputs therefore constitute the canonical
source for the reported model.

## Legacy code

`libs/legacy/` contains the superseded three-field exploratory pipeline. It is
retained for provenance and is not imported by the canonical model. Legacy
outputs are written below the ignored `data/legacy/` directory.

## Citation

Daimon Sakaguchi, Taisei Kawasaki, Mayu Itakura, Chihiro Tada, and Hiroaki
Gotoh, *Kinetics-Based Framework for Predicting Site- and Facial-Selectivity in
Ketone Reductions* (submitted, 2026).

## License

The source code is distributed under the MIT License. Experimental and derived
research data should be cited with the accompanying manuscript.
