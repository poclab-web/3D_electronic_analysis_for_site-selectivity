# Accepted-model data package

This directory is the self-contained data and reference-result package for the
model reported in the accompanying manuscript. Canonical reproduction does not
read historical `analysis_runs/`, legacy pickle files, or the external Gaussian
archive.

## Reproduction contract

1. Verify every immutable input against `inputs/input_manifest.csv`.
2. Load 161 frozen molecular rows and three aligned descriptor blocks.
3. Refit all 83 strict nested outer leave-one-out models.
4. Regenerate the full-training predictions, a--f diketone evaluation, compact
   result tables, and reference figures.
5. Compare the outputs with `expected_metrics.json` by running
   `scripts/verify_reproduction.py`.

Within each outer fold, the holdout is excluded from raw-grid scaling, spatial
selection, summary-feature scaling, and Lasso-alpha selection. A complete
reproduction must not use `--skip-nested`.

## Directory structure

- `inputs/`: immutable descriptor arrays, molecular metadata, training
  identities, provenance, and SHA-256 manifest.
- `results/model/`: nested-LOOCV predictions, full-training fit, coefficients,
  excluded-substrate predictions, and primary summary metrics.
- `results/diketones/`: full-training and outer-model diketone predictions,
  identity checks, semiquantitative evaluation, and uncertainty summaries.
- `results/publication_tables/`: compact derived tables used to construct
  manuscript and Supporting Information figures.
- `audits/`: descriptor-coordinate alignment and active-workbook refresh
  records. These document data transformations but are not model features.
- `comparators/`: reviewed, compact summaries of prespecified comparator
  models.
- `spatial_analysis/`: feature-level coefficients, realized effects, spatial
  summaries, and compressed matrices.
- `../validation/current_model/`: reference figures grouped by analysis type.

`model_specification.json` records the complete computational specification;
`expected_metrics.json` records numerical invariants and comparison tolerances.

## Accepted model

| Item | Specification |
| --- | --- |
| Training observations | 83 |
| Frozen metadata rows | 161 |
| Prediction rows | 161 |
| Descriptor blocks | electronic, electrostatic, projected C=O pi-star |
| Full-grid coordinates | 4,927 per block |
| Selected coordinates | 105 per block |
| Features | 321: `(105 + max + min) x 3` |
| Estimator | Lasso |
| Alpha candidates | 1, 0.1, 0.01, 0.001 |
| Full-training alpha | 0.01 |
| Validation | strict nested outer LOOCV |
| Nested outer R2 | 0.8037754478337535 |
| Nested outer RMSE | 0.5811724454040451 kcal/mol |
| Nested outer MAE | 0.43514599510153823 kcal/mol |
| Diketone evaluation | a-f |
| Semiquantitative RMSE | 13.122564725906527 percentage points |

The withdrawn x/y series and series g are not part of the reported evaluation.
The 2026-07-26 workbook refresh removed six non-training monoketone records;
the 83 training identities and experimental responses were unchanged.

## Canonical commands

Run from the repository root:

```bash
python libs/current_model.py --verify-inputs-only

OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
python libs/current_model.py \
  --workers 4 \
  --no-excel-refresh \
  --skip-contribution-cubes

python libs/analyze_current_model_spatial_contributions.py \
  --workers 4 \
  --no-excel-refresh

python scripts/verify_reproduction.py
```

`--no-excel-refresh` uses the versioned metadata and responses exactly as
stored. Without it, current workbook labels and responses are joined in memory
by InChIKey and the differences are written to `audits/`; the immutable input
package is never modified.

`--skip-contribution-cubes` omits only optional display cubes. Nested LOOCV,
predictions, compact tables, and versioned figures are unaffected.

## Principal outputs

- `results/model/summary.csv`: primary model and validation metrics.
- `results/model/outer_predictions.csv`: all 83 strict outer predictions.
- `results/model/fulltrain_inner_alpha_path.csv`: full-training inner-LOOCV
  alpha comparison.
- `results/model/nonzero_coefficients.csv`: nonzero full-training terms.
- `results/model/fulltrain_predictions_and_contributions.csv`: fitted values
  and block contributions.
- `results/diketones/diketone_predictions_by_outer_model.csv`: diketone
  predictions from every outer model.
- `results/diketones/diketone_semiquant_detail.csv`: experimental comparison
  for a-f.
- `results/diketones/diketone_primary8_outer83_68_interval.csv`: outer-model
  uncertainty summary.
- `results/model_comparison_current_vs_orbital_free.csv`: accepted/comparator
  model table.

## Updating the package

Gaussian-derived descriptors for new molecules must be generated outside the
repository. Only reviewed portable NPZ, CSV, or JSON inputs may be promoted
into `inputs/`. Every promotion requires an updated input manifest, explicit
provenance, an updated model specification where applicable, and a complete
reproduction check. Gaussian outputs and temporary caches remain excluded by
the repository data policy.

The historical Multiwfn version was not recorded. The checksummed frozen
descriptor package is therefore the canonical source for the reported
results. New calculations must record the Gaussian and Multiwfn versions in
their run manifest.
