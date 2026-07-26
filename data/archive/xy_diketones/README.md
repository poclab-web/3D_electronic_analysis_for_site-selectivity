# Archived x/y diketone analysis

This directory preserves the x and y diketone prediction inputs and the last
pre-withdrawal result snapshot. The series were removed from the current
manuscript scope on 2026-07-25 and are not loaded, verified, or reported by the
active a--f-only reproduction pipeline.

- `inputs/`: frozen x/y metadata and three-block descriptor arrays.
- `results/`: the last combined a--f/x/y tables retained for provenance.
- `figures/`: the last x/y-specific figures retained for provenance.
- `MANIFEST.csv`: byte sizes and SHA-256 hashes recorded at archival time.

The historical generator is
`libs/legacy/predict_external_diketone_xy.py`. It is an archived snapshot, not
an active supported command. Git history remains the authority for reconstructing
the complete pre-withdrawal runtime.

None of these files are model-training inputs. Removing x/y from the active
prediction scope does not alter the 83 training observations, 321 features,
Lasso coefficients, or strict nested-LOOCV metrics.
