# Superseded exploratory pipeline

This directory contains the historical three-field model-search and plotting
code retained for methodological provenance. It does not define the accepted
model, is not imported by `libs/current_model.py`, and is excluded from the
canonical reproduction workflow.

- `calc_grid.py`: historical cube-to-grid descriptor aggregation.
- `dataset.py`: historical workbook export and exploratory dataset plots.
- `eda.py`: exploratory structure/reactivity and physical-organic plots.
- `graph.py`: historical regression, contribution, and kinetic figures.
- `regression.py`: superseded multi-estimator model-search pipeline.
- `render_molecule.py`: legacy 3Dmol annotation helpers.
- `predict_external_diketone_xy.py`: archived generator and predictor for the
  x/y diketone series withdrawn from the current manuscript scope.

Legacy outputs are written below the Git-ignored `data/legacy/` directory.
These scripts are retained to document the analysis history; their validation
strategy predates the strict fold-local scaling, spatial selection, and alpha
selection implemented by the accepted model. Results produced by this
directory must not be reported as results of the publication model.
