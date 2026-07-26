# Accepted-model reference results

This directory contains compact, versioned outputs of the accepted model. The
files provide independently inspectable predictions, coefficients, uncertainty
summaries, and publication tables; they are not runtime inputs.

- `model/`: full-training and strict nested-LOOCV results.
- `diketones/`: a-f predictions, identity checks, semiquantitative
  evaluation, and outer-model uncertainty.
- `publication_tables/`: compact derived values used in manuscript or
  Supporting Information figures.
- `model_comparison_current_vs_orbital_free.csv`: reviewed comparison with the
  prespecified orbital-free model.

All values can be regenerated from the frozen inputs. Primary numerical
invariants are checked by `scripts/verify_reproduction.py` against
`../expected_metrics.json`.
