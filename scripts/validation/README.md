# Structural-class validation

`leave_one_structural_class_out.py` performs the finalized five-fold internal
validation in which each A--E ketone structural class is withheld completely.
It also reports interpretation-stability and diketone-transfer results.

`nested_structural_class_method_comparison.py` evaluates PLS, Ridge, Elastic
Net, Lasso, and OMP under the same outer structural-class splits, with
observation-level inner LOOCV used for method-specific hyperparameter
selection. The finalized Lasso grid is used so the Lasso result is identical
to the procedure reported in Figure S13B and Table S13.

Both scripts read the checksummed frozen inputs under `data/current_model/`
and write only to an explicitly supplied output directory.
