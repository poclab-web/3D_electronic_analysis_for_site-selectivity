# 3D_electronic_analysis_for_site-selectivity

![](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![](https://img.shields.io/badge/License-MIT-orange)
[![](https://img.shields.io/badge/DOI--brightgreen)]()

Here’s an English version of the README “Contents” section based on those Python files:

## Contents

This repository is organized as follows:

- `libs/eda.py`  
  Exploratory data analysis utilities for the competitive ketone-reduction dataset.  
  Provides functions to simulate and visualize reaction progress (stacked concentration plots for branched mechanisms), compare DFT barriers and experimental data, and generate diagnostic plots used in the manuscript and SI.

- `libs/dataset.py`  
  Dataset I/O and preprocessing utilities.  
  Loads the original Excel files, constructs RDKit `Mol` objects from SMILES, generates InChIKeys, removes invalid entries, exports cleaned tables with RDKit structure thumbnails, and produces overview plots such as ΔΔG‡ vs. \(k_2/k_1\) and simple correlation plots.

- `libs/calc_mol.py`  
  Molecular-level quantum-chemical workflow.  
  Generates conformers from SMILES, prepares and manages quantum-chemistry calculations (e.g., Gaussian) for geometry optimization and single-point calculations, extracts per-molecule properties (energies, charges, etc.), and assembles them into analysis-ready tables.

- `libs/calc_grid.py`  
  Construction of 3D electronic descriptors on a common grid.  
  Reads cube files and charge information, defines and aligns 3D grids around the reactive carbonyl region, samples electronic density / electrostatic potential (and related fields), and writes grid-based descriptor matrices used as inputs for regression models.

- `libs/regression.py`  
  Regression and model-selection routines.  
  Wraps scikit-learn regressors (PLS, Ridge, Lasso, Elastic Net, OMP, etc.), performs LOOCV and test-set evaluation, computes metrics (R², RMSE, etc.), and exports fitted coefficients and model summaries for downstream analysis and figure generation.

- `libs/graph.py`  
  Plotting utilities for figures used in the paper and SI.  
  Includes functions for experimental vs predicted ΔΔG‡ scatter plots, 2D and 3D visualizations of electronic / electrostatic / LUMO contributions, stacked and bar-type summary graphs, and other publication-quality visualizations.

- `definition_grid.ipynb` and `libs/render_moleculer.py`  
  Visualization tools for 3D electronic fields.  
  The notebook is used to interactively define and inspect the 3D grids, while `render_moleculer.py` provides Py3Dmol helpers to render molecules, grids, and density/field information as interactive 3D scenes.


## Original Article
in preparation.


## License
3D_electronic_analysis_for_site-selectivity is available under [MIT License](https://github.com/poclab-web/3D_electronic_analysis_for_site-selectivity/LICENSE.txt).