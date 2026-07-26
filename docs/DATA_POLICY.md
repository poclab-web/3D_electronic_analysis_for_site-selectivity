# Data and artifact policy

This repository is a compact, auditable package for reproducing the accepted
model reported in the accompanying manuscript. It is not an archive of every
exploratory analysis or quantum-chemistry working directory.

## Versioned in Git

The following materials belong in the public repository:

- Python source, environment specification, tests, and reproduction guidance.
- The experimental source workbook,
  `data/Details_of_experimental_results.xlsx`.
- Portable immutable inputs under `data/current_model/inputs/`, limited to
  NPZ, CSV, JSON, and Markdown files. Pickle is not an accepted runtime format.
- Compact accepted-model results under `data/current_model/results/`, including
  predictions, coefficients, validation metrics, diketone evaluation, and
  publication tables.
- Compact provenance records under `data/current_model/audits/` and reviewed
  comparator summaries under `data/current_model/comparators/`.
- Spatial-analysis tables and compressed matrices under
  `data/current_model/spatial_analysis/`.
- Final accepted-model PNGs under `data/validation/current_model/`.
- Minimal Gaussian input examples and human/machine-readable metadata under
  `examples/gaussian/`.
- Superseded source code under `libs/legacy/` when it materially documents the
  analysis history. Legacy outputs are not versioned.
- The compact withdrawn x/y input and result snapshot under
  `data/archive/xy_diketones/`, with a checksum manifest and an explicit
  statement that it is excluded from the accepted runtime.

The frozen-input manifest records relative paths, byte sizes, and SHA-256
hashes. Versioned inputs, results, manifests, and documentation must not
contain user names, host names, mount points, or other machine-specific paths.

## Regenerated and excluded from Git

The following artifacts are reproducible working products and remain ignored:

- temporary files under `data/current_model/work/`;
- legacy pickle bundles and exploratory tabular exports;
- conformer caches, Gaussian logs, checkpoints, formatted checkpoints, cube
  files, scratch directories, and run logs;
- model-contribution cubes and their per-cube launchers;
- unreviewed external-series caches under
  `data/validation/external_diketones/`;
- exploratory output under `data/legacy/`;
- Python environments and caches, test caches, editor metadata, and Office
  temporary files;
- internal Word-editing automation under `scripts/publication/`.

A canonical reproduction must use only versioned inputs. When accepted outputs
are updated, the code, input differences, random seeds, software versions, and
numerical verification must be updated in the same reviewed change.

## Maintained outside the repository

The following materials require controlled external storage:

- production Gaussian inputs, outputs, and scratch data other than the minimal
  examples in `examples/gaussian/`;
- superseded large tables, pickle files, historical models, and exploratory
  runs;
- editable manuscript, Supporting Information, response-to-reviewers, and
  figure-source files;
- internal reports and notes;
- third-party article PDFs, for which the repository should retain only formal
  citations, DOIs, or BibTeX as appropriate.

External archives should include a manifest containing the original relative
path, purpose, file count, total bytes, SHA-256, and archive date. File counts,
sizes, and checksums must be verified before local material is reorganized.
Important raw data and manuscript files must not rely on a single external
drive as their only backup.

## Promoting new data into the accepted package

1. Run Gaussian calculations and exploratory analyses outside the repository.
2. Convert only accepted descriptors to portable NPZ, CSV, or JSON files and
   remove machine-specific paths.
3. Update the input manifest and model specification, including checksums,
   provenance, and software versions.
4. Recompute the accepted results from the frozen package.
5. Run the independent verifier and regression tests.
6. Review numerical and visual differences before committing the inputs,
   compact results, and final PNGs together.

An ignore rule does not remove an already tracked file from Git. A tracked
artifact should be removed from the index only after its external archive has
been independently verified.

## Git-history scope

Earlier `main` history contains Gaussian cubes/checkpoints, historical
notebooks, and environment files that are no longer present in the current
tree. Removing files from the working tree or adding ignore rules does not
reduce existing history. As measured on 2026-07-21, a bundle containing only
the `main` branch compressed to approximately 166 MiB.

The local object database may also contain tool snapshots and temporary refs,
so its size is not a measure of the public branch or a fresh clone. History
rewriting with `git filter-repo` or an equivalent tool is a separate operation
that changes commit identifiers and normally requires a force push and fresh
clones. It must be undertaken only from a verified backup and with agreement
from all collaborators.
