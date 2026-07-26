# Immutable inputs for the accepted model

This directory contains the portable inputs required to reproduce the reported
model without rerunning Gaussian calculations. The seven files declared in
`input_manifest.csv` are the authoritative input set.

## Contents

- `model_arrays.npz`: raw electronic, electrostatic, and orbital arrays plus
  the 4,927 integer grid coordinates for each block.
- `model_metadata.csv`: identities, labels, SMILES, temperatures, experimental
  responses, and dataset roles for 161 molecules.
- `model_provenance.json`: descriptor version, block order, array dimensions,
  conformer protocol, quantum-chemical protocol, and source provenance.
- `train_rows.csv`: the 83 frozen training identities, row indices, InChIKeys,
  and response values.
- `projected_orbital_fullgrid_2bohr.npz`: full-grid projected C=O pi-star
  descriptor cache.
- `projected_orbital_manifest.csv`: sanitized conformer weights and orbital
  provenance using package-relative identifiers only.
- `display_geometries.json`: display geometries for optional contribution-cube
  regeneration; these geometries are not model features.
- `input_manifest.csv`: relative path, byte size, and SHA-256 for every
  authoritative input file.

The superseded `model_input_bundle.pkl` was used only during migration to this
portable package. It is excluded from Git and is not read by the accepted
runner.

The 2026-07-26 reviewed workbook refresh removed six non-training rows and
retained all 83 training identities and responses. The exact removed
InChIKeys are recorded in `model_provenance.json` and the refresh audit.

## Verification

Run from the repository root:

```bash
python libs/current_model.py --verify-inputs-only
python scripts/verify_reproduction.py
```

The first command validates hashes, schemas, coordinate alignment, and the 83
training identities. The second additionally constructs the 161-by-321 feature
matrix, recalculates the saved nested LOOCV metrics, and validates the spatial
fold, feature, and holdout identities.

## Curation rules

- Do not edit an individual frozen file without regenerating and reviewing the
  complete package.
- Align molecular records by InChIKey rather than mutable entry labels or row
  positions.
- Keep manifest paths relative to this directory and exclude user-, host-, and
  mount-specific paths.
- Store only numerical or string arrays readable with `allow_pickle=False`.
- Update byte sizes, SHA-256 values, `expected_metrics.json`, and reproduction
  results together.

Quantum-chemical calculations for new molecules belong outside the repository.
Only reviewed descriptors are promoted into this directory. The Multiwfn
version used for the historical descriptors was not recorded, so the
checksummed inputs are the canonical source for the publication results. New
descriptor-generation runs must record their Multiwfn version.
