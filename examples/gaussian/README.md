# Gaussian minimal example

This directory contains the only Gaussian calculation inputs intended for Git.
They are compact, inspectable examples of the geometry/frequency and
single-point calculations used when new electronic descriptors must be
generated.  The statistical results in `data/current_model/` are reproduced
from frozen, checksummed descriptors and therefore do **not** require Gaussian.

## Calculation

The reference calculation used Gaussian 16 Revision A.03 and the following
two-stage protocol:

- `benzophenone_opt_freq.gjf`: B3LYP-D3(BJ)/def2-SVP optimization and frequency
- `benzophenone_sp.gjf`: `wB97XD/def2TZVP` single point with SMD methanol and
  no symmetry constraints

The supplied single-point input already contains the aligned reference
optimized geometry, so the two files can be inspected or run independently.
For a new molecule, use the converged opt/freq geometry for the single point.

For example:

```bash
g16 benzophenone_opt_freq.gjf
g16 benzophenone_sp.gjf
formchk benzophenone_sp.chk benzophenone_sp.fchk
```

The reference energies, lowest harmonic frequency, and comparison tolerances
are recorded in `expected_results.json`.  Small differences can arise from
Gaussian revision, platform, integral implementation, or convergence details;
normal termination, absence of imaginary frequencies, and a matching molecular
specification must also be checked.

Do not add generated `.log`, `.chk`, `.fchk`, or `.cube` files to Git.  The
complete historical example (including volumetric cubes) is retained in the
external-SSD archive described in `docs/DATA_POLICY.md`.  Production Gaussian
jobs likewise belong outside this repository; set `MOLECULES_ROOT` when using
the descriptor-generation scripts.
