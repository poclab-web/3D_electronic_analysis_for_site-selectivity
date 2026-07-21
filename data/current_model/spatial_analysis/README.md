# Current-model spatial contribution analysis

The current 321-feature model was refit without changing its inputs or alpha.
Grid coordinates are reported in model grid units and Angstrom; one grid unit
is 2 Bohr = 1.058354 Angstrom. The y coordinate is the model's
folded coordinate. Spatial plots place x vertically.

## Full-training spatial grid counts

| Block | Nonzero grids | Stable grids (outer frequency >= 0.5) | Grid L1 fraction | Centered total-block SD |
|---|---:|---:|---:|---:|
| electronic | 24 | 24 | 0.523 | 1.034 kcal/mol |
| electrostatic | 24 | 25 | 0.454 | 0.948 kcal/mol |
| orbital | 3 | 3 | 0.023 | 0.081 kcal/mol |

## Interpretation rules

- A positive coefficient raises predicted DeltaDeltaG when that scaled grid value increases; a negative coefficient lowers it.
- Electrostatic grid values are signed, so coefficient sign alone is not a potential-sign assignment.
- Marker size in the realized-effect map uses the RMS of beta_j times the centered training feature.
- Selection frequency is more reliable than one full-fit coefficient for correlated neighboring grids.
- Block contribution spreads are correlated and must not be read as additive percentages of explained variance.

Stable spatial grids with outer selection frequency >= 0.5: 52.
The compressed training effect matrix permits later substrate-specific spatial maps without refitting.
