# Data-alignment audits

These records document transformations performed while assembling or
refreshing the accepted-model package. They are provenance artifacts rather
than model features.

- `descriptor_coordinate_alignment_audit.csv` records the reviewed
  electrostatic-coordinate zero padding used to align the three blocks.
- `excel_refresh_audit.json` summarizes the identity-based comparison between
  frozen metadata and the active experimental workbook.
- `excel_refresh_*.csv` contain row-level differences, additions, or removals.

Workbook alignment uses InChIKey and never relies on mutable entry labels or
spreadsheet row positions.
