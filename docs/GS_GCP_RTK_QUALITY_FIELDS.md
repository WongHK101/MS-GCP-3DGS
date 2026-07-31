# GS-GCP RTK Measurement-Quality Fields

The frozen v1.3.0 point table contains coordinate provenance, observation
counts, coordinate ranges, PDOP, and satellite counts, but it does not expose
the receiver-reported per-epoch HRMS/VRMS fields for all points.

`build_gcp_point_quality_table.py` creates a non-release enrichment that:

- preserves all frozen point-table rows and fields;
- derives per-point statistics from `20260505改正.csv`;
- binds the final coordinates to `20260505G改正.dat`;
- reports fixed-solution rate, duration, HRMS/VRMS, coordinate repeatability,
  PDOP, satellite counts, and source/record hashes;
- leaves missing quality fields blank instead of borrowing another point's
  measurements.

## Accuracy terminology

Receiver HRMS/VRMS values are solution precision estimates reported by the RTK
receiver. Coordinate standard deviations and ranges quantify repeatability
within the corrected 27-epoch sequence. Neither is an independent measurement
of absolute coordinate accuracy against a higher-order reference.

## Known base-point gap

The frozen point table contains four known/base points (`K002`, `NC08`, `NC94`,
and `NC96`) that have no directly attributable epoch-quality records in the
corrected epoch inventory. `NC94` is the only one used by the formal split.
Measurements from similarly named nearby check points must not be substituted.

## Version boundary

The enrichment is an audit sidecar. The externally accepted v1.3.0 release
directory and root digest remain immutable. Adding these fields to the formal
payload requires a new non-overwriting metadata release and review.
