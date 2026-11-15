# Data sources and counting units

The external-input manifest records provider URLs, known original file hashes,
access route and intended use. List it using `python scripts/fetch_external.py`.
To retrieve one known direct file, supply its ID explicitly:

```bash
python scripts/fetch_external.py --id RDAC2301 --destination external_data/sih
```

This checks the known SHA-256 and refuses to overwrite existing files. Some
provider/catalog URLs need manual selection or registration. No restricted
application-only dataset is required for the quick reproduction route.
Downloads remain subject to the provider's current terms.

AIH is an administrative admission authorization record, not a unique patient.
Released model rows are municipality-level counts/features or out-of-fold
municipality predictions. No N_AIH identifier or individual admission/date row
is included. Public raw SIH data should be downloaded from DATASUS, not committed.
Facility selection design files retain opaque candidate row indices and ISO3,
not third-party facility names, coordinates or an OSM source database.

The report-linked budget table contains heterogeneous source-defined surgery
measures and unmatched survey-to-OSM denominators. Uganda is reconstructed from
rounded hospital strata, Ghana's value is a hospital subtype, and Mali's scope
is ambiguous. The source inventory preserves these distinctions. q is a
facility-weighted local-population class, not SMOD urbanity or a calibrated
clinical capability composition. Budget and composition values are scenarios.

Main/supplementary table CSVs preserve their printed units; full-precision
analysis CSVs use fractions for many coverage fields. Do not subtract a
fraction from a percentage. `*_pp` is a percentage-point difference;
`*_percent`/`(%)` is a percentage. `all_30` and `coverage30` are fractions.
Repartition percentiles describe validation-design variation; bootstrap
intervals condition on fitted predictions and are not uncertainty in raw maps.
Blank/NaN fields are missing or inapplicable, not zero. Read each table's
corresponding manuscript legend for the estimand and denominator.
