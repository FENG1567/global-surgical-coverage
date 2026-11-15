# Full-analysis routes and their boundaries

## Code staging

Archive code preserves the analytical implementation; only author-specific
path prefixes were replaced with documented tokens. Original source hashes
and origin roles are in `metadata/source_provenance.json`. Materialize a
separate working project, leaving the release immutable:

```bash
python scripts/materialize_analysis.py --config config/paths.example.json
```

This compiles the C engines and places code under `workspace/project/`.
Copy the example configuration to ignored `config/paths.local.json` to choose
another workspace or override `__PROJECT_ROOT__`, `__BRAZIL_SOURCE_ROOT__`,
`__SIH_ROOT__`, `__SCRATCH_ROOT__`, `__AUTHOR_OUTPUTS__`, `__SIH_PARQUET_ROOT__`, or `__SIGTAP_ROOT__`.
Remaining legacy path references are enumerated in
`metadata/legacy_absolute_path_references.csv`; some historical contextual
branches need additional upstream source files. They are not quick-route dependencies.

## Scientific stage order

| Stage | Archived entrypoints | Inputs / outputs |
|---|---|---|
| Common grid | `src/01_build_grid.py`, `02_build_sources.py`, `03_osm_hospitals.py` | External friction/GHSL/SMOD, facilities, boundaries; working rasters and facility sources |
| Global profiles | `src/04_validate.py` through the relevant aggregation/correction scripts | Working travel rasters and facility-definition profiles; final threshold source table |
| Budget-exact witnesses | `src/54_cluster_budget.py`, `59_export_selection.py` | Country proportions and OSM candidates; exact country budgets, four central assignments |
| Budget/composition sensitivity | `revision_20261113/global/global_budget_sensitivity.py` | Lower/central/upper proportions; paired selections and exact country composition |
| Country endpoints | `upgrade_20261112/code/certify_full_population_v2.py` | Full country masks, catchment patterns, MILP endpoints; final certificates |
| Information revelation | `upgrade_20261112/code/extract_exact_cases.py`, `exact_information_value.py` | Derived three-country NPZ inputs, nested disclosure simulations |
| Brazil ingestion | `upgrade_20261112/code/brazil_extract.py`, `local_sigtap_contract.py` | SIH DBCs and monthly SIGTAP; strict procedure hospital/municipality aggregates |
| Procedure geography | `brazil_spatial_experiment.py`, `bra_state_constraints.py`, `mexico_spatial_experiment.py`, `mex_state_constraints.py`, boundary variants | Hospital points, activity aggregates, state/municipality polygons and rasters |
| Structural qualification and masking | `revision_20261113/clinical/code/clinical_reanalysis.py`, `audit_clinical_selections.py` | CNES structural declarations; exact state/composition selections and paired state/national errors |
| Denominator prediction | `.../prediction/code/reanalyse_prediction.py`, `supplement_flexible_threshold1.py`, `binary_temporal_strong.py` | Municipal analytical inputs; final n>=1 models, strong-baseline time holdout |
| Record timing | `revision_20261113/raw_time/code/audit_raw_time.py` | 1,620 SIH DBCs; only aggregate completeness, date/identifier-overlap counts and reconciliations |

Paths in the last three rows are relative to the materialized project.
Prediction code is under `upgrade_20261112/derived/bmc_revision_20261113/prediction/code/`.
The supplemental script hash must match the `flexible_n1_fixed` manifest,
not the stale original completion-list script. The failed old `flexible_n1`
results and old non-budget-exact clustering branch are excluded.

## External inputs and limits

- The global grid is 17,400 × 43,200 at **1/120 degree**, using the numerical
  `dx`/`dy` in `metadata/grid_metadata.json`. Its old descriptive `grid` string
  says 0.01 degree and is a legacy comment; do not use it as the resolution.
- The working global raster tree in the author environment is roughly 202 GB;
  processed sensitivity rasters add about 25 GB. Do not commit them to Git.
  No fixed runtime or memory guarantee is inferred from those file sizes.
- `metadata/external_data_manifest.csv` includes the 1,620 SIH source-file
  hashes, inherited geographic-source inventory and 12 assessment-report URLs.
  A hash mismatch must stop snapshot reproduction rather than silently accept
  a changed provider release. Live OSM queries cannot recover an unspecified
  historical snapshot merely by using the same query.
- CNES2025, historical Mexican CLUES/SAEH, broad facility layers and some
  inherited extract release identifiers have incomplete original provenance.
  The analytical code consumes hospital-level source inputs; the original
  Mexican admission-to-facility extraction was not rerun in the study upgrade.
  Those upstream file identities must be resolved for exact independent
  raw-to-map reconstruction. This release does not conceal or fill them with
  invented accession numbers.
- Geographic/DBC ingestion needs optional packages in
  `environment/requirements-analysis.txt`. These optional stages are not the
  independently installed/tested quick environment. Provider licences remain
  separate from author-result CC BY 4.0.

For a tested route requiring none of those raw files, use `reproduce_models.py`
and the released municipality-level inputs. Raw pipeline code is provided for
inspection and controlled reconstruction after source acquisition, not claimed
as a one-command, raw-data-complete workflow.

Legacy contextual SIH parquet inputs are now parameterized separately under
`__SIH_PARQUET_ROOT__`. They are intermediate per-admission files, not released
municipal aggregates. They require provider DBC ingestion with the original
procedure definition; their production is not exercised by the tested current
municipal route. Monthly SIGTAP dictionaries are under `__SIGTAP_ROOT__`.
For Natural Earth, retrieve and extract the complete ZIP with SHP/DBF/SHX/PRJ;
the inherited manifest hash refers to the extracted SHP, not the ZIP bytes.
