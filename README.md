# Geographic surgical coverage with unknown facility capability

Code and original aggregate results for **Geographic surgical coverage when facility-level capability is unobserved: global bounds and analyses of cholecystectomy and other procedures in Brazil and Mexico**.

This repository contains the final analytical results and the final flexible-model outputs.

## Quick reproduction

Use Python 3.12 in a clean environment. These commands need no raw admissions,
author drive letters, private servers or provider credentials:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r environment/requirements-validated.lock.txt
python scripts/verify_release.py
python scripts/reproduce_tables.py
python scripts/reproduce_figures.py
python -m unittest discover -s tests -v
```

Outputs appear in `artifacts/` (ignored by Git). Figure reproduction draws all
six main and four supplementary figures from CSV sources; it does not copy the
reference PDFs. The table command exports all 28 source CSVs as 27 numbered display tables and independently
checks main-table 2–4 arithmetic against underlying analytical aggregates.

To refit the primary denominator models from released municipality-level inputs:

```bash
python scripts/reproduce_models.py --service all --repartitions 10 --threshold 1
```

This fits nine models per procedure using the original ten paired five-fold
partitions, then compares AUC, Brier and log loss with the final stored results.
Use `--repartitions 1 --service chole` for a smaller check. Other published
inclusion thresholds are selected with `--threshold 5`, `10` or `20`.
These are models of observed AIH records, not unique patients or unmet need.

## Contents

| Directory | Contents |
|---|---|
| `scripts/` | Portable figure/table/model reproduction, validation, external-data retrieval and path adapters |
| `code/analysis/` | Scientific preprocessing, spatial C/Python engines, budget-exact selections, certification, procedure constructions and prediction code |
| `data/tables/` | Main tables 1–4 and supplementary tables S1–S23 |
| `data/figure_sources/` | All final figure inputs, including the three Figure 2 tables and S4 temporal inputs |
| `data/model_inputs/` | Municipality-level features and procedure counts, with no admission identifiers |
| `data/predictions/` | Municipality-level out-of-fold predictions, split into files below GitHub browser limits |
| `data/results/` | Full-precision analytical summaries, quotas, diagnostics and timing audits |
| `data/exact_cases/` | Derived catchment matrices and exact-case design metadata |
| `reference/figures/` | Ten frozen final PDF figures for comparison |
| `metadata/` | Provenance, external-input hashes, schemas, licensing map and release manifest |
| `docs/` | Analysis routes, source definitions, limitations and upload instructions |

## Reproduction levels

1. **Figures/tables:** fully self-contained from the supplied aggregate results.
2. **Municipal denominator model fitting:** self-contained from supplied city-level analytical inputs.
3. **Raw-to-spatial reconstruction:** requires external geographic and registry inputs,
   their original terms, and substantial disk space; see `docs/FULL_ANALYSIS.md`.
   The complete raw/raster pipeline was not rerun for this release. Several
   inherited facility/registry release identifiers remain incomplete; do not
   interpret the quick route as independent regeneration of those raw sources.

Global 5.58 pp and country-composition 4.41 pp differences are constructed lower
bounds, not optimized widths, confidence intervals or estimates of true global
coverage error. The q interval is illustrative, not clinically calibrated.

## Licences and citation

Author code and software documentation: **MIT** (`LICENSE`).
Authors' original aggregate results and figures: **CC BY 4.0** (`LICENSE-DATA.md`).
Third-party source data are not relicensed or redistributed; see
`THIRD_PARTY_NOTICES.md` and `metadata/external_data_manifest.csv`.
Cite the paper and version v1.0.0 of this repository using `CITATION.cff`: https://github.com/FENG1567/global-surgical-coverage/releases/tag/v1.0.0.

Contact: Xianzhi Meng, mengxianzhi@hrbmu.edu.cn.

## Publication figures and table display

The current plotting entry point is `scripts/reproduce_figures.py`, using the layout modules in `code/analysis/plotting/layout/`. The earlier readable-plot scripts remain as historical source versions. `make figures` exports editable PDF/SVG, 600 dpi RGB LZW TIFF and 300 dpi PNG. Main Table 1 summarizes all 27 certified countries; full country endpoints appear in current manuscript Table S4 (legacy analytical source `data/tables/Table_S2.csv`). `make tables` exports the analytical CSVs and the publication display tables, including the paired AUC presentation in Main Table 3. Display strings are in `metadata/table_display_spec.json`.

There are 27 numbered publication display tables. `data/tables/` also retains `Main_Table_1_full_fields.csv` as an auxiliary full-field endpoint source, so the source directory contains 28 CSVs. This auxiliary source is exported without creating a second numbered display table.


## Figure presentation revision 15 November 2026

The ten reference PDFs and the reproduction entry point use uppercase Arial Bold 18-pt panel labels and bold figure text at a native width of 183 mm. On Windows, installed Arial fonts are detected; elsewhere use `python scripts/reproduce_figures.py --font-dir /path/to/installed/Arial`. Font binaries are not distributed. Supplementary order follows the current manuscript: S1 information revelation, S2 strict validation and Mexico, S3 Brazil national composition, S4 state constraints and validation. Scientific source values are unchanged.


Figure reader review: all ten figures use explicit difference units and panel-specific legends. Plotting and reference PDFs match the current figure revision. The numerical source data are unchanged.


## Manuscript table numbering

`python scripts/reproduce_tables.py` exports CSV and Markdown tables under the current manuscript numbers, S1–S23. Files in `data/tables/` preserve their original analytical names for compatibility with the analysis scripts. `metadata/publication_table_crosswalk.csv` maps every current table number to its unchanged analytical CSV and checksum.

## Code availability

Author-developed analysis and reproduction code is publicly available under the MIT License at https://github.com/FENG1567/global-surgical-coverage, version v1.0.0. The repository includes aggregate inputs, figure and table reproduction scripts, municipality-level model-refitting code, environment specifications and source-data provenance. Authors' aggregate results and figures are licensed under CC BY 4.0. Reconstructing the full spatial analyses requires the third-party geographic and registry inputs listed in the external-data manifest, obtained under provider terms.

The automated GitHub workflow validates the release, reproduces tables and runs regression tests. Exact figure rendering additionally requires legally installed Arial Regular and Bold fonts; it is run through the documented local figure command rather than an unlicensed font download in CI.
