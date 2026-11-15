#!/usr/bin/env python3
"""Audit the five staged SIH cholecystectomy extracts without exporting records.

Outputs contain only aggregate counts and source provenance. No AIH identifiers
or row-level hospital/patient records are written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


YEARS = [2017, 2018, 2019, 2023, 2024]
PRIMARY_CODES = {"0407030026", "0407030034"}
EVENT_KEY = ["CNES", "PROC_REA", "MUNIC_RES", "MUNIC_MOV", "DT_INTER", "DT_SAIDA"]
RAW_COLUMNS = [
    "N_AIH", "ANO_CMPT", "MES_CMPT", "CNES", "MUNIC_RES", "MUNIC_MOV",
    "PROC_REA", "DT_INTER", "DT_SAIDA", "src_file",
]


def normalized(series: pd.Series, width: int | None = None) -> pd.Series:
    out = series.astype("string").str.strip().str.replace(r"\.0+$", "", regex=True)
    out = out.mask(out.eq(""), pd.NA)
    return out.str.zfill(width) if width else out


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def date_year(series: pd.Series) -> pd.Series:
    raw = normalized(series)
    parsed = pd.to_datetime(raw, format="%Y%m%d", errors="coerce")
    return parsed.dt.year.astype("Int64")


def write_csv(frame: pd.DataFrame, output: Path, name: str) -> None:
    frame.to_csv(output / name, index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    input_dir = args.project_root / "upgrade_20261112" / "inputs"
    municipality_path = input_dir / "brazil_municipality_annual.csv"
    hospital_path = input_dir / "brazil_hospital_annual.csv"

    # Keep Arrow's reader and conversion single-threaded to share server CPU.
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)

    frames: list[pd.DataFrame] = []
    year_audit: list[dict[str, object]] = []
    input_sources: list[dict[str, object]] = []
    for source_year in YEARS:
        path = args.raw_root / f"sih_{source_year}_cholec.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        parquet = pq.ParquetFile(path)
        missing = sorted(set(RAW_COLUMNS) - set(parquet.schema_arrow.names))
        if missing:
            raise ValueError(f"{path.name} missing required fields: {missing}")
        raw = parquet.read(columns=RAW_COLUMNS, use_threads=False).to_pandas()
        raw_rows = len(raw)
        raw["source_year"] = source_year
        raw["ANO_CMPT"] = normalized(raw["ANO_CMPT"])
        raw["MES_CMPT"] = normalized(raw["MES_CMPT"])
        raw["CNES"] = normalized(raw["CNES"], 7)
        raw["MUNIC_RES"] = normalized(raw["MUNIC_RES"], 6)
        raw["MUNIC_MOV"] = normalized(raw["MUNIC_MOV"], 6)
        raw["PROC_REA"] = normalized(raw["PROC_REA"])
        raw["N_AIH"] = normalized(raw["N_AIH"])
        raw["DT_INTER"] = normalized(raw["DT_INTER"])
        raw["DT_SAIDA"] = normalized(raw["DT_SAIDA"])
        raw["src_file"] = normalized(raw["src_file"])

        code_counts = raw["PROC_REA"].value_counts(dropna=False)
        strict = raw.loc[raw["PROC_REA"].isin(PRIMARY_CODES)].copy()
        strict["processing_year"] = strict["source_year"]
        strict["reported_processing_year"] = pd.to_numeric(strict["ANO_CMPT"], errors="coerce").astype("Int64")
        strict["admission_year"] = date_year(strict["DT_INTER"])
        strict["discharge_year"] = date_year(strict["DT_SAIDA"])
        strict["local_record"] = strict["MUNIC_RES"].eq(strict["MUNIC_MOV"]).astype("int8")

        proc_counts = raw["ANO_CMPT"].value_counts(dropna=False)
        date_admit = strict["admission_year"].astype("string")
        date_discharge = strict["discharge_year"].astype("string")
        year_audit.append({
            "source_file_year": source_year,
            "parquet_rows_all_codes": raw_rows,
            "parquet_primary_code_rows": len(strict),
            "parquet_nonprimary_code_rows": raw_rows - len(strict),
            "primary_code_missing_N_AIH": int(strict["N_AIH"].isna().sum()),
            "primary_code_missing_admission_date": int(strict["admission_year"].isna().sum()),
            "primary_code_missing_discharge_date": int(strict["discharge_year"].isna().sum()),
            "ANO_CMPT_values": json.dumps({str(k): int(v) for k, v in proc_counts.items()}, ensure_ascii=False),
            "primary_rows_with_ANO_CMPT_not_source_year": int(strict["reported_processing_year"].ne(source_year).fillna(True).sum()),
            "primary_rows_admission_year_not_source_year": int(strict["admission_year"].ne(source_year).fillna(True).sum()),
            "primary_rows_discharge_year_not_source_year": int(strict["discharge_year"].ne(source_year).fillna(True).sum()),
            "primary_admission_year_distribution": json.dumps({str(k): int(v) for k, v in date_admit.value_counts(dropna=False).items()}, ensure_ascii=False),
            "primary_discharge_year_distribution": json.dumps({str(k): int(v) for k, v in date_discharge.value_counts(dropna=False).items()}, ensure_ascii=False),
            "primary_PROCREA_distribution": json.dumps({str(k): int(v) for k, v in code_counts.loc[code_counts.index.isin(PRIMARY_CODES)].items()}, ensure_ascii=False),
        })
        input_sources.append({
            "path": str(path),
            "bytes": path.stat().st_size,
            "parquet_rows": parquet.metadata.num_rows,
            "sha256": sha256_file(path),
        })
        frames.append(strict[[
            "N_AIH", "source_year", "reported_processing_year", "MES_CMPT", "CNES",
            "MUNIC_RES", "MUNIC_MOV", "PROC_REA", "DT_INTER", "DT_SAIDA",
            "admission_year", "discharge_year", "local_record",
        ]])
        print(f"RAW_YEAR_AUDITED {source_year} rows={raw_rows} primary={len(strict)}", flush=True)

    records = pd.concat(frames, ignore_index=True)
    write_csv(pd.DataFrame(year_audit), output, "raw_processing_year_audit.csv")

    # Count admission and discharge years separately from the processing-file year.
    time_rows: list[dict[str, object]] = []
    for outcome_year in YEARS:
        part = records.loc[records["source_year"].eq(outcome_year)]
        for field, label in [("admission_year", "admission"), ("discharge_year", "discharge")]:
            distribution = part[field].astype("string").value_counts(dropna=False)
            for date_year_value, count in distribution.items():
                time_rows.append({
                    "source_file_processing_year": outcome_year,
                    "date_type": label,
                    "event_date_year": str(date_year_value),
                    "primary_code_records": int(count),
                })
    write_csv(pd.DataFrame(time_rows), output, "raw_processing_vs_service_year_counts.csv")

    # Exact event signatures are used only to count overlaps; identifiers/signatures
    # are never exported. A repeated N_AIH alone is not classified as duplication.
    valid_ids = records.loc[records["N_AIH"].notna()].copy()
    id_summary = valid_ids.groupby("N_AIH", sort=False).agg(
        n_rows=("source_year", "size"),
        n_processing_years=("source_year", "nunique"),
        n_distinct_cnes=("CNES", "nunique"),
        n_distinct_procedures=("PROC_REA", "nunique"),
        n_distinct_residence=("MUNIC_RES", "nunique"),
        n_distinct_treatment=("MUNIC_MOV", "nunique"),
        n_distinct_admission_dates=("DT_INTER", "nunique"),
        n_distinct_discharge_dates=("DT_SAIDA", "nunique"),
    )
    multi_year_ids = id_summary.loc[id_summary["n_processing_years"].gt(1)]
    exact_signatures = valid_ids.groupby(["N_AIH", *EVENT_KEY], dropna=False, sort=False).agg(
        n_rows=("source_year", "size"),
        n_processing_years=("source_year", "nunique"),
    )
    cross_year_exact = exact_signatures.loc[exact_signatures["n_processing_years"].gt(1)]
    cross_year_exact_ids = set(cross_year_exact.index.get_level_values("N_AIH"))
    duplicate_summary = pd.DataFrame([
        {"measure": "unique_nonmissing_N_AIH", "value": len(id_summary), "interpretation": "unique identifiers across the five selected processing-year extracts"},
        {"measure": "N_AIH_seen_in_multiple_processing_years", "value": len(multi_year_ids), "interpretation": "identifier overlap; not by itself evidence of duplicated event"},
        {"measure": "rows_belonging_to_multi_year_N_AIH", "value": int(multi_year_ids.n_rows.sum()) if len(multi_year_ids) else 0, "interpretation": "all rows with an identifier found in more than one processing year"},
        {"measure": "multi_year_N_AIH_with_exact_event_signature_repeated", "value": len(cross_year_exact_ids), "interpretation": "same identifier and same CNES/procedure/residence/treatment/admission/discharge signature in multiple processing years"},
        {"measure": "exact_event_signatures_repeated_across_processing_years", "value": len(cross_year_exact), "interpretation": "signature groups; duplicate processing is possible but not adjudicated"},
        {"measure": "multi_year_N_AIH_without_any_exact_signature_match", "value": len(set(multi_year_ids.index) - cross_year_exact_ids), "interpretation": "same identifier with no identical event signature across selected years; may reflect reused/changed identifiers or distinct records"},
        {"measure": "within_year_exact_event_duplicate_rows", "value": int(valid_ids.duplicated(subset=["source_year", "N_AIH", *EVENT_KEY], keep="first").sum()), "interpretation": "additional rows with identical event keys within a processing year"},
    ])
    write_csv(duplicate_summary, output, "raw_crossyear_identifier_overlap_summary.csv")

    pair_rows: list[dict[str, object]] = []
    year_tables = {year: valid_ids.loc[valid_ids["source_year"].eq(year)] for year in YEARS}
    for left_pos, left_year in enumerate(YEARS):
        for right_year in YEARS[left_pos + 1:]:
            left = year_tables[left_year]
            right = year_tables[right_year]
            left_ids = set(left["N_AIH"].dropna())
            right_ids = set(right["N_AIH"].dropna())
            shared = left_ids.intersection(right_ids)
            left_unique = left.loc[left["N_AIH"].isin(shared), ["N_AIH", *EVENT_KEY]].drop_duplicates()
            right_unique = right.loc[right["N_AIH"].isin(shared), ["N_AIH", *EVENT_KEY]].drop_duplicates()
            same_event = left_unique.merge(right_unique, on=["N_AIH", *EVENT_KEY], how="inner") if shared else pd.DataFrame()
            exact_ids = int(same_event["N_AIH"].nunique()) if len(same_event) else 0
            pair_rows.append({
                "left_processing_year": left_year,
                "right_processing_year": right_year,
                "shared_N_AIH_ids": len(shared),
                "shared_ids_with_any_exact_event_signature": exact_ids,
                "shared_ids_without_exact_event_signature": len(shared) - exact_ids,
                "matching_exact_event_signature_pairs": len(same_event),
                "interpretation": "same N_AIH across years is an overlap flag; exact signatures are candidate duplicate processing, not confirmed duplication",
            })
    write_csv(pd.DataFrame(pair_rows), output, "raw_crossyear_identifier_overlap_pairs.csv")

    municipality = pd.read_csv(municipality_path, dtype={"MUNIC_RES": "string"})
    municipality["MUNIC_RES"] = normalized(municipality["MUNIC_RES"], 6)
    municipality["year"] = pd.to_numeric(municipality["year"], errors="coerce").astype("Int64")
    hospital = pd.read_csv(hospital_path, dtype={"CNES": "string"})
    hospital["CNES"] = normalized(hospital["CNES"], 7)
    hospital["year"] = pd.to_numeric(hospital["year"], errors="coerce").astype("Int64")

    reconciliations: list[dict[str, object]] = []
    for year in YEARS:
        part = records.loc[records["source_year"].eq(year)].copy()
        raw_h = part.groupby("CNES", dropna=False).size().rename("raw_n").reset_index()
        agg_h = hospital.loc[hospital["year"].eq(year) & hospital["service"].eq("chole"), ["CNES", "n"]].rename(columns={"n": "aggregate_n"})
        joined_h = raw_h.merge(agg_h, on="CNES", how="outer", validate="one_to_one").fillna({"raw_n": 0, "aggregate_n": 0})
        joined_h["difference"] = joined_h["raw_n"] - joined_h["aggregate_n"]

        raw_m = part.groupby("MUNIC_RES", dropna=False).agg(
            raw_n=("N_AIH", "size"), raw_local=("local_record", "sum")
        ).reset_index()
        agg_m = municipality.loc[municipality["year"].eq(year) & municipality["service"].eq("chole"), ["MUNIC_RES", "n", "local"]].rename(columns={"n": "aggregate_n", "local": "aggregate_local"})
        joined_m = raw_m.merge(agg_m, on="MUNIC_RES", how="outer", validate="one_to_one").fillna({"raw_n": 0, "raw_local": 0, "aggregate_n": 0, "aggregate_local": 0})
        joined_m["n_difference"] = joined_m["raw_n"] - joined_m["aggregate_n"]
        joined_m["local_difference"] = joined_m["raw_local"] - joined_m["aggregate_local"]
        reconciliations.append({
            "processing_year": year,
            "raw_primary_code_records": len(part),
            "aggregate_municipality_resident_records": int(agg_m["aggregate_n"].sum()),
            "raw_municipalities": int(raw_m["MUNIC_RES"].nunique()),
            "aggregate_municipalities": int(agg_m["MUNIC_RES"].nunique()),
            "municipality_n_mismatches": int(joined_m["n_difference"].ne(0).sum()),
            "municipality_local_mismatches": int(joined_m["local_difference"].ne(0).sum()),
            "resident_record_absolute_difference_sum": int(joined_m["n_difference"].abs().sum()),
            "local_record_absolute_difference_sum": int(joined_m["local_difference"].abs().sum()),
            "raw_hospitals": int(raw_h["CNES"].nunique()),
            "aggregate_hospitals": int(agg_h["CNES"].nunique()),
            "hospital_n_mismatches": int(joined_h["difference"].ne(0).sum()),
            "hospital_record_absolute_difference_sum": int(joined_h["difference"].abs().sum()),
            "raw_local_records": int(part["local_record"].sum()),
            "aggregate_local_records": int(agg_m["aggregate_local"].sum()),
        })
    write_csv(pd.DataFrame(reconciliations), output, "raw_aggregate_reconciliation.csv")

    summary = {
        "scope": "five selected processing-year cholecystectomy extracts only; raw records were read on server and not exported",
        "selected_processing_years": YEARS,
        "primary_SIGTAP_codes": sorted(PRIMARY_CODES),
        "event_signature_fields": EVENT_KEY,
        "source_files": input_sources,
        "aggregate_inputs": [str(municipality_path), str(hospital_path)],
        "aggregate_reconciliation_status": "see raw_aggregate_reconciliation.csv; exact row-level equality is evaluated by residence municipality and CNES",
        "identifier_interpretation": "N_AIH overlap is not alone treated as duplicate; exact field signatures are candidates for duplicate processing and require source-level adjudication",
        "years_outside_selected_windows": "not scanned in this focused pass; claims processed in other years may be absent from the target-year extracts",
        "software": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "pyarrow": pa.__version__},
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("RAW_AUDIT_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
