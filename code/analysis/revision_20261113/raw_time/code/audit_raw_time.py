"""Aggregate-only raw SIH date, identifier, overlap, and reconciliation audit."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dbfread import DBF
from pyreaddbc import dbc2dbf


HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "results"
SOURCE_ROOT = Path(r"__SIH_ROOT__")
SAVED_AGGREGATES = Path(
    r"__PROJECT_ROOT__\upgrade_20261112\derived\brazil"
)
YEARS = (2023, 2024, 2017, 2018, 2019)
EXPECTED_STATES = {
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA",
    "MG", "MS", "MT", "PA", "PB", "PE", "PI", "PR", "RJ", "RN",
    "RO", "RR", "RS", "SC", "SE", "SP", "TO",
}
SERVICE_CODES = {
    "chole": ("0407030026", "0407030034"),
    "cesarean": ("0411010026", "0411010034", "0411010042"),
    "appendectomy": ("0407020039", "0407020047"),
}
CODE_TO_SERVICE = {code: service for service, codes in SERVICE_CODES.items() for code in codes}
CODE_BYTES = np.array([code.encode("ascii") for code in CODE_TO_SERVICE])
SELECT_FIELDS = (
    "N_AIH", "ANO_CMPT", "MES_CMPT", "DT_INTER", "DT_SAIDA", "CNES",
    "PROC_REA", "MUNIC_RES", "MUNIC_MOV", "IDENT", "SEQ_AIH5",
)
FILE_RE = re.compile(r"RD([A-Z]{2})(\d{2})(\d{2})\.dbc$", re.IGNORECASE)
VALID_ID = re.compile(r"\d{13}$")
VALID_HOSPITAL = re.compile(r"\d{7}$")
VALID_MUNICIPALITY = re.compile(r"\d{6}$")
RUN_UTC = datetime.now(timezone.utc).isoformat(timespec="seconds")


def emit(event: str, **fields: object) -> None:
    print(json.dumps({"time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                      "event": event, **fields}, ensure_ascii=False), flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_files() -> list[Path]:
    files: list[Path] = []
    for year in YEARS:
        folder = SOURCE_ROOT / str(year)
        files.extend(path for path in folder.rglob("*")
                     if path.is_file() and path.suffix.lower() == ".dbc")
    year_order = {year: index for index, year in enumerate(YEARS)}
    return sorted(files, key=lambda p: (year_order[int(p.parents[1].name)], p.name.upper()))


def count_rows(counter: Counter, year: int, service: str, groups: pd.Series) -> None:
    for key, value in groups.items():
        group = key if isinstance(key, tuple) else (key,)
        counter[(year, service, *group)] += int(value)


def weighted_quantile(counts: Counter, probability: float) -> int | None:
    if not counts:
        return None
    total = sum(counts.values())
    target = max(1, int(np.ceil(total * probability)))
    cumulative = 0
    for value in sorted(counts):
        cumulative += counts[value]
        if cumulative >= target:
            return int(value)
    return int(max(counts))


def row_signature_counts(df: pd.DataFrame, year: int, scope: str,
                         columns: list[str], target: Counter) -> None:
    grouped = df.groupby(columns, dropna=False, sort=False).size()
    for key, value in grouped.items():
        vals = key if isinstance(key, tuple) else (key,)
        target[(year, scope, *vals)] += int(value)


def extract_file(source: Path) -> tuple[dict, pd.DataFrame]:
    match = FILE_RE.fullmatch(source.name)
    if not match:
        raise ValueError("unexpected_filename")
    uf, yy, mm = match.groups()
    processing_year = 2000 + int(yy)
    month = int(mm)
    folder_year = int(source.parents[1].name)
    if processing_year != folder_year or month not in range(1, 13) or uf.upper() not in EXPECTED_STATES:
        raise ValueError("filename_period_or_state_mismatch")

    size_bytes = source.stat().st_size
    digest = sha256_file(source)
    scratch = Path(tempfile.mkdtemp(prefix="sih_raw_time_audit_"))
    decoded = scratch / "decoded.dbf"
    data = None
    dbf = None
    try:
        dbc2dbf(str(source), str(decoded))
        dbf = DBF(str(decoded), encoding="latin1", load=False)
        field_names = {field.name for field in dbf.fields}
        missing = sorted(set(SELECT_FIELDS) - field_names)
        if missing:
            raise ValueError("missing_required_fields:" + ",".join(missing))

        dtype_names = ["deleted"]
        dtype_formats = ["S1"]
        dtype_offsets = [0]
        position = 1
        for field in dbf.fields:
            dtype_names.append(field.name)
            dtype_formats.append(f"S{field.length}")
            dtype_offsets.append(position)
            position += field.length
        dtype = np.dtype({"names": dtype_names, "formats": dtype_formats,
                          "offsets": dtype_offsets, "itemsize": dbf.header.recordlen})
        data = np.memmap(decoded, dtype=dtype, mode="r", offset=dbf.header.headerlen,
                         shape=(dbf.header.numrecords,))
        procedures = np.char.strip(data["PROC_REA"])
        selected = (np.isin(procedures, CODE_BYTES) & (data["deleted"] != b"*"))
        columns = {name: np.char.decode(np.char.strip(data[name][selected]), "latin1")
                   for name in SELECT_FIELDS}
        df = pd.DataFrame(columns)
        df["service"] = df["PROC_REA"].map(CODE_TO_SERVICE)
        df["processing_year"] = processing_year
        df["processing_month"] = month

        # Values are only used in memory for grouping; neither identifiers nor rows are emitted.
        for name in ("DT_INTER", "DT_SAIDA"):
            parsed = pd.to_datetime(df[name], format="%Y%m%d", errors="coerce")
            df[f"_{name}_parsed"] = parsed
            df[f"_{name}_year"] = parsed.dt.year
            df[f"_{name}_period"] = parsed.dt.to_period("M")

        audit = {
            "source_file": str(source),
            "processing_year": processing_year,
            "month": month,
            "uf": uf.upper(),
            "size_bytes": size_bytes,
            "sha256": digest,
            "source_rows": int(len(data)),
            "selected_rows": int(len(df)),
            "status": "success",
        }
        return audit, df
    finally:
        if data is not None:
            try:
                data._mmap.close()
            except Exception:
                pass
        del data, dbf
        gc.collect()
        resolved = scratch.resolve()
        temp_root = Path(tempfile.gettempdir()).resolve()
        if resolved.parent == temp_root and resolved.name.startswith("sih_raw_time_audit_"):
            shutil.rmtree(resolved)


def save_manifest(rows: list[dict], final: bool = False) -> None:
    path = RESULTS / ("input_manifest.csv" if final else "input_manifest.partial.csv")
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def save_same_year_inputs(counter: Counter, years: tuple[int, ...]) -> None:
    rows = [{"year": year, "service": service, "MUNIC_RES": municipality,
             "n": int(values[0]), "local": int(values[1])}
            for (year, service, municipality), values in counter.items()
            if year in years]
    frame = pd.DataFrame(rows, columns=["year", "service", "MUNIC_RES", "n", "local"])
    if not frame.empty:
        frame = frame.sort_values(["year", "service", "MUNIC_RES"], kind="stable")
    frame.to_csv(RESULTS / "same_year_admission_municipality_outcomes.csv",
                 index=False, encoding="utf-8-sig")


def reconcile_aggregate(label: str, actual_rows: list[tuple], saved_path: Path,
                        key_columns: list[str], value_columns: list[str]) -> dict:
    if not saved_path.exists():
        return {"aggregate": label, "saved_path": str(saved_path), "status": "missing_saved_aggregate"}
    actual = pd.DataFrame(actual_rows, columns=key_columns + value_columns)
    if actual.empty:
        actual = pd.DataFrame(columns=key_columns + value_columns)
    saved = pd.read_csv(saved_path, dtype={column: "string" for column in key_columns})
    for column in key_columns:
        actual[column] = actual[column].astype("string")
        saved[column] = saved[column].astype("string")
    for column in value_columns:
        actual[column] = pd.to_numeric(actual[column], errors="coerce").fillna(0).astype("int64")
        saved[column] = pd.to_numeric(saved[column], errors="coerce").fillna(0).astype("int64")
    joined = actual.merge(saved, on=key_columns, how="outer", suffixes=("_actual", "_saved"), indicator=True)
    for column in value_columns:
        joined[f"{column}_actual"] = joined[f"{column}_actual"].fillna(0).astype("int64")
        joined[f"{column}_saved"] = joined[f"{column}_saved"].fillna(0).astype("int64")
    value_mismatch = pd.Series(False, index=joined.index)
    abs_diff: dict[str, int] = {}
    for column in value_columns:
        diff = joined[f"{column}_actual"] - joined[f"{column}_saved"]
        value_mismatch |= diff.ne(0)
        abs_diff[column] = int(diff.abs().sum())
    return {
        "aggregate": label,
        "saved_path": str(saved_path),
        "status": "compared",
        "actual_groups": int(len(actual)),
        "saved_groups": int(len(saved)),
        "matched_key_groups": int((joined["_merge"] == "both").sum()),
        "actual_only_groups": int((joined["_merge"] == "left_only").sum()),
        "saved_only_groups": int((joined["_merge"] == "right_only").sum()),
        "groups_with_value_mismatch": int(value_mismatch.sum()),
        "absolute_difference_sum_by_value": abs_diff,
    }


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    files = source_files()
    emit("audit_started", run_started_utc=RUN_UTC, input_files=len(files), years=list(YEARS),
         python=sys.executable, reader_workers=1)

    file_rows: list[dict] = []
    procedure_counts: Counter = Counter()
    ano_counts: Counter = Counter()
    date_year_counts: Counter = Counter()
    date_status_counts: Counter = Counter()
    date_lag_counts: Counter = Counter()
    selected_code_counts: Counter = Counter()
    hospital_counts: Counter = Counter()
    municipality_counts: Counter = Counter()
    same_year_municipality_counts: Counter = Counter()
    id_counts: Counter = Counter()
    exact_record_counts: Counter = Counter()
    event_counts: Counter = Counter()
    identifier_event_counts: Counter = Counter()
    identifiers: dict[tuple[int, str], set[str]] = {}

    current_year: int | None = None
    processed_files = 0
    for source in files:
        expected_year = int(source.parents[1].name)
        if current_year is not None and expected_year != current_year:
            save_manifest(file_rows)
            if current_year == 2024:
                save_same_year_inputs(same_year_municipality_counts, (2023, 2024))
                emit("early_outcomes_ready", years=[2023, 2024],
                     path=str(RESULTS / "same_year_admission_municipality_outcomes.csv"))
        current_year = expected_year
        try:
            audit, df = extract_file(source)
            year = int(audit["processing_year"])
            processed_files += 1
            file_rows.append(audit)

            for service, subset in df.groupby("service", sort=False):
                service = str(service)
                procedure_counts[(year, service)] += int(len(subset))
                for code, count in subset["PROC_REA"].value_counts().items():
                    selected_code_counts[(year, service, str(code))] += int(count)

                for ano, count in subset["ANO_CMPT"].value_counts(dropna=False).items():
                    ano_counts[(year, service, str(ano).strip() or "missing")] += int(count)

                for date_field in ("DT_INTER", "DT_SAIDA"):
                    parsed = subset[f"_{date_field}_parsed"]
                    date_year = subset[f"_{date_field}_year"]
                    valid_count = int(parsed.notna().sum())
                    date_status_counts[(year, service, date_field, "valid")] += valid_count
                    date_status_counts[(year, service, date_field, "missing_or_invalid")] += int(len(subset) - valid_count)
                    year_labels = date_year.astype("Int64").astype("string").fillna("missing_or_invalid")
                    for label, count in year_labels.value_counts(dropna=False).items():
                        date_year_counts[(year, service, date_field, str(label))] += int(count)

                    if valid_count:
                        processing_period = pd.Period(year=year, month=int(audit["month"]), freq="M")
                        valid_periods = subset.loc[parsed.notna(), f"_{date_field}_period"]
                        lags = processing_period.ordinal - valid_periods.map(lambda period: period.ordinal)
                        for lag, count in lags.value_counts().items():
                            date_lag_counts[(year, service, date_field, int(lag))] += int(count)

                identifiers_valid = subset["N_AIH"].str.fullmatch(VALID_ID).fillna(False) & subset["N_AIH"].ne("0" * 13)
                valid_ids = subset.loc[identifiers_valid, "N_AIH"]
                grouped_ids = valid_ids.value_counts()
                for identifier, count in grouped_ids.items():
                    id_counts[(year, service, str(identifier))] += int(count)
                    id_counts[(year, "all3", str(identifier))] += int(count)
                for scope in (service, "all3"):
                    identifiers.setdefault((year, scope), set()).update(valid_ids.tolist())

                valid_hospital = subset["CNES"].str.fullmatch(VALID_HOSPITAL).fillna(False) & subset["CNES"].ne("0000000")
                count_rows(hospital_counts, year, service,
                           subset.loc[valid_hospital].groupby("CNES", sort=False).size())

                valid_municipality = (
                    subset["MUNIC_RES"].str.fullmatch(VALID_MUNICIPALITY).fillna(False)
                    & subset["MUNIC_RES"].ne("000000")
                    & subset["MUNIC_MOV"].str.fullmatch(VALID_MUNICIPALITY).fillna(False)
                    & subset["MUNIC_MOV"].ne("000000")
                )
                flow = subset.loc[valid_municipality].copy()
                flow["local"] = flow["MUNIC_RES"].eq(flow["MUNIC_MOV"]).astype("int64")
                grouped_municipality = flow.groupby("MUNIC_RES", sort=False).agg(n=("local", "size"), local=("local", "sum"))
                for municipality, row in grouped_municipality.iterrows():
                    key = (year, service, str(municipality))
                    old_n, old_local = municipality_counts.get(key, (0, 0))
                    municipality_counts[key] = (old_n + int(row["n"]), old_local + int(row["local"]))

                same_year = flow.loc[flow["_DT_INTER_year"].eq(year)]
                grouped_same = same_year.groupby("MUNIC_RES", sort=False).agg(n=("local", "size"), local=("local", "sum"))
                for municipality, row in grouped_same.iterrows():
                    key = (year, service, str(municipality))
                    old_n, old_local = same_year_municipality_counts.get(key, (0, 0))
                    same_year_municipality_counts[key] = (old_n + int(row["n"]), old_local + int(row["local"]))

                record_columns = ["N_AIH", "SEQ_AIH5", "PROC_REA", "ANO_CMPT", "MES_CMPT",
                                  "DT_INTER", "DT_SAIDA", "CNES", "MUNIC_RES", "MUNIC_MOV", "IDENT"]
                row_signature_counts(subset, year, service, record_columns, exact_record_counts)
                row_signature_counts(subset, year, "all3", record_columns, exact_record_counts)

                event_columns = ["PROC_REA", "DT_INTER", "DT_SAIDA", "CNES", "MUNIC_RES", "MUNIC_MOV"]
                row_signature_counts(subset, year, service, event_columns, event_counts)
                row_signature_counts(subset, year, "all3", event_columns, event_counts)

                valid_subset = subset.loc[identifiers_valid]
                id_event_columns = ["N_AIH"] + event_columns
                row_signature_counts(valid_subset, year, service, id_event_columns, identifier_event_counts)
                row_signature_counts(valid_subset, year, "all3", id_event_columns, identifier_event_counts)

            if processed_files % 10 == 0:
                save_manifest(file_rows)
            emit("file_complete", processed=processed_files, total=len(files),
                 file=source.name, processing_year=expected_year,
                 source_rows=int(audit["source_rows"]), selected_rows=int(audit["selected_rows"]))
            del df
            gc.collect()
        except Exception as exc:
            failure = {
                "source_file": str(source), "processing_year": expected_year,
                "status": "failed", "error_type": type(exc).__name__, "error": str(exc)[:500],
            }
            file_rows.append(failure)
            save_manifest(file_rows)
            emit("file_failed", processed=processed_files, total=len(files), file=source.name,
                 error_type=type(exc).__name__, error=str(exc)[:500])
            return 2

    if current_year == 2024:
        save_same_year_inputs(same_year_municipality_counts, (2023, 2024))
        emit("early_outcomes_ready", years=[2023, 2024],
             path=str(RESULTS / "same_year_admission_municipality_outcomes.csv"))
    save_manifest(file_rows, final=True)
    partial_manifest = RESULTS / "input_manifest.partial.csv"
    if partial_manifest.exists():
        partial_manifest.unlink()

    completeness_rows = []
    for year in (2017, 2018, 2019, 2023, 2024):
        rows = [row for row in file_rows if row.get("processing_year") == year and row.get("status") == "success"]
        pairs = [(row.get("uf"), int(row.get("month"))) for row in rows]
        pair_counts = Counter(pairs)
        observed_states = {uf for uf, _ in pairs}
        missing = sorted((state, month) for state in EXPECTED_STATES for month in range(1, 13)
                         if pair_counts[(state, month)] == 0)
        duplicates = sum(max(0, count - 1) for count in pair_counts.values())
        completeness_rows.append({
            "processing_year": year,
            "expected_month_state_files": 324,
            "observed_successful_files": len(rows),
            "observed_states": len(observed_states),
            "missing_state_month_pairs": len(missing),
            "duplicate_state_month_pairs": duplicates,
            "complete_27_states_x_12_months": len(rows) == 324 and len(observed_states) == 27
                and not missing and not duplicates,
        })
    pd.DataFrame(completeness_rows).to_csv(RESULTS / "input_completeness.csv", index=False,
                                           encoding="utf-8-sig")

    pd.DataFrame([{"processing_year": y, "service": s, "selected_records": n}
                  for (y, s), n in procedure_counts.items()]).sort_values(
        ["processing_year", "service"]).to_csv(RESULTS / "procedure_counts_by_processing_year.csv",
                                                  index=False, encoding="utf-8-sig")
    pd.DataFrame([{"processing_year": y, "service": s, "PROC_REA": code, "records": n}
                  for (y, s, code), n in selected_code_counts.items()]).sort_values(
        ["processing_year", "service", "PROC_REA"]).to_csv(
        RESULTS / "procedure_code_counts_by_processing_year.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"processing_year": y, "service": s, "ANO_CMPT": ano, "records": n}
                  for (y, s, ano), n in ano_counts.items()]).sort_values(
        ["processing_year", "service", "ANO_CMPT"]).to_csv(RESULTS / "counts_by_processing_year_and_ano_cmpt.csv",
                                                             index=False, encoding="utf-8-sig")
    pd.DataFrame([{"processing_year": y, "service": s, "date_field": field,
                   "date_year": date_year, "records": n}
                  for (y, s, field, date_year), n in date_year_counts.items()]).sort_values(
        ["processing_year", "service", "date_field", "date_year"]).to_csv(
        RESULTS / "counts_by_processing_year_and_claim_date_year.csv", index=False, encoding="utf-8-sig")

    date_summaries = []
    for year in (2017, 2018, 2019, 2023, 2024):
        for service in SERVICE_CODES:
            for field in ("DT_INTER", "DT_SAIDA"):
                lags = Counter({lag: n for (y, s, f, lag), n in date_lag_counts.items()
                                if y == year and s == service and f == field})
                status = {name: date_status_counts.get((year, service, field, name), 0)
                          for name in ("valid", "missing_or_invalid")}
                total_lags = sum(lags.values())
                date_summaries.append({
                    "processing_year": year, "service": service, "date_field": field,
                    "valid_dates": status["valid"], "missing_or_invalid_dates": status["missing_or_invalid"],
                    "date_year_equals_processing_year": sum(n for (y, s, f, date_year), n in date_year_counts.items()
                        if y == year and s == service and f == field and date_year == str(year)),
                    "date_year_before_processing_year": sum(n for (y, s, f, date_year), n in date_year_counts.items()
                        if y == year and s == service and f == field and date_year.isdigit() and int(date_year) < year),
                    "date_year_after_processing_year": sum(n for (y, s, f, date_year), n in date_year_counts.items()
                        if y == year and s == service and f == field and date_year.isdigit() and int(date_year) > year),
                    "processing_minus_date_month_min": min(lags) if lags else None,
                    "processing_minus_date_month_p50": weighted_quantile(lags, 0.50),
                    "processing_minus_date_month_p90": weighted_quantile(lags, 0.90),
                    "processing_minus_date_month_p95": weighted_quantile(lags, 0.95),
                    "processing_minus_date_month_max": max(lags) if lags else None,
                    "processing_month_later_than_date_rows": sum(n for lag, n in lags.items() if lag > 0),
                    "same_processing_month_rows": lags.get(0, 0),
                    "processing_month_before_date_rows": sum(n for lag, n in lags.items() if lag < 0),
                    "lagged_valid_date_rows": total_lags,
                })
    pd.DataFrame(date_summaries).to_csv(RESULTS / "date_lag_summary.csv", index=False,
                                        encoding="utf-8-sig")

    dup_summary = []
    scopes = ["all3", *SERVICE_CODES.keys()]
    for year in (2017, 2018, 2019, 2023, 2024):
        for scope in scopes:
            exact_rows = [(key, n) for key, n in exact_record_counts.items() if key[0] == year and key[1] == scope]
            event_rows = [(key, n) for key, n in event_counts.items() if key[0] == year and key[1] == scope]
            id_rows = [(key, n) for key, n in id_counts.items() if key[0] == year and key[1] == scope]
            id_event_rows = [(key, n) for key, n in identifier_event_counts.items() if key[0] == year and key[1] == scope]
            id_event_variant_counts: Counter = Counter()
            for key, _ in id_event_rows:
                id_event_variant_counts[key[:3]] += 1
            repeated_id_entries = [(key, n) for key, n in id_rows if n > 1]
            dup_summary.append({
                "processing_year": year,
                "scope": scope,
                "selected_field_identical_record_rows_beyond_first": sum(n - 1 for _, n in exact_rows if n > 1),
                "selected_field_identical_record_signature_groups": sum(1 for _, n in exact_rows if n > 1),
                "event_signature_rows_beyond_first": sum(n - 1 for _, n in event_rows if n > 1),
                "event_signature_groups_with_duplicates": sum(1 for _, n in event_rows if n > 1),
                "valid_identifier_groups": len(id_rows),
                "records_with_identifier_seen_more_than_once": sum(n for _, n in repeated_id_entries),
                "repeated_identifier_excess_records": sum(n - 1 for _, n in repeated_id_entries),
                "repeated_identifier_groups": len(repeated_id_entries),
                "repeated_identifiers_with_multiple_event_signatures": sum(
                    1 for key, n in repeated_id_entries if id_event_variant_counts.get(key, 0) > 1),
            })
    pd.DataFrame(dup_summary).to_csv(RESULTS / "within_year_duplicate_summary.csv", index=False,
                                     encoding="utf-8-sig")

    overlap_rows = []
    for scope in scopes:
        for i, first in enumerate((2017, 2018, 2019, 2023, 2024)):
            for second in (2017, 2018, 2019, 2023, 2024)[i + 1:]:
                first_ids = identifiers.get((first, scope), set())
                second_ids = identifiers.get((second, scope), set())
                overlap_rows.append({
                    "scope": scope, "processing_year_1": first, "processing_year_2": second,
                    "distinct_valid_N_AIH_in_year_1": len(first_ids),
                    "distinct_valid_N_AIH_in_year_2": len(second_ids),
                    "distinct_N_AIH_in_both_processing_years": len(first_ids & second_ids),
                })
    pd.DataFrame(overlap_rows).to_csv(RESULTS / "cross_year_identifier_overlap_counts.csv",
                                      index=False, encoding="utf-8-sig")

    save_same_year_inputs(same_year_municipality_counts, (2023, 2024))
    hospital_rows = [(year, service, cnes, count)
                     for (year, service, cnes), count in hospital_counts.items()]
    municipality_rows = [(year, service, municipality, values[0], values[1])
                         for (year, service, municipality), values in municipality_counts.items()]
    reconciliation = [
        reconcile_aggregate(
            "hospital", hospital_rows,
            SAVED_AGGREGATES / "hospital_annual_2017_2018_2019_2023_2024.csv",
            ["year", "service", "CNES"], ["n"]),
        reconcile_aggregate(
            "municipality", municipality_rows,
            SAVED_AGGREGATES / "municipality_annual_2017_2018_2019_2023_2024.csv",
            ["year", "service", "MUNIC_RES"], ["n", "local"]),
    ]
    (RESULTS / "aggregate_reconciliation.json").write_text(
        json.dumps(reconciliation, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "run_started_utc": RUN_UTC,
        "run_finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_files_discovered": len(files),
        "input_files_successful": sum(row.get("status") == "success" for row in file_rows),
        "processing_year_order": list(YEARS),
        "count_unit": "selected paid AIH records in source processing-year DBCs; no person-level deduplication",
        "source_ano_cmpt_mismatch_selected_records": sum(
            n for (year, service, ano), n in ano_counts.items() if ano != str(year)),
        "saved_aggregate_reconciliation": reconciliation,
        "all_years_complete_27x12": all(row["complete_27_states_x_12_months"] for row in completeness_rows),
    }
    (RESULTS / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_chinese_notes(summary, completeness_rows, dup_summary)
    emit("audit_complete", files=processed_files, years=list(YEARS),
         results=str(RESULTS), complete=summary["all_years_complete_27x12"])
    return 0


def write_chinese_notes(summary: dict, completeness_rows: list[dict], duplicate_rows: list[dict]) -> None:
    status_lines = [
        "# SIH 原始日期与跨年重叠审计",
        "",
        f"运行区间（UTC）：{summary['run_started_utc']} 至 {summary['run_finished_utc']}。",
        f"成功读取 DBC：{summary['input_files_successful']} / {summary['input_files_discovered']}。",
        "",
        "## 审计范围与口径",
        "",
        "读取 D:\\sih_rd_data 下 2017、2018、2019、2023、2024 五个处理年份的公开月度 DBC。手术代码为胆囊切除、剖宫产和阑尾切除各自指定代码。计数单位是所选付费 AIH 记录，不是患者；脚本仅在内存中使用 N_AIH 和相关字段作分组，没有导出行级记录或标识符。",
        "处理年份来自来源目录并与文件名 RD<UF><YY><MM>.dbc 交叉核对；ANO_CMPT 单独核对。DT_INTER 按入院日期处理，DT_SAIDA 按出院日期处理。日期 lag 以来源处理年月减去日期所在年月，正值表示该日期早于处理月份。",
        "完整性按 27 个州 × 12 个月 = 324 个文件/年核验。input_manifest.csv 为每个文件保留大小和 SHA-256。",
        "",
        "## 重复与标识符口径",
        "",
        "selected-field identical record 以 N_AIH、SEQ_AIH5、PROC_REA、ANO_CMPT、MES_CMPT、DT_INTER、DT_SAIDA、CNES、MUNIC_RES、MUNIC_MOV、IDENT 完全相同为准，统计首条之后的重复记录数。event signature 以 PROC_REA、DT_INTER、DT_SAIDA、CNES、MUNIC_RES、MUNIC_MOV 相同为准，不要求 N_AIH 相同。二者均为审计用所选字段签名，不等同于对 DBC 全字段字节的相同性判断。",
        "同一 N_AIH 的多条记录按不同 event signature 汇总，作为同一标识符对应多个所选事件记录的计数；输出仅汇总数量，不含实际标识符。跨年重叠只对符合 13 位数字且非全零的 N_AIH 计数，按处理年份对和 all3/单病种统计 distinct N_AIH 交集。",
        "",
        "## 同年入院结局敏感性输入",
        "",
        "same_year_admission_municipality_outcomes.csv 仅包含 2023/2024 中 DT_INTER 入院年份等于处理年份、且 MUNIC_RES 与 MUNIC_MOV 都为有效六位代码的记录；按 year、service、MUNIC_RES 聚合 n 与 local（MUNIC_RES == MUNIC_MOV）。这只是可用处理年文件中的同年入院记录，不是完整同年入院队列，因为后续月份的索赔尚不可见。",
        "",
        "## 结果文件",
        "",
        "- input_manifest.csv：来源文件、字节数、SHA-256、成功读取行数和所选记录数。",
        "- input_completeness.csv：每年州-月份覆盖。",
        "- procedure_counts_by_processing_year.csv 与 counts_by_processing_year_and_ano_cmpt.csv：按处理年份/ANO_CMPT 的记录计数。",
        "- counts_by_processing_year_and_claim_date_year.csv、date_lag_summary.csv：入院/出院日期年分布与处理月 lag 汇总。",
        "- within_year_duplicate_summary.csv、cross_year_identifier_overlap_counts.csv：重复记录与跨年 N_AIH 重叠的聚合计数。",
        "- aggregate_reconciliation.json：重新提取的医院/居住地市 n、local 与已保存年度聚合的分组核对。",
        "",
        "## 完整性结果",
        "",
    ]
    for row in completeness_rows:
        status_lines.append(
            f"- {row['processing_year']}：成功文件 {row['observed_successful_files']}/324；州数 {row['observed_states']}/27；缺失州-月份 {row['missing_state_month_pairs']}；重复州-月份 {row['duplicate_state_month_pairs']}；完整={row['complete_27_states_x_12_months']}。"
        )
    status_lines.extend(["", "## 重复记录汇总", ""])
    for row in duplicate_rows:
        if row["scope"] == "all3":
            status_lines.append(
                f"- {row['processing_year']} all3：所选字段完全重复记录超出首条 {row['selected_field_identical_record_rows_beyond_first']}；事件签名重复超出首条 {row['event_signature_rows_beyond_first']}；重复 N_AIH 组 {row['repeated_identifier_groups']}；其中含多个事件签名的组 {row['repeated_identifiers_with_multiple_event_signatures']}。"
            )
    status_lines.append("")
    (RESULTS / "audit_notes_zh.md").write_text("\n".join(status_lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
