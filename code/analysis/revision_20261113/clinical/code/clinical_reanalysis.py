#!/usr/bin/env python3
"""CNES-structure eligibility bounds and cholecystectomy label-masking audit.

This script reads public/previously aggregated facility inputs and writes only
aggregate facility, municipal, state, and population coverage summaries. It
does not read or export patient-level records.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path("__PROJECT_ROOT__")
STUDY = Path("__BRAZIL_SOURCE_ROOT__")
UPGRADE = BASE / "upgrade_20261112"
OUT = BASE / "revision_20261113" / "clinical"
RAW_CNES = STUDY / "01_data/raw/cnes/cnes_estabelecimentos.csv"
HOSPITALS = UPGRADE / "inputs/brazil_hospital_points.csv"
ACTIVITY = UPGRADE / "inputs/brazil_hospital_annual.csv"
GRID = BASE / "work/grid"
OLD_BOUNDS = UPGRADE / "derived/brazil_state_constraints_margin4"
MUNICIPALITY_SHP = STUDY / "river/data/official/ibge_malha/extracted/BR_Municipios_2022.shp"
PIXEL = 1.0 / 120.0
GRID_SHAPE = (17400, 43200)
HOSP_CODES = {"05", "07", "15", "20", "21"}
UF_CODES = {
    11: "RO", 12: "AC", 13: "AM", 14: "RR", 15: "PA", 16: "AP", 17: "TO",
    21: "MA", 22: "PI", 23: "CE", 24: "RN", 25: "PB", 26: "PE", 27: "AL",
    28: "SE", 29: "BA", 31: "MG", 32: "ES", 33: "RJ", 35: "SP", 41: "PR",
    42: "SC", 43: "RS", 50: "MS", 51: "MT", 52: "GO", 53: "DF",
}
SERVICES = {
    "chole": ("ST_CENTRO_CIRURGICO", "surgical_center"),
    "cesarean": ("ST_CENTRO_OBSTETRICO", "obstetric_center"),
    "appendectomy": ("ST_CENTRO_CIRURGICO", "surgical_center"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_cnes(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(7)


def normalize_flag(value: object) -> int | None:
    token = str(value).strip().upper()
    if token in {"1", "1.0"}:
        return 1
    if token in {"0", "0.0"}:
        return 0
    return None


def classify_flag(values: pd.Series) -> dict[str, object]:
    raw = [str(value).strip() for value in values.tolist()]
    normalized = [normalize_flag(value) for value in raw]
    has_one = any(value == 1 for value in normalized)
    has_zero = any(value == 0 for value in normalized)
    missing_count = sum(value == "" for value in raw)
    unrecognized_count = sum(value != "" and norm is None for value, norm in zip(raw, normalized))
    conflict = has_one and has_zero
    if conflict:
        status = "conflict_positive_and_zero"
    elif has_one:
        status = "positive"
    elif missing_count or unrecognized_count:
        status = "unknown"
    elif has_zero:
        status = "explicit_zero"
    else:
        status = "unknown"
    return {
        "status": status,
        "raw_rows": len(raw),
        "missing_rows": missing_count,
        "unrecognized_rows": unrecognized_count,
        "conflict": conflict,
        "raw_values": "|".join(sorted(set(raw))),
    }


def read_cnes_metadata(candidate_ids: set[str]) -> tuple[dict[str, dict[str, object]], int]:
    usecols = [
        "CO_CNES", "TP_UNIDADE", "ST_CENTRO_CIRURGICO",
        "ST_CENTRO_OBSTETRICO", "ST_ATEND_HOSPITALAR",
    ]
    chunks: list[pd.DataFrame] = []
    raw_rows = 0
    for chunk in pd.read_csv(
        RAW_CNES, sep=";", dtype=str, encoding="utf-8", usecols=usecols,
        chunksize=100_000, keep_default_na=False, low_memory=False,
    ):
        raw_rows += len(chunk)
        if raw_rows % 200_000 == 0 or raw_rows == len(chunk):
            print(json.dumps({"status": "cnes_scan", "rows_scanned": raw_rows}), flush=True)
        chunk["CO_CNES"] = normalize_cnes(chunk["CO_CNES"])
        matched = chunk.loc[chunk["CO_CNES"].isin(candidate_ids)].copy()
        if not matched.empty:
            chunks.append(matched)
    if not chunks:
        raise RuntimeError("No candidate CNES IDs matched the independent CNES source")
    matched = pd.concat(chunks, ignore_index=True)
    metadata: dict[str, dict[str, object]] = {}
    for cnes, group in matched.groupby("CO_CNES", sort=False):
        unit_values = sorted({str(v).strip() for v in group["TP_UNIDADE"] if str(v).strip()})
        unit_type = unit_values[0] if len(unit_values) == 1 else (
            "multiple:" + "|".join(unit_values) if unit_values else ""
        )
        surg = classify_flag(group["ST_CENTRO_CIRURGICO"])
        obst = classify_flag(group["ST_CENTRO_OBSTETRICO"])
        hospital_service = classify_flag(group["ST_ATEND_HOSPITALAR"])
        positive_hospital_type = unit_type in HOSP_CODES or hospital_service["status"] == "positive"
        type_unknown = not unit_values or bool(hospital_service["status"] == "unknown")
        metadata[cnes] = {
            "raw_record_count": len(group),
            "unit_type": unit_type,
            "hospital_type_positive": bool(positive_hospital_type),
            "hospital_type_unknown": bool(type_unknown),
            "hospital_service_status": hospital_service["status"],
            "surg_status": surg["status"],
            "surg_missing_rows": surg["missing_rows"],
            "surg_unrecognized_rows": surg["unrecognized_rows"],
            "surg_conflict": surg["conflict"],
            "surg_raw_values": surg["raw_values"],
            "obst_status": obst["status"],
            "obst_missing_rows": obst["missing_rows"],
            "obst_unrecognized_rows": obst["unrecognized_rows"],
            "obst_conflict": obst["conflict"],
            "obst_raw_values": obst["raw_values"],
        }
    return metadata, raw_rows


def write_append(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def load_aggregate_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    h = pd.read_csv(HOSPITALS, dtype={"cnes": str, "uf": str, "municipio6": str})
    h["cnes"] = normalize_cnes(h["cnes"])
    if h["cnes"].duplicated().any():
        raise ValueError("Candidate hospital table contains duplicate CNES identifiers")
    counts = pd.read_csv(ACTIVITY, dtype={"CNES": str})
    counts["CNES"] = normalize_cnes(counts["CNES"])
    counts["year"] = pd.to_numeric(counts["year"], errors="coerce")
    counts["n"] = pd.to_numeric(counts["n"], errors="coerce").fillna(0)
    return h, counts


def source_eligibility(
    h: pd.DataFrame, counts: pd.DataFrame, metadata: dict[str, dict[str, object]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    joined = pd.DataFrame.from_dict(metadata, orient="index").rename_axis("cnes").reset_index()
    h = h.merge(joined, on="cnes", how="left", validate="one_to_one")
    missing_record = h["raw_record_count"].isna()
    h["raw_record_count"] = h["raw_record_count"].fillna(0).astype(int)
    for field in ["surg_status", "obst_status", "hospital_service_status"]:
        h[field] = h[field].fillna("unknown")
    for field in ["surg_conflict", "obst_conflict", "hospital_type_positive", "hospital_type_unknown"]:
        h[field] = h[field].fillna(False).astype(bool)
    h["unit_type"] = h["unit_type"].fillna("")
    h["surg_missing_rows"] = h["surg_missing_rows"].fillna(0).astype(int)
    h["surg_unrecognized_rows"] = h["surg_unrecognized_rows"].fillna(0).astype(int)
    h["obst_missing_rows"] = h["obst_missing_rows"].fillna(0).astype(int)
    h["obst_unrecognized_rows"] = h["obst_unrecognized_rows"].fillna(0).astype(int)
    h["metadata_record_missing"] = missing_record

    c24 = counts.loc[counts["year"].eq(2024)].groupby(["service", "CNES"], as_index=False)["n"].sum()
    activity_map = {
        (str(row.service), str(row.CNES)): float(row.n)
        for row in c24.itertuples(index=False)
    }
    eligibility_rows: list[dict[str, object]] = []
    for service, (raw_field, flag_kind) in SERVICES.items():
        status_col = "obst_status" if service == "cesarean" else "surg_status"
        active_volumes = np.array([
            activity_map.get((service, cnes), 0.0) for cnes in h["cnes"].tolist()
        ])
        h[f"activity_{service}_2024"] = active_volumes
        h[f"active_{service}_2024"] = active_volumes > 0
        status = h[status_col].astype(str).to_numpy()
        pos = status == "positive"
        conflict = status == "conflict_positive_and_zero"
        unknown = status == "unknown"
        explicit_zero = status == "explicit_zero"
        broad = pos | conflict | unknown
        affirmative = pos
        for set_name, included in [
            ("full_original", np.ones(len(h), dtype=bool)),
            ("A_affirmative_structure_only", affirmative),
            ("B_structure_positive_or_unknown", broad),
        ]:
            active_in = included & (active_volumes > 0)
            active_out = (~included) & (active_volumes > 0)
            eligibility_rows.append({
                "service": service,
                "structure_indicator": flag_kind,
                "candidate_set": set_name,
                "candidate_count": int(included.sum()),
                "active_hospitals_2024_in_set": int(active_in.sum()),
                "activity_records_2024_in_set": int(active_volumes[active_in].sum()),
                "active_hospitals_excluded": int(active_out.sum()),
                "activity_records_excluded": int(active_volumes[active_out].sum()),
                "flag_positive_candidates": int(pos.sum()),
                "flag_conflicted_candidates": int(conflict.sum()),
                "flag_unknown_candidates": int(unknown.sum()),
                "flag_explicit_zero_candidates": int(explicit_zero.sum()),
                "active_with_explicit_zero": int(((active_volumes > 0) & explicit_zero).sum()),
                "active_with_conflicted_metadata": int(((active_volumes > 0) & conflict).sum()),
                "active_with_unknown_metadata": int(((active_volumes > 0) & unknown).sum()),
                "candidate_rows_missing_from_raw_CNES": int((included & missing_record).sum()),
                "candidate_with_hospital_type_support": int((included & h["hospital_type_positive"].to_numpy()).sum()),
                "candidate_with_hospital_type_unknown": int((included & h["hospital_type_unknown"].to_numpy()).sum()),
                "raw_indicator_missing_fields_in_set": int((
                    included & h["surg_missing_rows"].gt(0).to_numpy()
                    if service != "cesarean"
                    else included & h["obst_missing_rows"].gt(0).to_numpy()
                ).sum()),
                "raw_indicator_unrecognized_fields_in_set": int((
                    included & h["surg_unrecognized_rows"].gt(0).to_numpy()
                    if service != "cesarean"
                    else included & h["obst_unrecognized_rows"].gt(0).to_numpy()
                ).sum()),
            })
    return h, pd.DataFrame(eligibility_rows)


def load_spatial_inputs(margin_degrees: int) -> dict[str, object]:
    scope = pd.read_csv(UPGRADE / "derived/country_scope/population_and_bounds.csv")
    row = scope.loc[scope["iso3"].eq("BRA")].iloc[0]
    margin = int(margin_degrees * 120)
    r0 = max(0, int(row.r0) - margin)
    r1 = min(GRID_SHAPE[0], int(row.r1) + margin)
    c0 = max(0, int(row.c0) - margin)
    c1 = min(GRID_SHAPE[1], int(row.c1) + margin)
    h, w = r1 - r0, c1 - c0
    top = 85.0 - r0 * PIXEL
    friction = np.memmap(GRID / "friction.f32", dtype="float32", mode="r", shape=GRID_SHAPE)[r0:r1, c0:c1].copy()
    population = np.memmap(GRID / "pop2025.f32", dtype="float32", mode="r", shape=GRID_SHAPE)[r0:r1, c0:c1].copy()
    country = np.memmap(GRID / "country.i16", dtype="int16", mode="r", shape=GRID_SHAPE)[r0:r1, c0:c1]
    urban = np.memmap(GRID / "urban.u8", dtype="uint8", mode="r", shape=GRID_SHAPE)[r0:r1, c0:c1].astype(bool)
    rural = np.memmap(GRID / "rural.u8", dtype="uint8", mode="r", shape=GRID_SHAPE)[r0:r1, c0:c1].astype(bool)
    target = country == int(row.idx)
    weights = np.where(target, population, 0.0)
    population_sum = float(weights.sum(dtype=np.float64))
    if abs(population_sum / float(row.population) - 1.0) >= 1e-8:
        raise ValueError("Brazil population denominator does not match the established country mask")

    import geopandas as gpd
    import rasterio
    from rasterio import features

    polygons = gpd.read_file(MUNICIPALITY_SHP)
    codecol = next(c for c in polygons.columns if c.upper().startswith("CD_MUN"))
    polygons["m6"] = pd.to_numeric(polygons[codecol].astype(str).str[:6], errors="coerce")
    transform = rasterio.transform.from_origin(-180 + c0 * PIXEL, top, PIXEL, PIXEL)
    municipality = features.rasterize(
        zip(polygons.geometry, polygons["m6"].fillna(0).astype(int)),
        out_shape=(h, w), transform=transform, fill=0, dtype="int32",
    )
    valid_muni = (municipality > 0) & target
    unmatched_target = target & ~valid_muni
    state_grid = municipality // 10000
    state_codes, state_inverse = np.unique(state_grid[valid_muni], return_inverse=True)
    state_population = np.bincount(
        state_inverse, weights=population[valid_muni].astype(np.float64)
    )
    assigned_population_sum = float(state_population.sum(dtype=np.float64))
    unmatched_population_sum = float(population[unmatched_target].sum(dtype=np.float64))
    if not np.isclose(
        assigned_population_sum + unmatched_population_sum,
        population_sum,
        rtol=0.0,
        atol=max(1e-6, population_sum * 1e-10),
    ):
        raise ValueError("Assigned and unmatched state-polygon support does not reconcile to the Brazil country mask")
    state_population_rows = [
        {
            "state_code": int(code),
            "state": UF_CODES.get(int(code), str(code)),
            "polygon_assigned_population": float(pop),
            "share_of_full_country_mask_population": float(pop / population_sum),
        }
        for code, pop in zip(state_codes, state_population)
    ]
    denominator_audit = {
        "country": "BRA",
        "country_mask_grid_population": population_sum,
        "country_scope_reference_population": float(row.population),
        "country_mask_relative_difference_from_reference": float(population_sum / float(row.population) - 1.0),
        "country_mask_grid_pixels": int(target.sum()),
        "state_polygon_assigned_population": assigned_population_sum,
        "state_polygon_assigned_share_of_country_mask_population": float(assigned_population_sum / population_sum),
        "state_polygon_assigned_grid_pixels": int(valid_muni.sum()),
        "unmatched_country_mask_population": unmatched_population_sum,
        "unmatched_share_of_country_mask_population": float(unmatched_population_sum / population_sum),
        "unmatched_country_mask_grid_pixels": int(unmatched_target.sum()),
        "state_population_by_code": state_population_rows,
        "country_metric_denominator": "all population cells in the established Brazil country mask",
        "state_metric_denominator": "population in the Brazil country mask assigned to rasterized 2022 IBGE municipality polygons and grouped by state; unmatched cells are excluded from state denominators and are not redistributed",
    }

    return {
        "r0": r0, "r1": r1, "c0": c0, "c1": c1, "H": h, "W": w, "top": top,
        "friction": friction, "population": population, "country": country,
        "urban": urban, "rural": rural, "target": target, "weights": weights,
        "valid_muni": valid_muni, "state_codes": state_codes,
        "state_inverse": state_inverse, "state_population": state_population,
        "state_names": [UF_CODES.get(int(code), str(code)) for code in state_codes],
        "population_sum": population_sum, "denominator_audit": denominator_audit,
    }


def facility_groups(h: pd.DataFrame, spatial: dict[str, object]) -> dict[str, object]:
    lat = pd.to_numeric(h["lat"], errors="coerce").to_numpy()
    lon = pd.to_numeric(h["lon"], errors="coerce").to_numpy()
    rr = np.floor((85.0 - lat) / PIXEL).astype(int) - int(spatial["r0"])
    cc = np.floor((lon + 180.0) / PIXEL).astype(int) - int(spatial["c0"])
    H, W = int(spatial["H"]), int(spatial["W"])
    if not ((rr >= 1) & (rr < H - 1) & (cc >= 1) & (cc < W - 1)).all():
        raise ValueError("A candidate facility coordinate falls outside the established crop")
    population = spatial["population"]
    density = np.array([
        population[r - 1:r + 2, c - 1:c + 2].sum(dtype=np.float64)
        for r, c in zip(rr, cc)
    ])
    high = density > np.median(density)
    smod_urban = spatial["urban"][rr, cc]
    state = h["uf"].astype(str).str.zfill(2).to_numpy()
    group_codes, unique_groups = pd.MultiIndex.from_arrays(
        [state, high.astype(int), smod_urban.astype(int)]
    ).factorize()
    cell = np.floor(lat).astype(int) * 1000 + np.floor(lon).astype(int)
    return {
        "rr": rr, "cc": cc, "density": density, "high": high,
        "smod_urban": smod_urban, "state": state, "groups": group_codes,
        "unique_groups": unique_groups, "cell": cell,
    }


def build_c_engine() -> Path:
    code = BASE / "src/ttdijkstra.c"
    source = code.read_text(encoding="utf-8")
    source = source.replace(
        "const double PIX = 1.0 / 120.0;",
        "const double PIX = 1.0 / 120.0;\n    const double LAT_TOP = (argc > 8) ? atof(argv[8]) : 85.0;",
    )
    source = source.replace("double lat = 85.0 - PIX", "double lat = LAT_TOP - PIX")
    code_out = OUT / "code/ttdijkstra_clinical.c"
    code_out.parent.mkdir(parents=True, exist_ok=True)
    code_out.write_text(source, encoding="utf-8")
    binary = OUT / "code/ttdijkstra_clinical"
    subprocess.run(["gcc", "-O3", "-o", str(binary), str(code_out), "-lm"], check=True)
    return binary


def save_state_rows(
    path: Path, candidate_set: str, scenario: str, coverage30: np.ndarray,
    spatial: dict[str, object], selection_sha256: str = "",
) -> None:
    rows = []
    for i, code in enumerate(spatial["state_codes"]):
        rows.append({
            "candidate_set": candidate_set,
            "scenario": scenario,
            "state_code": int(code),
            "state": UF_CODES.get(int(code), str(code)),
            "coverage30": float(coverage30[i]),
            "selection_sha256": selection_sha256,
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def evaluate_selection(
    binary: Path,
    spatial: dict[str, object],
    facilities: dict[str, object],
    selection: np.ndarray,
    candidate_set: str,
    scenario: str,
    scenario_type: str,
    service: str,
    eligible_n: int,
    observed_positive_n: int,
    selection_sha256: str,
    out_results: Path,
    out_states: Path,
) -> tuple[dict[str, object], np.ndarray]:
    selection = np.asarray(selection, dtype=int)
    if len(selection) != observed_positive_n or len(np.unique(selection)) != len(selection):
        raise ValueError(f"Invalid selection size or duplicate facility in {scenario}")
    H, W = int(spatial["H"]), int(spatial["W"])
    source = np.zeros((H, W), dtype=np.uint8)
    rr, cc = facilities["rr"], facilities["cc"]
    source[rr[selection], cc[selection]] = 1
    unique_source_pixels = int(source.sum())
    work = OUT / "work"
    work.mkdir(parents=True, exist_ok=True)
    source_path = work / "sources.u8"
    friction_path = work / "friction_crop.f32"
    travel_path = work / "travel_time.f32"
    source.tofile(source_path)
    if not friction_path.exists():
        spatial["friction"].tofile(friction_path)
    log_path = OUT / "logs" / f"{candidate_set}_{service}_{scenario}.log"
    args = [
        str(binary), str(friction_path), str(source_path), str(travel_path),
        str(H), str(W), "120", "0", str(float(spatial["top"])),
    ]
    print(json.dumps({
        "status": "dijkstra_start",
        "candidate_set": candidate_set,
        "service": service,
        "scenario": scenario,
        "selected_hospitals": int(len(selection)),
        "grid_cells": int(H * W),
    }), flush=True)
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True)
    dijkstra_seconds = time.perf_counter() - start
    travel = np.memmap(travel_path, dtype="float32", mode="r", shape=(H, W))
    population = spatial["population"]
    target = spatial["target"]
    urban = spatial["urban"]
    rural = spatial["rural"]
    row: dict[str, object] = {
        "candidate_set": candidate_set,
        "scenario": scenario,
        "service": service,
        "scenario_type": scenario_type,
        "eligible_candidates": int(eligible_n),
        "observed_positive_hospitals": int(observed_positive_n),
        "selected_hospitals": int(len(selection)),
        "unique_source_pixels": unique_source_pixels,
        "selection_sha256": selection_sha256,
        "reused_from_scenario": "",
        "dijkstra_ran": True,
        "dijkstra_seconds": round(dijkstra_seconds, 3),
        "realized_high_local_population_share": float(facilities["high"][selection].mean()) if len(selection) else np.nan,
        "realized_smod_urban_share": float(facilities["smod_urban"][selection].mean()) if len(selection) else np.nan,
    }
    for threshold in [15, 30, 60, 120]:
        covered = travel <= threshold
        for label, mask in [
            ("all", target),
            ("urban", target & urban),
            ("rural", target & rural),
        ]:
            denominator = float(population[mask].sum(dtype=np.float64))
            numerator = float(population[mask & covered].sum(dtype=np.float64))
            row[f"{label}_{threshold}"] = numerator / denominator if denominator > 0 else np.nan
    valid_muni = spatial["valid_muni"]
    state_population = population[valid_muni].astype(np.float64)
    travel_values = np.asarray(travel[valid_muni])
    state_inverse = spatial["state_inverse"]
    state_covered = np.bincount(
        state_inverse,
        weights=state_population * (travel_values <= 30),
        minlength=len(spatial["state_codes"]),
    )
    state_coverage = state_covered / spatial["state_population"]
    save_state_rows(out_states, candidate_set, scenario, state_coverage, spatial, selection_sha256)
    write_append(out_results, row)
    del travel
    return row, state_coverage


def stable_selection_hash(cnes: np.ndarray, selection: np.ndarray) -> str:
    payload = "\n".join(sorted(str(value) for value in cnes[np.asarray(selection, dtype=int)]))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def evaluate_or_reuse(
    binary: Path,
    spatial: dict[str, object],
    facilities: dict[str, object],
    cnes: np.ndarray,
    selection: np.ndarray,
    candidate_set: str,
    scenario: str,
    scenario_type: str,
    service: str,
    eligible_n: int,
    observed_positive_n: int,
    out_results: Path,
    out_states: Path,
    cache: dict[str, dict[str, object]],
) -> tuple[dict[str, object], np.ndarray]:
    selection = np.asarray(selection, dtype=int)
    selection_sha256 = stable_selection_hash(cnes, selection)
    if selection_sha256 in cache:
        prior = cache[selection_sha256]
        row: dict[str, object] = {
            "candidate_set": candidate_set,
            "scenario": scenario,
            "service": service,
            "scenario_type": scenario_type,
            "eligible_candidates": int(eligible_n),
            "observed_positive_hospitals": int(observed_positive_n),
            "selected_hospitals": int(len(selection)),
            "unique_source_pixels": prior["unique_source_pixels"],
            "selection_sha256": selection_sha256,
            "reused_from_scenario": str(prior["source_scenario"]),
            "dijkstra_ran": False,
            "dijkstra_seconds": 0.0,
            "realized_high_local_population_share": float(facilities["high"][selection].mean()) if len(selection) else np.nan,
            "realized_smod_urban_share": float(facilities["smod_urban"][selection].mean()) if len(selection) else np.nan,
        }
        row.update(prior["coverage"])
        state_coverage = np.asarray(prior["state_coverage"], dtype=float).copy()
        save_state_rows(out_states, candidate_set, scenario, state_coverage, spatial, selection_sha256)
        write_append(out_results, row)
        return row, state_coverage
    row, state_coverage = evaluate_selection(
        binary, spatial, facilities, selection, candidate_set, scenario,
        scenario_type, service, eligible_n, observed_positive_n,
        selection_sha256, out_results, out_states,
    )
    coverage_columns = {
        key: value for key, value in row.items()
        if key in {
            f"{scope}_{threshold}"
            for scope in ["all", "urban", "rural"]
            for threshold in [15, 30, 60, 120]
        }
    }
    cache[selection_sha256] = {
        "coverage": coverage_columns,
        "state_coverage": state_coverage.copy(),
        "unique_source_pixels": row["unique_source_pixels"],
        "source_scenario": scenario,
    }
    return row, state_coverage


def feasible_group_counts(
    groups: np.ndarray, observed_idx: np.ndarray, eligible_idx: np.ndarray
) -> dict[int, int]:
    quotas = {
        int(group): int(np.count_nonzero(groups[observed_idx] == group))
        for group in np.unique(groups[observed_idx])
    }
    eligible_counts = {
        int(group): int(np.count_nonzero(groups[eligible_idx] == group))
        for group in np.unique(groups[eligible_idx])
    }
    for group, quota in quotas.items():
        if eligible_counts.get(group, 0) < quota:
            raise ValueError(f"Infeasible state×joint quota in group {group}: {quota}")
    return quotas


def construct_cluster(
    eligible_idx: np.ndarray,
    groups: np.ndarray,
    cells: np.ndarray,
    quotas: dict[int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    selected: list[int] = []
    for group, quota in quotas.items():
        candidates = eligible_idx[groups[eligible_idx] == group]
        occupied = np.unique(cells[candidates])
        order = rng.permutation(occupied)
        group_selected: list[int] = []
        for cell in order:
            same_cell = candidates[cells[candidates] == cell]
            take = min(len(same_cell), quota - len(group_selected))
            if take > 0:
                group_selected.extend(same_cell[:take].tolist())
            if len(group_selected) == quota:
                break
        if len(group_selected) != quota:
            raise ValueError(f"Could not satisfy clustered quota for group {group}")
        selected.extend(group_selected)
    return np.asarray(selected, dtype=int)


def construct_random(
    eligible_idx: np.ndarray,
    groups: np.ndarray,
    quotas: dict[int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    selected: list[int] = []
    for group, quota in quotas.items():
        candidates = eligible_idx[groups[eligible_idx] == group]
        if len(candidates) < quota:
            raise ValueError(f"Could not satisfy random quota for group {group}")
        if quota:
            selected.extend(rng.choice(candidates, size=quota, replace=False).tolist())
    return np.asarray(selected, dtype=int)


def construct_local_priority(
    eligible_idx: np.ndarray,
    groups: np.ndarray,
    density: np.ndarray,
    cnes: np.ndarray,
    quotas: dict[int, int],
    highest_first: bool,
) -> np.ndarray:
    selected: list[int] = []
    for group, quota in quotas.items():
        candidates = eligible_idx[groups[eligible_idx] == group]
        direction = -1.0 if highest_first else 1.0
        order = sorted(candidates.tolist(), key=lambda i: (direction * float(density[i]), str(cnes[i])))
        selected.extend(order[:quota])
    return np.asarray(selected, dtype=int)


def summarize_masking(
    scenario_rows: pd.DataFrame,
    state_rows: pd.DataFrame,
    out_metrics: Path,
    out_rank: Path,
) -> None:
    metrics: list[dict[str, object]] = []
    rank_rows: list[dict[str, object]] = []
    for candidate_set in ["full_original", "B_structure_positive_or_unknown"]:
        states = state_rows.loc[state_rows["candidate_set"].eq(candidate_set)].copy()
        reference_scenario = (
            "observed_activity_reference"
            if candidate_set == "full_original"
            else "chole_B_observed_subset"
        )
        truth = states.loc[states["scenario"].eq(reference_scenario)].set_index("state_code")["coverage30"]
        if truth.empty:
            raise ValueError(f"Missing observed reference for {candidate_set}")
        truth_rank = truth.rank(method="average", ascending=True)
        random_scenarios = sorted(
            s for s in states["scenario"].unique().tolist()
            if s.startswith("random_")
        )
        estimates_by_rule: list[tuple[str, list[str]]] = [
            ("random", random_scenarios),
            ("high_local_population_priority", ["local_population_high_priority"]),
            ("low_local_population_priority", ["local_population_low_priority"]),
        ]
        for rule, names in estimates_by_rule:
            per_draw = []
            for name in names:
                estimate = states.loc[states["scenario"].eq(name)].set_index("state_code")["coverage30"]
                aligned = pd.concat([truth.rename("truth"), estimate.rename("estimate")], axis=1).dropna()
                if len(aligned) < 3:
                    continue
                abs_error_pp = (aligned["estimate"] - aligned["truth"]).abs() * 100
                true_rank = aligned["truth"].rank(method="average", ascending=True)
                estimated_rank = aligned["estimate"].rank(method="average", ascending=True)
                spearman = float(true_rank.corr(estimated_rank, method="pearson"))
                truth_low5 = set(true_rank.nsmallest(5).index)
                estimate_low5 = set(estimated_rank.nsmallest(5).index)
                estimate_national = scenario_rows.loc[
                    scenario_rows["candidate_set"].eq(candidate_set)
                    & scenario_rows["scenario"].eq(name),
                    "all_30",
                ]
                reference_national = scenario_rows.loc[
                    scenario_rows["candidate_set"].eq(candidate_set)
                    & scenario_rows["scenario"].eq(reference_scenario),
                    "all_30",
                ]
                if estimate_national.empty or reference_national.empty:
                    raise ValueError(f"Missing national coverage row for {candidate_set}:{name}")
                per_draw.append({
                    "candidate_set": candidate_set,
                    "rule": rule,
                    "scenario": name,
                    "national_abs_error_pp": abs(
                        float(estimate_national.iloc[0]) - float(reference_national.iloc[0])
                    ) * 100,
                    "mean_absolute_state_error_pp": float(abs_error_pp.mean()),
                    "maximum_absolute_state_error_pp": float(abs_error_pp.max()),
                    "state_rank_spearman": spearman,
                    "lowest_five_state_overlap": int(len(truth_low5 & estimate_low5)),
                })
            metrics.extend(per_draw)
        high_priority = states.loc[
            states["scenario"].eq("local_population_high_priority")
        ].set_index("state_code")["coverage30"]
        low_priority = states.loc[
            states["scenario"].eq("local_population_low_priority")
        ].set_index("state_code")["coverage30"]
        high_rank = high_priority.rank(method="average", ascending=True)
        low_rank = low_priority.rank(method="average", ascending=True)
        for code in truth.index:
            row: dict[str, object] = {
                "candidate_set": candidate_set,
                "state_code": int(code),
                "state": UF_CODES.get(int(code), str(code)),
                "observed_coverage30": float(truth.loc[code]),
                "observed_rank_lowest_first": float(truth_rank.loc[code]),
            }
            random_ranks = []
            random_coverages = []
            random_bottom5 = []
            for name in random_scenarios:
                estimate = states.loc[
                    states["scenario"].eq(name)
                ].set_index("state_code")["coverage30"]
                random_ranks.append(float(estimate.rank(method="average", ascending=True).loc[code]))
                random_coverages.append(float(estimate.loc[code]))
                random_bottom5.append(code in set(estimate.nsmallest(5).index))
            row.update({
                "random_coverage30_median": float(np.median(random_coverages)),
                "random_coverage30_min": float(np.min(random_coverages)),
                "random_coverage30_max": float(np.max(random_coverages)),
                "random_rank_median": float(np.median(random_ranks)),
                "random_rank_min": float(np.min(random_ranks)),
                "random_rank_max": float(np.max(random_ranks)),
                "random_lowest_five_probability": float(np.mean(random_bottom5)),
                "high_priority_coverage30": float(high_priority.loc[code]),
                "high_priority_rank_lowest_first": float(high_rank.loc[code]),
                "high_priority_rank_shift_vs_observed": float(high_rank.loc[code]) - float(truth_rank.loc[code]),
                "high_priority_lowest_five": bool(code in set(high_priority.nsmallest(5).index)),
                "low_priority_coverage30": float(low_priority.loc[code]),
                "low_priority_rank_lowest_first": float(low_rank.loc[code]),
                "low_priority_rank_shift_vs_observed": float(low_rank.loc[code]) - float(truth_rank.loc[code]),
                "low_priority_lowest_five": bool(code in set(low_priority.nsmallest(5).index)),
            })
            rank_rows.append(row)
    if metrics:
        metric_frame = pd.DataFrame(metrics)
        metric_frame.to_csv(out_metrics, index=False)
    if rank_rows:
        pd.DataFrame(rank_rows).to_csv(out_rank, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--margin-degrees", type=int, default=4)
    parser.add_argument("--m2-repetitions", type=int, default=8)
    parser.add_argument("--m1-random-repetitions", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--result-dir", type=Path, default=OUT / "results")
    args = parser.parse_args()
    result_dir = args.result_dir
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "eligibility_status.csv", "m2_scenarios.csv", "m2_bounds.csv",
        "m1_scenarios.csv", "m1_state_coverage.csv", "m2_state_coverage.csv",
        "m1_scenario_metrics.csv", "m1_state_rank_change.csv",
        "run_manifest.json", "population_denominator_audit.json",
        "state_population_denominators.csv", "observed_reference_comparison.json",
    ]:
        path = result_dir / name
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite prior aggregate output: {path}")

    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    h, counts = load_aggregate_inputs()
    metadata, raw_rows = read_cnes_metadata(set(h["cnes"]))
    h, eligibility = source_eligibility(h, counts, metadata)
    eligibility.to_csv(result_dir / "eligibility_status.csv", index=False)
    spatial = load_spatial_inputs(args.margin_degrees)
    (result_dir / "population_denominator_audit.json").write_text(
        json.dumps(spatial["denominator_audit"], indent=2), encoding="utf-8"
    )
    pd.DataFrame(spatial["denominator_audit"]["state_population_by_code"]).to_csv(
        result_dir / "state_population_denominators.csv", index=False
    )
    facilities = facility_groups(h, spatial)
    rng = np.random.default_rng(args.seed)
    binary = build_c_engine()

    manifest = {
        "started_at_utc": started,
        "result_dir": str(result_dir),
        "analysis": "Brazil 2025 CNES structure screens for three procedures plus 2024 SUS AIH cholecystectomy label masking",
        "cnes_source": {
            "path": str(RAW_CNES),
            "source_url": "https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/CNES/cnes_estabelecimentos_csv.zip",
            "snapshot": "CNES 2025 establishment snapshot, from surgical_desert project contract",
            "size_bytes": RAW_CNES.stat().st_size,
            "sha256_from_source_manifest": "308372de39054d0dc4a26108ff5e96e4abd8a1da6a730e82343a8c84d40d7f73",
            "raw_rows_scanned": int(raw_rows),
            "candidate_records_matched": int(sum(int(v["raw_record_count"]) for v in metadata.values())),
            "candidate_ids_matched": int(len(metadata)),
        },
        "candidate_points": {
            "path": str(HOSPITALS), "rows": int(len(h)),
            "sha256": sha256(HOSPITALS), "size_bytes": HOSPITALS.stat().st_size,
        },
        "activity_aggregate": {
            "path": str(ACTIVITY), "sha256": sha256(ACTIVITY),
            "size_bytes": ACTIVITY.stat().st_size,
            "label_definition": "facility has >=1 2024 SUS-funded AIH record in the named procedure group",
            "not_gold_standard_for": "all-provider capability, private activity, need, or full clinical eligibility",
        },
        "spatial_inputs": {
            "grid_directory": str(GRID),
            "population": "pop2025.f32",
            "country_mask": "country.i16",
            "motorized_friction": "friction.f32",
            "settlement_masks": ["urban.u8", "rural.u8"],
            "municipality_shapes": str(MUNICIPALITY_SHP),
            "crop_margin_degrees": int(args.margin_degrees),
            "single_thread_c_engine": str(binary),
        },
        "population_denominators": spatial["denominator_audit"],
        "eligibility_rules": {
            "A_affirmative_structure_only": "relevant CNES structure flag is an unambiguous positive; missing, unknown, conflict, and explicit zero are outside this affirmative-only subset, not recoded as negative",
            "B_structure_positive_or_unknown": "relevant CNES structure flag is positive, missing/unrecognized, or internally conflicting; only explicit zero without another status is excluded",
            "chole_and_appendectomy": "ST_CENTRO_CIRURGICO",
            "cesarean": "ST_CENTRO_OBSTETRICO",
            "observed_positive_override": False,
            "hospital_type_codes_descriptive": sorted(HOSP_CODES),
            "unavailable_in_source": ["procedure-specific anesthesia record", "procedure-specific named surgical service"],
        },
        "quota_definition": "within each selected candidate set, preserve observed-active 2024 hospital counts in state × above/below full-original-candidate median of local 3×3 population × facility-point SMOD urban/rural cells",
        "m2": {
            "procedures": list(SERVICES),
            "repetitions_per_eligible_set_and_procedure": int(args.m2_repetitions),
        },
        "m1": {
            "procedure": "chole",
            "random_repetitions_per_set": int(args.m1_random_repetitions),
            "rules": [
                "uniform random assignment within exact state×joint quota cells",
                "highest continuous local-population density first within the same exact quota cells",
                "lowest continuous local-population density first within the same exact quota cells",
            ],
            "reference": "observed SUS-funded procedure-activity configuration, a source-specific activity reference rather than a capability gold standard",
            "reported_subnational_unit": "Brazilian state; coverage permits cross-state travel to any selected facility",
            "interpretation": "descriptive empirical masking; 30 minutes is a measurement threshold, not a clinical or WHO adequacy target for elective cholecystectomy",
        },
        "seed": int(args.seed),
        "outputs_are_aggregate_only": True,
    }
    (result_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Recompute the full observed cholecystectomy activity configuration with
    # this run's engine and denominators. It is a source-specific activity
    # reference, not a gold standard for all capability.
    m1_scenarios_path = result_dir / "m1_scenarios.csv"
    m1_state_path = result_dir / "m1_state_coverage.csv"
    m2_scenarios_path = result_dir / "m2_scenarios.csv"
    m2_state_path = result_dir / "m2_state_coverage.csv"
    cnes = h["cnes"].astype(str).to_numpy()
    observed_full_chole = np.flatnonzero(h["activity_chole_2024"].to_numpy(dtype=float) > 0)
    evaluation_cache: dict[str, dict[str, object]] = {}

    full_indices = np.arange(len(h), dtype=int)
    activity = {
        service: h[f"activity_{service}_2024"].to_numpy(dtype=float)
        for service in SERVICES
    }
    full_observed, _ = evaluate_or_reuse(
        binary, spatial, facilities, cnes, observed_full_chole,
        "full_original", "observed_activity_reference",
        "observed_activity_registry_reference", "chole", len(full_indices),
        len(observed_full_chole), m1_scenarios_path, m1_state_path,
        evaluation_cache,
    )
    prior_coverage = pd.read_csv(OLD_BOUNDS / "coverage_results.csv")
    prior_observed = prior_coverage.loc[
        prior_coverage["scenario"].eq("chole_observed"), "all_30"
    ]
    if len(prior_observed) != 1:
        raise RuntimeError("Expected one prior national cholecystectomy reference for comparison")
    observed_reference_comparison = {
        "prior_country_all_30": float(prior_observed.iloc[0]),
        "recomputed_country_all_30": float(full_observed["all_30"]),
        "difference_percentage_points": float(
            (full_observed["all_30"] - float(prior_observed.iloc[0])) * 100.0
        ),
        "prior_state_reference_reused": False,
        "recomputed_state_reference_denominator": spatial["denominator_audit"]["state_metric_denominator"],
    }
    (result_dir / "observed_reference_comparison.json").write_text(
        json.dumps(observed_reference_comparison, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "status": "recomputed_observed_reference",
        "all_30": full_observed["all_30"],
        "prior_all_30": observed_reference_comparison["prior_country_all_30"],
        "difference_percentage_points": observed_reference_comparison["difference_percentage_points"],
        "dijkstra_seconds": full_observed["dijkstra_seconds"],
    }), flush=True)
    set_names = ["A_affirmative_structure_only", "B_structure_positive_or_unknown"]
    broad_chole_row: dict[str, object] | None = None
    broad_chole_state: np.ndarray | None = None

    # M2: each procedure is screened against its own CNES service-structure
    # indicator. Budgets and state×joint summaries are recalculated within A/B.
    for service, (raw_field, flag_kind) in SERVICES.items():
        status_col = "obst_status" if service == "cesarean" else "surg_status"
        status = h[status_col].astype(str).to_numpy()
        eligibility_masks = {
            "A_affirmative_structure_only": status == "positive",
            "B_structure_positive_or_unknown": np.isin(
                status, ["positive", "unknown", "conflict_positive_and_zero"]
            ),
        }
        for set_name in set_names:
            eligible_idx = np.flatnonzero(eligibility_masks[set_name])
            observed_idx = eligible_idx[activity[service][eligible_idx] > 0]
            k = len(observed_idx)
            if k == 0:
                print(json.dumps({"status": "no_observed_active_hospitals_in_set", "service": service, "candidate_set": set_name}), flush=True)
                continue
            quotas = feasible_group_counts(facilities["groups"], observed_idx, eligible_idx)
            short_set = "A" if set_name.startswith("A_") else "B"
            observed_scenario = f"{service}_{short_set}_observed_subset"
            row, state_coverage = evaluate_or_reuse(
                binary, spatial, facilities, cnes, observed_idx, set_name,
                observed_scenario, "observed_activity_within_structure_subset",
                service, len(eligible_idx), k, m2_scenarios_path, m2_state_path,
                evaluation_cache,
            )
            if service == "chole" and set_name == "B_structure_positive_or_unknown":
                broad_chole_row = row
                broad_chole_state = state_coverage
                write_append(m1_scenarios_path, row)
                save_state_rows(
                    m1_state_path, set_name, observed_scenario, state_coverage,
                    spatial, str(row["selection_sha256"]),
                )
            print(json.dumps({"completed": observed_scenario, "all_30": row["all_30"], "reused": row["reused_from_scenario"]}), flush=True)

            for repetition in range(args.m2_repetitions):
                selected = construct_cluster(
                    eligible_idx, facilities["groups"], facilities["cell"], quotas, rng
                )
                if len(selected) != k:
                    raise AssertionError("Clustered construction did not preserve the exact service budget")
                scenario = f"{service}_{short_set}_state_joint_cluster_{repetition:02d}"
                row, _ = evaluate_or_reuse(
                    binary, spatial, facilities, cnes, selected, set_name,
                    scenario, "feasible_clustered_witness", service,
                    len(eligible_idx), k, m2_scenarios_path, m2_state_path,
                    evaluation_cache,
                )
                print(json.dumps({"completed": scenario, "all_30": row["all_30"], "reused": row["reused_from_scenario"]}), flush=True)

    if broad_chole_row is None or broad_chole_state is None:
        raise ValueError("Broad B chole activity reference is required for M1 and was not available")

    # M1: mask individual cholecystectomy facility labels while retaining total,
    # state and exact joint-cell counts. The reference is the activity-record
    # configuration restricted to each universe, never a capability gold label.
    m1_scenarios_path = result_dir / "m1_scenarios.csv"
    m1_state_path = result_dir / "m1_state_coverage.csv"
    for set_name in ["full_original", "B_structure_positive_or_unknown"]:
        eligible_idx = full_indices if set_name == "full_original" else np.flatnonzero(
            h["surg_status"].isin(["positive", "unknown", "conflict_positive_and_zero"]).to_numpy()
        )
        observed_idx = eligible_idx[np.flatnonzero(activity["chole"][eligible_idx] > 0)]
        quotas = feasible_group_counts(facilities["groups"], observed_idx, eligible_idx)
        priorities = [
            ("local_population_high_priority", construct_local_priority(
                eligible_idx, facilities["groups"], facilities["density"],
                cnes, quotas, highest_first=True,
            )),
            ("local_population_low_priority", construct_local_priority(
                eligible_idx, facilities["groups"], facilities["density"],
                cnes, quotas, highest_first=False,
            )),
        ]
        selections = list(priorities)
        for repetition in range(args.m1_random_repetitions):
            selections.append((
                f"random_{repetition:02d}",
                construct_random(eligible_idx, facilities["groups"], quotas, rng),
            ))
        for scenario, selected in selections:
            row, _ = evaluate_or_reuse(
                binary, spatial, facilities, cnes, selected, set_name,
                scenario, "masked_label_estimator", "chole",
                len(eligible_idx), len(observed_idx), m1_scenarios_path,
                m1_state_path, evaluation_cache,
            )
            print(json.dumps({"completed": f"M1:{set_name}:{scenario}", "all_30": row["all_30"], "reused": row["reused_from_scenario"]}), flush=True)

    # M2 bounds: each value is the range among the observed subset and the
    # enumerated feasible witnesses. It is a lower bound on the full conditional
    # width, not an optimized endpoint pair.
    prior = pd.read_csv(OLD_BOUNDS / "coverage_results.csv")
    bounds_rows: list[dict[str, object]] = []
    m2_frame = pd.read_csv(m2_scenarios_path)
    for service, (raw_field, flag_kind) in SERVICES.items():
        prior_service = prior.loc[prior["scenario"].str.startswith(f"{service}_")]
        if prior_service.empty:
            raise RuntimeError(f"Missing previous full-candidate bounds for {service}")
        for set_name in set_names:
            values = m2_frame.loc[
                m2_frame["service"].eq(service) & m2_frame["candidate_set"].eq(set_name)
            ]
            if values.empty:
                continue
            low30, high30 = float(values["all_30"].min()), float(values["all_30"].max())
            low120, high120 = float(values["all_120"].min()), float(values["all_120"].max())
            eligibility_row = eligibility.loc[
                eligibility["service"].eq(service) & eligibility["candidate_set"].eq(set_name)
            ].iloc[0]
            bounds_rows.append({
                "service": service,
                "structure_indicator": flag_kind,
                "candidate_set": set_name,
                "candidate_count": int(eligibility_row["candidate_count"]),
                "observed_active_hospitals_in_set": int(eligibility_row["active_hospitals_2024_in_set"]),
                "activity_records_2024_in_set": int(eligibility_row["activity_records_2024_in_set"]),
                "excluded_active_hospitals": int(eligibility_row["active_hospitals_excluded"]),
                "excluded_activity_records_2024": int(eligibility_row["activity_records_excluded"]),
                "witness_count_including_observed_subset": int(len(values)),
                "minimum_30": low30,
                "maximum_30": high30,
                "lower_bound_width_30_pp": (high30 - low30) * 100,
                "minimum_120": low120,
                "maximum_120": high120,
                "lower_bound_width_120_pp": (high120 - low120) * 100,
                "full_original_minimum_30": float(prior_service["all_30"].min()),
                "full_original_maximum_30": float(prior_service["all_30"].max()),
                "full_original_lower_bound_width_30_pp": float(prior_service["all_30"].max() - prior_service["all_30"].min()) * 100,
                "full_original_minimum_120": float(prior_service["all_120"].min()),
                "full_original_maximum_120": float(prior_service["all_120"].max()),
                "full_original_lower_bound_width_120_pp": float(prior_service["all_120"].max() - prior_service["all_120"].min()) * 100,
                "interpretation": "constructed feasible-set width lower bound conditional on CNES structural proxy, candidate universe, and recalculated state×joint quotas",
            })
    pd.DataFrame(bounds_rows).to_csv(result_dir / "m2_bounds.csv", index=False)

    # M1 scenarios are written incrementally. De-duplicate by named aggregate
    # scenario before calculating state-level error and rank summaries.
    state_file = pd.read_csv(m1_state_path).drop_duplicates(
        ["candidate_set", "scenario", "state_code"], keep="last"
    )
    scenario_file = pd.read_csv(m1_scenarios_path).drop_duplicates(
        ["candidate_set", "scenario"], keep="last"
    )
    summarize_masking(
        scenario_file, state_file,
        result_dir / "m1_scenario_metrics.csv",
        result_dir / "m1_state_rank_change.csv",
    )
    (result_dir / "run_manifest.json").write_text(json.dumps({
        **manifest,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "complete",
        "observed_reference_comparison": observed_reference_comparison,
        "selection_deduplication": "exact sorted-CNES-set SHA-256 cache; identical spatial selections reuse coverage across candidate sets, procedures and estimators",
        "result_files": sorted(p.name for p in result_dir.iterdir() if p.is_file()),
    }, indent=2), encoding="utf-8")
    print("CLINICAL_REANALYSIS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
