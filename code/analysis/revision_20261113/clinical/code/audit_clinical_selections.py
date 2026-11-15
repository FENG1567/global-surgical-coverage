#!/usr/bin/env python3
"""Recreate clinical allocations and verify saved ID hashes and quota cells.

This is a selection-only audit: it reads the shared aggregate inputs and
facility metadata, recreates the deterministic selections, and does not invoke
the travel-time engine or Dijkstra.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ANALYSIS_SCRIPT = HERE / "clinical_reanalysis.py"
spec = importlib.util.spec_from_file_location("clinical_reanalysis", ANALYSIS_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load analysis module: {ANALYSIS_SCRIPT}")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def get_expected_row(frame: pd.DataFrame, candidate_set: str, scenario: str) -> pd.Series:
    matches = frame.loc[
        frame["candidate_set"].eq(candidate_set) & frame["scenario"].eq(scenario)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one saved row for {candidate_set}:{scenario}; found {len(matches)}"
        )
    return matches.iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=analysis.BASE / "revision_20261113/clinical/results_reconciled_20261113",
    )
    args = parser.parse_args()
    result_dir = args.result_dir
    for name in [
        "selection_quota_audit.csv", "selection_group_quota_audit.csv",
        "selection_quota_audit.json", "m2_combined_screen_bounds.csv",
        "boundary120_recomputed.csv", "boundary120_recomputed_states.csv",
        "full_original_all9_bounds.csv", "boundary120_audit.json",
    ]:
        if (result_dir / name).exists():
            raise FileExistsError(f"Refusing to overwrite existing audit output: {result_dir / name}")
    manifest_path = result_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Selection audit requires a completed clinical run")

    m1_saved = pd.read_csv(result_dir / "m1_scenarios.csv")
    m2_saved = pd.read_csv(result_dir / "m2_scenarios.csv")
    eligibility_saved = pd.read_csv(result_dir / "eligibility_status.csv")
    h, counts = analysis.load_aggregate_inputs()
    metadata, raw_rows = analysis.read_cnes_metadata(set(h["cnes"]))
    h, _eligibility = analysis.source_eligibility(h, counts, metadata)
    spatial = analysis.load_spatial_inputs(
        int(manifest["spatial_inputs"]["crop_margin_degrees"])
    )
    facilities = analysis.facility_groups(h, spatial)
    cnes = h["cnes"].astype(str).to_numpy()
    groups = facilities["groups"]
    rng = np.random.default_rng(int(manifest["seed"]))
    m2_repetitions = int(manifest["m2"]["repetitions_per_eligible_set_and_procedure"])
    m1_repetitions = int(manifest["m1"]["random_repetitions_per_set"])

    scenario_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    combined_screen_rows: list[dict[str, object]] = []

    def audit_selection(
        phase: str,
        candidate_set: str,
        scenario: str,
        service: str,
        selection: np.ndarray,
        quotas: dict[int, int] | None,
        saved_frame: pd.DataFrame,
    ) -> None:
        selection = np.asarray(selection, dtype=int)
        saved = get_expected_row(saved_frame, candidate_set, scenario)
        selected_count = int(len(selection))
        unique_id_count = int(len(np.unique(selection)))
        unique_ids_match = unique_id_count == selected_count
        selection_hash = analysis.stable_selection_hash(cnes, selection)
        unique_source_pixels = int(
            len(np.unique(np.column_stack((facilities["rr"][selection], facilities["cc"][selection])), axis=0))
        )
        budget_matches = selected_count == int(saved["observed_positive_hospitals"])
        selection_hash_matches = selection_hash == str(saved["selection_sha256"])
        saved_count_matches = selected_count == int(saved["selected_hospitals"])
        saved_source_pixels_match = unique_source_pixels == int(saved["unique_source_pixels"])

        quota_mismatches = 0
        quota_cell_count = 0
        if quotas is not None:
            quota_map = {int(group): int(quota) for group, quota in quotas.items()}
            actual_counts = np.bincount(groups[selection], minlength=len(facilities["unique_groups"]))
            quota_cell_count = len(quota_map)
            for group in sorted(set(quota_map) | set(np.flatnonzero(actual_counts))):
                expected_count = quota_map.get(int(group), 0)
                selected_group_count = int(actual_counts[int(group)])
                matches = selected_group_count == expected_count
                quota_mismatches += int(not matches)
                group_label = facilities["unique_groups"][int(group)]
                state, high_local_population, smod_urban = group_label
                group_rows.append({
                    "phase": phase,
                    "candidate_set": candidate_set,
                    "scenario": scenario,
                    "service": service,
                    "group_state_code": str(state),
                    "high_local_population": int(high_local_population),
                    "smod_urban": int(smod_urban),
                    "expected_quota": expected_count,
                    "selected_count": selected_group_count,
                    "matches_quota": matches,
                })

        scenario_rows.append({
            "phase": phase,
            "candidate_set": candidate_set,
            "scenario": scenario,
            "service": service,
            "selected_count": selected_count,
            "saved_selected_count": int(saved["selected_hospitals"]),
            "observed_positive_budget": int(saved["observed_positive_hospitals"]),
            "unique_id_count": unique_id_count,
            "selection_sha256": selection_hash,
            "saved_selection_sha256": str(saved["selection_sha256"]),
            "selection_hash_matches": selection_hash_matches,
            "selected_count_matches_saved": saved_count_matches,
            "selected_count_matches_budget": budget_matches,
            "unique_ids_match_selected_count": unique_ids_match,
            "unique_source_pixels": unique_source_pixels,
            "saved_unique_source_pixels": int(saved["unique_source_pixels"]),
            "unique_source_pixels_match": saved_source_pixels_match,
            "quota_cell_count": quota_cell_count,
            "quota_mismatched_cells": quota_mismatches,
            "quota_counts_match": quota_mismatches == 0,
            "quota_check": "not_applicable" if quotas is None else "checked",
            "dijkstra_ran_for_saved_row": bool(saved["dijkstra_ran"]),
        })

    observed_full = np.flatnonzero(h["activity_chole_2024"].to_numpy(dtype=float) > 0)
    audit_selection(
        "M1", "full_original", "observed_activity_reference", "chole",
        observed_full, None, m1_saved,
    )

    activity = {
        service: h[f"activity_{service}_2024"].to_numpy(dtype=float)
        for service in analysis.SERVICES
    }
    set_names = ["A_affirmative_structure_only", "B_structure_positive_or_unknown"]

    # A and B are distinct rules but may resolve to identical facility IDs for
    # a given procedure. Report bounds over unique CNES-set hashes in that case
    # so different random draws are not mistaken for different eligibility sets.
    for service in analysis.SERVICES:
        rows = m2_saved.loc[m2_saved["service"].eq(service)].copy()
        unique_rows = rows.drop_duplicates("selection_sha256", keep="first")
        coverage_fields = [
            f"{scope}_{threshold}"
            for scope in ["all", "urban", "rural"]
            for threshold in [15, 30, 60, 120]
        ]
        for selection_hash, duplicates in rows.groupby("selection_sha256", sort=False):
            for field in coverage_fields:
                values = duplicates[field].astype(float).to_numpy()
                if not np.allclose(values, values[0], rtol=0.0, atol=1e-12):
                    raise AssertionError(
                        f"Duplicate selection hash has inconsistent {field}: {service}:{selection_hash}"
                    )
        a_idx = np.flatnonzero(
            h["surg_status"].eq("positive").to_numpy()
        ) if service != "cesarean" else np.flatnonzero(h["obst_status"].eq("positive").to_numpy())
        status_column = "obst_status" if service == "cesarean" else "surg_status"
        statuses = h[status_column].astype(str).to_numpy()
        a_eligible = statuses == "positive"
        b_eligible = np.isin(statuses, ["positive", "unknown", "conflict_positive_and_zero"])
        candidate_sets_identical = bool(np.array_equal(a_eligible, b_eligible))
        observed_a = a_idx[activity[service][a_idx] > 0]
        observed_b = np.flatnonzero(b_eligible & (activity[service] > 0))
        observed_sets_identical = bool(np.array_equal(observed_a, observed_b))
        service_eligibility = eligibility_saved.loc[
            eligibility_saved["service"].eq(service)
            & eligibility_saved["candidate_set"].isin(set_names)
        ].set_index("candidate_set")
        active_counts = np.array([
            int(service_eligibility.loc[name, "active_hospitals_2024_in_set"])
            for name in set_names
        ])
        candidate_counts = np.array([
            int(service_eligibility.loc[name, "candidate_count"])
            for name in set_names
        ])
        if len(active_counts) != 2 or len(candidate_counts) != 2:
            raise ValueError(f"Missing A/B eligibility rows for {service}")
        observed_rows = unique_rows.loc[
            unique_rows["scenario"].str.endswith("_observed_subset")
        ]
        if observed_rows.empty:
            raise ValueError(f"Missing observed M2 reference rows for {service}")
        combined_screen_rows.append({
            "service": service,
            "candidate_count_A": int(candidate_counts[0]),
            "candidate_count_B": int(candidate_counts[1]),
            "observed_active_hospitals_A": int(active_counts[0]),
            "observed_active_hospitals_B": int(active_counts[1]),
            "candidate_universe_identical_A_B": candidate_sets_identical,
            "observed_active_set_identical_A_B": observed_sets_identical,
            "unknown_flag_candidates": int(service_eligibility.loc[set_names[1], "flag_unknown_candidates"]),
            "conflicted_flag_candidates": int(service_eligibility.loc[set_names[1], "flag_conflicted_candidates"]),
            "raw_M2_rows_A_plus_B": int(len(rows)),
            "unique_selection_hashes": int(len(unique_rows)),
            "duplicate_rows_removed": int(len(rows) - len(unique_rows)),
            "unique_observed_reference_all_30": float(observed_rows.iloc[0]["all_30"]),
            "minimum_all_30": float(unique_rows["all_30"].min()),
            "maximum_all_30": float(unique_rows["all_30"].max()),
            "combined_unique_width_30_pp": float((unique_rows["all_30"].max() - unique_rows["all_30"].min()) * 100.0),
            "minimum_all_120": float(unique_rows["all_120"].min()),
            "maximum_all_120": float(unique_rows["all_120"].max()),
            "combined_unique_width_120_pp": float((unique_rows["all_120"].max() - unique_rows["all_120"].min()) * 100.0),
            "interpretation": "union of A/B scenario rows deduplicated by exact sorted-CNES-set SHA-256; candidate sets are compared separately in eligibility_status.csv",
        })
    for service in analysis.SERVICES:
        status_col = "obst_status" if service == "cesarean" else "surg_status"
        status = h[status_col].astype(str).to_numpy()
        eligibility_masks = {
            "A_affirmative_structure_only": status == "positive",
            "B_structure_positive_or_unknown": np.isin(
                status, ["positive", "unknown", "conflict_positive_and_zero"]
            ),
        }
        for candidate_set in set_names:
            eligible_idx = np.flatnonzero(eligibility_masks[candidate_set])
            observed_idx = eligible_idx[activity[service][eligible_idx] > 0]
            k = len(observed_idx)
            if k == 0:
                continue
            quotas = analysis.feasible_group_counts(groups, observed_idx, eligible_idx)
            short_set = "A" if candidate_set.startswith("A_") else "B"
            observed_scenario = f"{service}_{short_set}_observed_subset"
            audit_selection(
                "M2", candidate_set, observed_scenario, service, observed_idx,
                quotas, m2_saved,
            )
            for repetition in range(m2_repetitions):
                selected = analysis.construct_cluster(
                    eligible_idx, groups, facilities["cell"], quotas, rng
                )
                scenario = f"{service}_{short_set}_state_joint_cluster_{repetition:02d}"
                audit_selection(
                    "M2", candidate_set, scenario, service, selected, quotas, m2_saved
                )

    # The main run also writes the B cholecystectomy observed-subset row into
    # the M1 masking table, in addition to its M2 row.
    b_chole_idx = np.flatnonzero(
        h["surg_status"].isin(
            ["positive", "unknown", "conflict_positive_and_zero"]
        ).to_numpy()
    )
    b_chole_observed = b_chole_idx[activity["chole"][b_chole_idx] > 0]
    b_chole_quotas = analysis.feasible_group_counts(groups, b_chole_observed, b_chole_idx)
    audit_selection(
        "M1", "B_structure_positive_or_unknown", "chole_B_observed_subset",
        "chole", b_chole_observed, b_chole_quotas, m1_saved,
    )

    full_indices = np.arange(len(h), dtype=int)
    for candidate_set in ["full_original", "B_structure_positive_or_unknown"]:
        eligible_idx = (
            full_indices
            if candidate_set == "full_original"
            else np.flatnonzero(
                h["surg_status"].isin(
                    ["positive", "unknown", "conflict_positive_and_zero"]
                ).to_numpy()
            )
        )
        observed_idx = eligible_idx[activity["chole"][eligible_idx] > 0]
        quotas = analysis.feasible_group_counts(groups, observed_idx, eligible_idx)
        priorities = [
            ("local_population_high_priority", analysis.construct_local_priority(
                eligible_idx, groups, facilities["density"], cnes, quotas, highest_first=True
            )),
            ("local_population_low_priority", analysis.construct_local_priority(
                eligible_idx, groups, facilities["density"], cnes, quotas, highest_first=False
            )),
        ]
        selections = list(priorities)
        for repetition in range(m1_repetitions):
            selections.append((
                f"random_{repetition:02d}",
                analysis.construct_random(eligible_idx, groups, quotas, rng),
            ))
        for scenario, selected in selections:
            audit_selection("M1", candidate_set, scenario, "chole", selected, quotas, m1_saved)

    audit = pd.DataFrame(scenario_rows)
    group_audit = pd.DataFrame(group_rows)
    if len(audit) != len(m1_saved) + len(m2_saved):
        raise AssertionError(
            f"Audit recreated {len(audit)} scenarios; expected {len(m1_saved) + len(m2_saved)}"
        )
    failed_scenarios = audit.loc[
        ~audit["selection_hash_matches"]
        | ~audit["selected_count_matches_saved"]
        | ~audit["selected_count_matches_budget"]
        | ~audit["unique_ids_match_selected_count"]
        | ~audit["unique_source_pixels_match"]
        | ~audit["quota_counts_match"]
    ]
    if not group_audit.empty and not group_audit["matches_quota"].all():
        raise AssertionError("One or more recreated selections failed an exact group quota")

    audit.to_csv(result_dir / "selection_quota_audit.csv", index=False)
    group_audit.to_csv(result_dir / "selection_group_quota_audit.csv", index=False)
    combined_screen = pd.DataFrame(combined_screen_rows)
    combined_screen.to_csv(result_dir / "m2_combined_screen_bounds.csv", index=False)

    # Recheck only the original minimum-120 scenario for each service under
    # the expanded margin-4 crop; compare against the original full 9-scenario
    # results and use its observed scenario as the maximum reference.
    original_all9_dir = analysis.UPGRADE / "derived/brazil_state_constraints"
    original_all9 = pd.read_csv(original_all9_dir / "coverage_results.csv")
    binary = analysis.build_c_engine()
    cnes_index = {value: index for index, value in enumerate(cnes)}
    boundary_rows: list[dict[str, object]] = []
    boundary_results_path = result_dir / "boundary120_recomputed.csv"
    boundary_states_path = result_dir / "boundary120_recomputed_states.csv"
    for service in analysis.SERVICES:
        service_prior = original_all9.loc[original_all9["service"].eq(service)]
        if service_prior.empty:
            raise ValueError(f"Missing original all-nine scenarios for {service}")
        old_min = service_prior.loc[service_prior["all_120"].idxmin()]
        old_max = service_prior.loc[service_prior["all_120"].idxmax()]
        selection_path = original_all9_dir / f"{old_min['scenario']}_selection.csv"
        selected_ids = analysis.normalize_cnes(
            pd.read_csv(selection_path, dtype=str)["CLUES"]
        ).tolist()
        if len(selected_ids) != int(old_min["k"]) or len(set(selected_ids)) != len(selected_ids):
            raise ValueError(f"Invalid original minimum-120 selection file for {service}")
        missing_ids = sorted(set(selected_ids) - set(cnes_index))
        if missing_ids:
            raise ValueError(f"Original selected IDs are absent from the current facility universe for {service}")
        selected = np.array([cnes_index[value] for value in selected_ids], dtype=int)
        rerun_scenario = f"margin4_recheck_{service}_original_min120"
        selection_hash = analysis.stable_selection_hash(cnes, selected)
        rerun, _state_coverage = analysis.evaluate_selection(
            binary, spatial, facilities, selected, "full_original", rerun_scenario,
            "original_all9_minimum_120_boundary_recheck", service, len(h),
            len(selected), selection_hash, boundary_results_path, boundary_states_path,
        )
        old_min_120 = float(old_min["all_120"])
        old_observed_max_120 = float(old_max["all_120"])
        recomputed_min_120 = float(rerun["all_120"])
        difference_pp = (recomputed_min_120 - old_min_120) * 100.0
        boundary_rows.append({
            "service": service,
            "original_all9_minimum_scenario": str(old_min["scenario"]),
            "original_all9_minimum_120": old_min_120,
            "margin4_recomputed_minimum_scenario_120": rerun_scenario,
            "margin4_recomputed_minimum_120": recomputed_min_120,
            "recomputed_minus_original_minimum_120_pp": difference_pp,
            "original_all9_observed_maximum_scenario": str(old_max["scenario"]),
            "original_all9_observed_maximum_120": old_observed_max_120,
            "original_all9_width_120_pp": (old_observed_max_120 - old_min_120) * 100.0,
            "width_with_margin4_recomputed_minimum_pp": (old_observed_max_120 - recomputed_min_120) * 100.0,
            "minimum_reproduced_within_0_01_pp": bool(abs(difference_pp) <= 0.01),
            "selected_hospitals": int(len(selected)),
            "selection_sha256": selection_hash,
            "dijkstra_seconds": float(rerun["dijkstra_seconds"]),
            "selection_file": str(selection_path),
            "scope_note": "Only the original minimum-120 witness was rerun at margin 4; the other seven constructed scenarios were not rerun.",
        })
    boundary_frame = pd.DataFrame(boundary_rows)
    boundary_frame.to_csv(result_dir / "full_original_all9_bounds.csv", index=False)
    boundary_summary = {
        "status": "complete",
        "services_rechecked": int(len(boundary_frame)),
        "minimum_witnesses_reproduced_within_0_01_pp": int(
            boundary_frame["minimum_reproduced_within_0_01_pp"].sum()
        ),
        "dijkstra_scenarios_run": int(len(boundary_frame)),
        "sum_dijkstra_seconds": float(boundary_frame["dijkstra_seconds"].sum()),
        "scope_note": "One original minimum-120 witness per service was rerun at crop margin 4 and compared with the original all-nine output; all other original constructed witnesses were left untouched.",
    }
    (result_dir / "boundary120_audit.json").write_text(
        json.dumps({"summary": boundary_summary, "results": boundary_rows}, indent=2),
        encoding="utf-8",
    )
    audit_summary = {
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selection_regenerated_from_manifest_seed": int(manifest["seed"]),
        "dijkstra_run": False,
        "audit_scenario_count": int(len(audit)),
        "m1_scenario_count": int(len(m1_saved)),
        "m2_scenario_count": int(len(m2_saved)),
        "selection_hash_matches": int(audit["selection_hash_matches"].sum()),
        "selection_hash_failures": int((~audit["selection_hash_matches"]).sum()),
        "budget_and_saved_count_failures": int((~audit["selected_count_matches_saved"]).sum() + (~audit["selected_count_matches_budget"]).sum()),
        "duplicate_id_scenarios": int((~audit["unique_ids_match_selected_count"]).sum()),
        "unique_source_pixel_failures": int((~audit["unique_source_pixels_match"]).sum()),
        "group_quota_cells_checked": int(len(group_audit)),
        "group_quota_cell_failures": int((~group_audit["matches_quota"]).sum()) if not group_audit.empty else 0,
        "scenario_failures": int(len(failed_scenarios)),
        "cnes_source_rows_scanned": int(raw_rows),
        "id_handling": "facility CNES identifiers were used in-memory to recreate SHA-256 selection hashes; raw identifiers were not written to audit outputs",
    }
    (result_dir / "selection_quota_audit.json").write_text(
        json.dumps(audit_summary, indent=2), encoding="utf-8"
    )
    manifest["selection_quota_audit"] = audit_summary
    manifest["m2_combined_screen_bounds"] = combined_screen_rows
    manifest["boundary120_audit"] = boundary_summary
    manifest["result_files"] = sorted(path.name for path in result_dir.iterdir() if path.is_file())
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(audit_summary), flush=True)


if __name__ == "__main__":
    main()
