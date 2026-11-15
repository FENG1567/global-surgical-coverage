#!/usr/bin/env python3
"""Fit only the four flexible grouped-binomial models for the n>=1 cohort.

This bounded supplement joins its outputs to an already completed primary
analysis so standard models and other thresholds do not need to be refit.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HELPER_PATH = Path(__file__).with_name("reanalyse_prediction.py")
SPEC = importlib.util.spec_from_file_location("reanalyse_prediction", HELPER_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"could not load prediction helpers from {HELPER_PATH}")
MODEL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODEL)

THRESHOLD = 1
FLEXIBLE_MODELS = MODEL.FLEXIBLE_MODELS
KEYS = ["service", "threshold", "repartition", "m6", "fold"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit_service(
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    service: str,
    repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = MODEL.observed_outcome(features, outcomes, service, 2024)
    data = MODEL.add_blocks(
        data.loc[MODEL.spatial_valid_mask(data) & data["n"].ge(THRESHOLD)]
        .copy().sort_values("m6").reset_index(drop=True)
    )
    assignments = MODEL.repartitions(data["block_id"].to_numpy(), repetitions, seed)
    fold_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    score_rows: list[dict[str, Any]] = []

    for repetition in range(repetitions):
        oof = {model: np.full(len(data), np.nan, dtype=float) for model in FLEXIBLE_MODELS}
        prediction_count=np.zeros(len(data),dtype=np.int8)
        for fold in range(5):
            test_mask = assignments[repetition] == fold
            train = data.loc[~test_mask]
            test = data.loc[test_mask]
            prediction_count[test_mask]+=1
            if train["local"].sum() <= 0 or (train["n"] - train["local"]).sum() <= 0:
                raise ValueError(f"grouped-binomial training split lacks successes/failures: {service} rep={repetition} fold={fold}")
            for model in FLEXIBLE_MODELS:
                probability, diagnostics = MODEL.grouped_binomial_fit(train, test, model)
                if not np.isfinite(probability).all():
                    raise ValueError(f"non-finite flexible prediction: {service} rep={repetition} fold={fold} model={model}")
                oof[model][test_mask] = probability
                fold_rows.append({
                    "validation": "grouped_binomial_spatial",
                    "service": service,
                    "threshold": THRESHOLD,
                    "repartition": repetition,
                    "fold": fold,
                    "model": model,
                    "n_train_municipalities": len(train),
                    "n_test_municipalities": len(test),
                    "train_resident_AIHS": int(train["n"].sum()),
                    "train_local_AIHS": int(train["local"].sum()),
                    **diagnostics,
                })

        if not np.all(prediction_count==1) or any(not np.isfinite(values).all() for values in oof.values()):
            raise ValueError(f"incomplete flexible OOF predictions for {service} repartition={repetition}")
        for model in FLEXIBLE_MODELS:
            scored = MODEL.grouped_scores(data["local"].to_numpy(dtype=float), data["n"].to_numpy(dtype=float), oof[model])
            observed = int(data["local"].sum())
            total = int(data["n"].sum())
            score_rows.append({
                "service": service,
                "threshold": THRESHOLD,
                "repartition": repetition,
                "model": model,
                "municipalities": len(data),
                "resident_AIHS": total,
                "observed_local_AIHS": observed,
                "observed_local_rate": observed / total,
                "count_error": scored["predicted_local_records"] - observed,
                **scored,
            })
        pred = data[["m6", "n", "local", "block_id"]].copy()
        pred.insert(0, "service", service)
        pred.insert(1, "threshold", THRESHOLD)
        pred.insert(2, "repartition", repetition)
        pred["fold"] = assignments[repetition]
        pred["observed_rate"] = pred["local"] / pred["n"]
        for model in FLEXIBLE_MODELS:
            pred[f"p_{model}"] = oof[model]
            pred[f"expected_local_{model}"] = pred["n"] * oof[model]
        prediction_frames.append(pred)

    return pd.concat(prediction_frames, ignore_index=True), pd.DataFrame(score_rows), pd.DataFrame(fold_rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--coordinates", type=Path, required=True)
    parser.add_argument("--source-analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--denominator-repetitions", type=int, default=10)
    parser.add_argument("--bootstrap-replicates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261111)
    args = parser.parse_args()
    if args.denominator_repetitions < 1 or args.bootstrap_replicates < 20:
        parser.error("repetitions must be positive and bootstrap replicates at least 20")

    upgrade = args.project_root / "upgrade_20261112"
    outcome_path = upgrade / "inputs" / "brazil_municipality_annual.csv"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = args.source_analysis_dir
    feature_by_service, _, _ = MODEL.build_features(args.project_root, args.coordinates)
    outcomes = MODEL.load_outcomes(outcome_path)

    supplemental_predictions: list[pd.DataFrame] = []
    supplemental_scores: list[pd.DataFrame] = []
    supplemental_folds: list[pd.DataFrame] = []
    for service in MODEL.SERVICES:
        pred, score, fold = fit_service(
            feature_by_service[service], outcomes, service,
            args.denominator_repetitions, args.seed,
        )
        supplemental_predictions.append(pred)
        supplemental_scores.append(score)
        supplemental_folds.append(fold)
        print(f"FLEXIBLE_THRESHOLD1_COMPLETE {service} n={score['municipalities'].iloc[0]}", flush=True)

    supp_pred = pd.concat(supplemental_predictions, ignore_index=True)
    supp_scores = pd.concat(supplemental_scores, ignore_index=True)
    supp_folds = pd.concat(supplemental_folds, ignore_index=True)
    supp_pred.to_csv(output / "flexible_threshold1_oof_predictions.csv.gz", index=False)
    supp_scores.to_csv(output / "flexible_threshold1_scores_by_repartition.csv", index=False)
    supp_folds.to_csv(output / "flexible_threshold1_fold_diagnostics.csv", index=False)

    base_pred_path = source / "denominator_spatial_oof_predictions.csv.gz"
    base_scores_path = source / "denominator_spatial_scores_by_repartition.csv"
    base_folds_path = source / "denominator_glm_fold_diagnostics.csv"
    base_boot_path = source / "denominator_spatial_block_bootstrap.csv"
    base_pred = pd.read_csv(base_pred_path, dtype={"m6": "string"})
    supp_merge_columns = [
        f"p_{model}" for model in FLEXIBLE_MODELS
    ] + [f"expected_local_{model}" for model in FLEXIBLE_MODELS]
    joined = base_pred.merge(
        supp_pred[KEYS + supp_merge_columns],
        on=KEYS,
        how="left",
        validate="one_to_one",
        suffixes=("", "_supplement"),
    )
    n1 = joined["threshold"].eq(THRESHOLD)
    for column in supp_merge_columns:
        supplemental_column = f"{column}_supplement"
        if joined.loc[n1, supplemental_column].isna().any():
            raise ValueError(f"missing threshold-1 flexible predictions in merged OOF column {column}")
        joined.loc[n1, column] = joined.loc[n1, supplemental_column]
        joined = joined.drop(columns=[supplemental_column])
    if joined.loc[n1, [f"p_{model}" for model in FLEXIBLE_MODELS]].isna().any().any():
        raise ValueError("threshold-1 merged OOF table still has non-finite flexible predictions")

    base_scores = pd.read_csv(base_scores_path)
    merged_scores = pd.concat([base_scores, supp_scores], ignore_index=True)
    base_folds = pd.read_csv(base_folds_path)
    merged_folds = pd.concat([base_folds, supp_folds], ignore_index=True)
    rates = pd.read_csv(source / "observed_denominator_rates.csv").to_dict(orient="records")
    merged_absolute, _ = MODEL.summarize_spatial_scores(merged_scores, "denominator", rates)
    merged_increments = MODEL.summarize_increments(merged_scores, MODEL.COMPARISONS)
    old_bootstrap = pd.read_csv(base_boot_path)
    threshold1_bootstrap = MODEL.block_bootstrap_spatial_deltas(
        joined, THRESHOLD, MODEL.COMPARISONS, args.bootstrap_replicates, args.seed + 901
    )
    merged_bootstrap = pd.concat([
        old_bootstrap.loc[~old_bootstrap["eligibility_threshold_resident_AIHS"].eq(THRESHOLD)],
        threshold1_bootstrap,
    ], ignore_index=True)

    joined.to_csv(output / "denominator_spatial_oof_predictions_merged.csv.gz", index=False)
    merged_scores.to_csv(output / "denominator_spatial_scores_by_repartition_merged.csv", index=False)
    merged_folds.to_csv(output / "denominator_glm_fold_diagnostics_merged.csv", index=False)
    merged_absolute.to_csv(output / "denominator_spatial_absolute_performance_merged.csv", index=False)
    merged_increments.to_csv(output / "denominator_spatial_increment_summary_merged.csv", index=False)
    merged_bootstrap.to_csv(output / "denominator_spatial_block_bootstrap_merged.csv", index=False)

    manifest = {
        "objective": "supplement only the four flexible grouped-binomial models for the n>=1 2024 cohort and merge with completed primary outputs",
        "cohort": "spatial-valid municipalities with at least one resident procedure AIH record in 2024",
        "services": MODEL.SERVICES,
        "models_fit": FLEXIBLE_MODELS,
        "repartitions": args.denominator_repetitions,
        "folds_per_repartition": 5,
        "seed": args.seed,
        "merge_keys": KEYS,
        "source_analysis_dir": str(source),
        "outcome_path": str(outcome_path),
        "executables": {
            "supplement_script_sha256": sha256_file(Path(__file__).resolve()),
            "prediction_helper_sha256": sha256_file(HELPER_PATH.resolve()),
        },
        "source_files": {
            path.name: sha256_file(path)
            for path in [base_pred_path, base_scores_path, base_folds_path, base_boot_path]
        },
        "fit_counts": {
            "flexible_glm_fits": int(len(supp_folds)),
            "supplemental_prediction_rows": int(len(supp_pred)),
            "merged_prediction_rows": int(len(joined)),
            "merged_score_rows": int(len(merged_scores)),
            "merged_fold_diagnostic_rows": int(len(merged_folds)),
        },
        "diagnostic_summary": {
            "all_fits_converged": bool(supp_folds["converged"].astype(bool).all()),
            "rank_drops": int(pd.to_numeric(supp_folds["design_columns_dropped_by_qr"]).gt(0).sum()),
        },
        "uncertainty": "the threshold-1 block bootstrap is recomputed on merged nine-model OOF data; threshold-10 bootstrap is retained from the primary analysis",
    }
    manifest_path = output / "result_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "result_manifest.sha256").write_text(f"{sha256_file(manifest_path)}  {manifest_path.name}\n", encoding="utf-8")
    print("FLEXIBLE_THRESHOLD1_SUPPLEMENT_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
