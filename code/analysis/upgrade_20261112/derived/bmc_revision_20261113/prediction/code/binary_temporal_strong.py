#!/usr/bin/env python3
"""Binary 2023-to-2024 validation for the prespecified strong-baseline models.

The split is by source processing year. It is not an admission-calendar-year
holdout; use the raw-date audit when describing temporal interpretation.
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
from sklearn.metrics import log_loss, roc_auc_score


SCRIPT = Path(__file__).with_name("reanalyse_prediction.py")
SPEC = importlib.util.spec_from_file_location("reanalyse_prediction", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"could not load prediction analysis helpers from {SCRIPT}")
MODEL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODEL)


TEMPORAL_MODELS = [
    "population_urban",
    "population_urban_OSM",
    "population_urban_activity",
    "strong_baseline",
    "strong_baseline_plus_activity",
]
TEMPORAL_COMPARISONS = [
    ("population_urban_OSM", "population_urban"),
    ("population_urban_activity", "population_urban"),
    ("strong_baseline_plus_activity", "strong_baseline"),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weighted_binary_scores(y: np.ndarray, p: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    """Compute municipality-weighted scores for a block-bootstrap replicate."""
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    weights = np.asarray(weights, dtype=float)
    keep = weights > 0
    if not keep.any():
        raise ValueError("binary bootstrap replicate has no sampled municipalities")
    total = float(weights[keep].sum())
    auc = (
        float(roc_auc_score(y[keep], p[keep], sample_weight=weights[keep]))
        if np.unique(y[keep]).size == 2 else np.nan
    )
    brier = float(np.sum(weights[keep] * (y[keep] - p[keep]) ** 2) / total)
    loss = float(log_loss(y[keep], p[keep], sample_weight=weights[keep], labels=[0, 1]))
    return {
        "auc": auc,
        "brier": brier,
        "log_loss": loss,
        "predicted_positive_probability_sum": float(np.sum(weights[keep] * p[keep])),
    }


def run_service(
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    service: str,
    bootstrap_replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_all = MODEL.observed_outcome(features, outcomes, service, 2023)
    test_all = MODEL.observed_outcome(features, outcomes, service, 2024)
    train = MODEL.add_blocks(
        train_all.loc[MODEL.spatial_valid_mask(train_all) & train_all["n"].ge(10)]
        .copy().sort_values("m6").reset_index(drop=True)
    )
    test = MODEL.add_blocks(
        test_all.loc[MODEL.spatial_valid_mask(test_all) & test_all["n"].ge(10)]
        .copy().sort_values("m6").reset_index(drop=True)
    )
    for label, frame in [("training", train), ("test", test)]:
        if frame.empty or frame["local"].gt(0).nunique() != 2:
            raise ValueError(f"{service} binary temporal {label} set must contain both outcome classes")

    y_train = train["local"].gt(0).astype("int8").to_numpy()
    y_test = test["local"].gt(0).astype("int8").to_numpy()
    prediction = test[["m6", "n", "local", "block_id"]].copy()
    prediction.insert(0, "service", service)
    prediction.insert(1, "train_year", 2023)
    prediction.insert(2, "test_year", 2024)
    prediction["has_local"] = y_test
    prediction["municipality_seen_in_training"] = prediction["m6"].isin(train["m6"])
    prediction["observed_rate"] = test["local"].to_numpy(dtype=float) / test["n"].to_numpy(dtype=float)

    score_rows: list[dict[str, Any]] = []
    for model in TEMPORAL_MODELS:
        probability = MODEL.binary_fit(train.assign(has_local=y_train), test, model)
        prediction[f"p_{model}"] = probability
        metrics = MODEL.binary_scores(y_test, probability)
        score_rows.append({
            "service": service,
            "eligibility_threshold_resident_AIHS": 10,
            "model": model,
            "train_year": 2023,
            "test_year": 2024,
            "n_train_municipalities": len(train),
            "n_test_municipalities": len(test),
            "n_test_municipalities_not_seen_in_training": int((~prediction["municipality_seen_in_training"]).sum()),
            "train_positive_municipalities": int(y_train.sum()),
            "test_positive_municipalities": int(y_test.sum()),
            "test_positive_fraction": float(y_test.mean()),
            **metrics,
        })
    rng = np.random.default_rng(seed)
    blocks = test["block_id"].to_numpy()
    bootstrap: dict[tuple[str, str, str], list[float]] = {
        (left, right, metric): []
        for left, right in TEMPORAL_COMPARISONS
        for metric in ["auc", "brier", "log_loss", "predicted_positive_probability_sum"]
    }
    for _ in range(bootstrap_replicates):
        weights = MODEL.block_multiplicity(blocks, rng)
        score_map = {
            model: weighted_binary_scores(y_test, prediction[f"p_{model}"].to_numpy(dtype=float), weights)
            for model in TEMPORAL_MODELS
        }
        for left, right in TEMPORAL_COMPARISONS:
            for metric in ["auc", "brier", "log_loss", "predicted_positive_probability_sum"]:
                bootstrap[(left, right, metric)].append(score_map[left][metric] - score_map[right][metric])

    bootstrap_rows: list[dict[str, Any]] = []
    for (left, right, metric), values in bootstrap.items():
        arr = np.asarray(values, dtype=float)
        valid = arr[np.isfinite(arr)]
        bootstrap_rows.append({
            "service": service,
            "eligibility_threshold_resident_AIHS": 10,
            "comparison": f"{left} minus {right}",
            "metric": metric,
            "block_bootstrap_replicates": bootstrap_replicates,
            "valid_block_bootstrap_replicates": int(len(valid)),
            "block_bootstrap_p025": float(np.quantile(valid, 0.025)) if len(valid) else np.nan,
            "block_bootstrap_median": float(np.median(valid)) if len(valid) else np.nan,
            "block_bootstrap_p975": float(np.quantile(valid, 0.975)) if len(valid) else np.nan,
            "uncertainty_interpretation": (
                "paired percentile interval from resampling whole two-degree blocks in the 2024 test municipalities; "
                "conditional on this processing-year split, included municipalities, and model design"
            ),
        })

    return prediction, pd.DataFrame(score_rows), pd.DataFrame(bootstrap_rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--coordinates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--outcome-path", type=Path)
    parser.add_argument("--services", nargs="+", choices=MODEL.SERVICES, default=MODEL.SERVICES)
    parser.add_argument("--bootstrap-replicates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261111)
    args = parser.parse_args()
    if args.bootstrap_replicates < 20:
        parser.error("bootstrap replicates must be at least 20")

    upgrade = args.project_root / "upgrade_20261112"
    outcome_path = (args.outcome_path or upgrade / "inputs" / "brazil_municipality_annual.csv").resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    feature_by_service, _, sources = MODEL.build_features(args.project_root, args.coordinates)
    outcomes = MODEL.load_outcomes(outcome_path)
    predictions: list[pd.DataFrame] = []
    scores: list[pd.DataFrame] = []
    bootstraps: list[pd.DataFrame] = []
    denominator_predictions: list[pd.DataFrame] = []
    denominator_scores: list[pd.DataFrame] = []
    denominator_bootstraps: list[pd.DataFrame] = []
    for index, service in enumerate(args.services):
        pred, score, boot = run_service(
            feature_by_service[service], outcomes, service, args.bootstrap_replicates,
            args.seed + 100 * index,
        )
        predictions.append(pred)
        scores.append(score)
        bootstraps.append(boot)
        print(
            f"BINARY_TEMPORAL_COMPLETE {service} "
            f"n_train={int(score['n_train_municipalities'].iloc[0])} "
            f"n_test={int(score['n_test_municipalities'].iloc[0])}",
            flush=True,
        )
        for threshold in [1, 10]:
            den_pred, den_score, den_boot = MODEL.temporal_validation(
                feature_by_service[service], outcomes, service, threshold,
                args.bootstrap_replicates, args.seed + 500 + index * 20 + threshold,
            )
            denominator_predictions.append(den_pred)
            denominator_scores.append(den_score)
            denominator_bootstraps.append(den_boot)
            print(
                f"GROUPED_BINOMIAL_TEMPORAL_COMPLETE {service} threshold={threshold} "
                f"n={len(den_pred)}",
                flush=True,
            )

    pd.concat(predictions, ignore_index=True).to_csv(output / "binary_temporal_predictions.csv.gz", index=False)
    pd.concat(scores, ignore_index=True).to_csv(output / "binary_temporal_absolute_performance.csv", index=False)
    pd.concat(bootstraps, ignore_index=True).to_csv(output / "binary_temporal_block_bootstrap.csv", index=False)
    pd.concat(denominator_predictions, ignore_index=True).to_csv(output / "denominator_temporal_predictions.csv.gz", index=False)
    pd.concat(denominator_scores, ignore_index=True).to_csv(output / "denominator_temporal_absolute_performance.csv", index=False)
    pd.concat(denominator_bootstraps, ignore_index=True).to_csv(output / "denominator_temporal_block_bootstrap.csv", index=False)
    design = {
        "objective": "predict any local resident procedure AIH among municipalities with at least 10 resident procedure AIHs",
        "temporal_split": "train on source processing year 2023 and test on source processing year 2024; this is not an admission-calendar-year holdout",
        "outcome_definition": "one when local resident AIH count is greater than zero; municipality-year outcome aggregated from repeated AIH records",
        "models": TEMPORAL_MODELS,
        "comparisons": [f"{left} minus {right}" for left, right in TEMPORAL_COMPARISONS],
        "metrics": "absolute municipal AUC, Brier score, log loss, and predicted positive probability sum",
        "grouped_binomial_models": MODEL.STANDARD_MODELS,
        "grouped_binomial_thresholds": [1, 10],
        "uncertainty": "paired percentile intervals resampling whole two-degree blocks in the 2024 test set; these are conditional intervals, not confidence intervals over records or all municipalities",
        "features": {
            "population_urban": "log1p(GHSL2025 municipality population) and urban population fraction",
            "population_urban_OSM": "population_urban plus 30-minute OSM facility access",
            "population_urban_activity": "population_urban plus 2017-2019 service-active facility access",
            "strong_baseline": "population_urban plus prior local treatment presence and log1p prior volume",
            "strong_baseline_plus_activity": "strong_baseline plus 2017-2019 service-active facility access",
        },
        "outcome_path": str(outcome_path),
        "feature_sources": sources.to_dict(orient="records"),
        "executables": {
            "temporal_extension_sha256": sha256_file(Path(__file__).resolve()),
            "prediction_helper_sha256": sha256_file(SCRIPT.resolve()),
        },
        "outcome_input_sha256": sha256_file(outcome_path),
        "n_bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
    }
    design_path = output / "analysis_design.json"
    design_path.write_text(json.dumps(design, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "analysis_design.sha256").write_text(f"{sha256_file(design_path)}  {design_path.name}\n", encoding="utf-8")
    print("BINARY_TEMPORAL_ANALYSIS_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
