#!/usr/bin/env python3
"""Grouped-binomial and binary spatial/temporal prediction reanalysis.

The unit in the denominator analysis is a municipality with a count of
resident procedure AIH records, represented as grouped successes/failures.
AIHs are records, not unique patients. Probability scores are record-weighted;
uncertainty resamples whole two-degree geographic blocks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import patsy
import scipy
import sklearn
import statsmodels.api as sm
import warnings
from scipy.linalg import qr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import SplineTransformer
from statsmodels.tools.sm_exceptions import ConvergenceWarning as SMConvergenceWarning
from statsmodels.tools.sm_exceptions import SingularMatrixWarning


SERVICES = ["chole", "cesarean", "appendectomy"]
THRESHOLDS = [1, 5, 10, 20]
BLOCK_BOOTSTRAPS = 200
BASE_FEATURES = ["logpop", "urban_share"]

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "population_urban": {"columns": ["logpop", "urban_share"], "spline": False},
    "population_urban_OSM": {"columns": ["logpop", "urban_share", "D1_osm"], "spline": False},
    "population_urban_activity": {"columns": ["logpop", "urban_share", "D2_act_pre"], "spline": False},
    "strong_baseline": {"columns": ["logpop", "urban_share", "prior_local_service", "log_prior_treatment_count"], "spline": False},
    "strong_baseline_plus_activity": {"columns": ["logpop", "urban_share", "prior_local_service", "log_prior_treatment_count", "D2_act_pre"], "spline": False},
    "flexible_population_urban": {"columns": ["logpop", "urban_share"], "spline": True},
    "flexible_population_urban_plus_activity": {"columns": ["logpop", "urban_share", "D2_act_pre"], "spline": True},
    "flexible_strong_baseline": {"columns": ["logpop", "urban_share", "prior_local_service", "log_prior_treatment_count"], "spline": True},
    "flexible_strong_baseline_plus_activity": {"columns": ["logpop", "urban_share", "prior_local_service", "log_prior_treatment_count", "D2_act_pre"], "spline": True},
}

STANDARD_MODELS = [
    "population_urban",
    "population_urban_OSM",
    "population_urban_activity",
    "strong_baseline",
    "strong_baseline_plus_activity",
]
FLEXIBLE_MODELS = [
    "flexible_population_urban",
    "flexible_population_urban_plus_activity",
    "flexible_strong_baseline",
    "flexible_strong_baseline_plus_activity",
]
COMPARISONS = [
    ("population_urban_activity", "population_urban"),
    ("population_urban_OSM", "population_urban"),
    ("strong_baseline_plus_activity", "strong_baseline"),
    ("flexible_population_urban_plus_activity", "flexible_population_urban"),
    ("flexible_strong_baseline_plus_activity", "flexible_strong_baseline"),
]


def normalize_code(series: pd.Series, width: int | None = None) -> pd.Series:
    out = series.astype("string").str.strip().str.replace(r"\.0+$", "", regex=True)
    out = out.mask(out.eq(""), pd.NA)
    return out.str.zfill(width) if width else out


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_manifest(paths: list[Path]) -> list[dict[str, Any]]:
    return [{
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    } for path in paths]


def build_features(project_root: Path, coordinates_path: Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    upgrade = project_root / "upgrade_20261112"
    pop_path = upgrade / "derived" / "brazil_spatial_v1" / "municipality_population.csv"
    candidate_path = upgrade / "derived" / "brazil_spatial_v1" / "candidate_facilities_with_local_pop.csv"
    population = pd.read_csv(pop_path, dtype={"m6": "string"})
    population["m6"] = normalize_code(population["m6"], 6)
    if population["m6"].duplicated().any():
        raise ValueError("municipality_population.csv must have one row per m6")
    coordinates = pd.read_csv(coordinates_path, dtype={"codigo_ibge": "string"})
    coordinates["m6"] = normalize_code(coordinates["codigo_ibge"], 7).str[:6]
    coordinates = coordinates[["m6", "latitude", "longitude"]].drop_duplicates("m6")
    if coordinates["m6"].duplicated().any():
        raise ValueError("municipality coordinates are not unique by m6 after de-duplication")
    candidate = pd.read_csv(candidate_path, dtype={"CLUES": "string", "municipio6": "string"})
    candidate["CLUES"] = normalize_code(candidate["CLUES"], 7)
    candidate["municipio6"] = normalize_code(candidate["municipio6"], 6)
    if candidate["CLUES"].duplicated().any():
        raise ValueError("candidate facility CNES identifiers are not unique")

    region_codes = {
        **dict.fromkeys([11, 12, 13, 14, 15, 16, 17], "North"),
        **dict.fromkeys([21, 22, 23, 24, 25, 26, 27, 28, 29], "Northeast"),
        **dict.fromkeys([31, 32, 33, 35], "Southeast"),
        **dict.fromkeys([41, 42, 43], "South"),
        **dict.fromkeys([50, 51, 52, 53], "Midwest"),
    }

    result: dict[str, pd.DataFrame] = {}
    for service in SERVICES:
        exposure_path = upgrade / "derived" / "brazil_spatial_v1" / f"{service}_prior_municipality.csv"
        exposure = pd.read_csv(exposure_path, dtype={"m6": "string"})
        exposure["m6"] = normalize_code(exposure["m6"], 6)
        exposure = exposure[["m6", "coverage30"]].rename(columns={"coverage30": "D2_act_pre"})
        if exposure["m6"].duplicated().any():
            raise ValueError(f"{service} historical exposure must be unique by m6")
        prior_column = f"n_{service}_pre"
        prior = candidate.groupby("municipio6", dropna=False)[prior_column].sum(min_count=1).rename("prior_treatment_count").reset_index().rename(columns={"municipio6": "m6"})
        features = population.merge(coordinates, on="m6", how="left", validate="one_to_one")
        features = features.merge(exposure, on="m6", how="left", validate="one_to_one")
        features = features.merge(prior, on="m6", how="left", validate="one_to_one")
        features["prior_treatment_count"] = features["prior_treatment_count"].fillna(0)
        features["prior_local_service"] = features["prior_treatment_count"].gt(0).astype("int8")
        features["log_prior_treatment_count"] = np.log1p(features["prior_treatment_count"].astype(float))
        features["logpop"] = np.log1p(pd.to_numeric(features["pop"], errors="coerce"))
        features["urban_share"] = pd.to_numeric(features["urban_population"], errors="coerce") / pd.to_numeric(features["pop"], errors="coerce")
        state_prefix = pd.to_numeric(features["m6"].str[:2], errors="coerce")
        features["region"] = state_prefix.map(region_codes)
        result[service] = features
    sources = pd.DataFrame({
        "source_role": ["municipality_population", "candidate_facilities", "coordinates"] + [f"{s}_historical_exposure" for s in SERVICES],
        "path": [str(pop_path), str(candidate_path), str(coordinates_path)] + [str(upgrade / "derived" / "brazil_spatial_v1" / f"{s}_prior_municipality.csv") for s in SERVICES],
    })
    return result, candidate, sources


def load_outcomes(path: Path) -> pd.DataFrame:
    outcomes = pd.read_csv(path, dtype={"MUNIC_RES": "string"})
    outcomes["m6"] = normalize_code(outcomes["MUNIC_RES"], 6)
    outcomes["year"] = pd.to_numeric(outcomes["year"], errors="coerce").astype(int)
    outcomes["n"] = pd.to_numeric(outcomes["n"], errors="coerce")
    outcomes["local"] = pd.to_numeric(outcomes["local"], errors="coerce")
    if (outcomes["local"].dropna() < 0).any() or (outcomes["local"] > outcomes["n"]).any():
        raise ValueError("local counts must be between zero and resident AIH counts")
    if outcomes.duplicated(["year", "service", "m6"]).any():
        raise ValueError("annual municipality outcome table has duplicate year/service/m6 keys")
    return outcomes


def observed_outcome(features: pd.DataFrame, outcomes: pd.DataFrame, service: str, year: int) -> pd.DataFrame:
    outcome = outcomes.loc[outcomes["service"].eq(service) & outcomes["year"].eq(year), ["m6", "n", "local"]]
    data = features.merge(outcome, on="m6", how="left", validate="one_to_one")
    observed = data["n"].notna() & data["n"].gt(0)
    data.loc[~observed, ["n", "local"]] = np.nan
    data["outcome_observed"] = observed
    return data


def required_spatial_columns() -> list[str]:
    return ["pop", "urban_population", "D1_osm", "D2_act_pre", "latitude", "longitude", "logpop", "urban_share"]


def spatial_valid_mask(data: pd.DataFrame) -> pd.Series:
    return data[required_spatial_columns()].notna().all(axis=1) & data["pop"].gt(0) & data["urban_share"].between(0, 1)


def add_blocks(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out["block_lat"] = np.floor(out["latitude"].astype(float) / 2).astype(int)
    out["block_lon"] = np.floor(out["longitude"].astype(float) / 2).astype(int)
    out["block_id"] = out["block_lat"] * 1000 + out["block_lon"]
    return out


def repartitions(blocks: np.ndarray, repetitions: int, seed: int) -> np.ndarray:
    unique = np.unique(blocks)
    rng = np.random.default_rng(seed)
    assignments = np.empty((repetitions, len(blocks)), dtype=np.int8)
    for rep in range(repetitions):
        permutation = rng.permutation(unique)
        mapping = {block: index % 5 for index, block in enumerate(permutation)}
        assignments[rep] = np.asarray([mapping[b] for b in blocks], dtype=np.int8)
    return assignments


def design_matrices(train: pd.DataFrame, test: pd.DataFrame, spec: dict[str, Any], intercept: bool) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    columns = spec["columns"]
    if spec["spline"]:
        spline = SplineTransformer(n_knots=4, degree=3, knots="quantile", include_bias=False)
        train_base = spline.fit_transform(train[BASE_FEATURES].astype(float))
        test_base = spline.transform(test[BASE_FEATURES].astype(float))
        extras = [column for column in columns if column not in BASE_FEATURES]
        if extras:
            train_extra = train[extras].astype(float).to_numpy()
            test_extra = test[extras].astype(float).to_numpy()
            train_x = np.column_stack([train_base, train_extra])
            test_x = np.column_stack([test_base, test_extra])
        else:
            train_x, test_x = train_base, test_base
    else:
        train_x = train[columns].astype(float).to_numpy()
        test_x = test[columns].astype(float).to_numpy()
    if intercept:
        train_x = np.column_stack([np.ones(len(train_x)), train_x])
        test_x = np.column_stack([np.ones(len(test_x)), test_x])
    if not (np.isfinite(train_x).all() and np.isfinite(test_x).all()):
        raise ValueError("non-finite predictor values reached model fitting")
    # Drop any exactly aliased training columns before fitting. Pivoted QR also
    # protects held-out predictions from unidentified spline coefficients.
    _, triangular, pivots = qr(train_x, mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(triangular))
    scale = float(diagonal.max()) if len(diagonal) else 0.0
    tolerance = max(train_x.shape) * np.finfo(float).eps * scale
    rank = int(np.count_nonzero(diagonal > tolerance))
    kept = np.sort(pivots[:rank])
    info = {
        "design_columns_before_qr": int(train_x.shape[1]),
        "design_rank_after_qr": rank,
        "design_columns_dropped_by_qr": int(train_x.shape[1] - rank),
    }
    return train_x[:, kept], test_x[:, kept], info


def grouped_binomial_fit(train: pd.DataFrame, test: pd.DataFrame, model: str) -> tuple[np.ndarray, dict[str, Any]]:
    spec = MODEL_SPECS[model]
    x_train, x_test, design_info = design_matrices(train, test, spec, intercept=True)
    endog = np.column_stack([train["local"].to_numpy(dtype=float), (train["n"] - train["local"]).to_numpy(dtype=float)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", SingularMatrixWarning)
        warnings.simplefilter("error", SMConvergenceWarning)
        fit = sm.GLM(endog, x_train, family=sm.families.Binomial()).fit(maxiter=100, tol=1e-8, scale="X2")
    singular_warnings = [str(item.message) for item in caught if issubclass(item.category, SingularMatrixWarning)]
    if singular_warnings:
        raise RuntimeError(f"rank warning after QR filtering for {model}: {singular_warnings[0]}")
    if not fit.converged:
        raise RuntimeError(f"grouped-binomial model did not converge: {model}")
    probability = np.asarray(fit.predict(x_test), dtype=float)
    return np.clip(probability, 1e-8, 1 - 1e-8), {
        "overdispersion_pearson_scale": float(fit.scale),
        "pearson_chi2": float(fit.pearson_chi2),
        "residual_df": float(fit.df_resid),
        "converged": bool(fit.converged),
        "n_parameters": int(len(fit.params)),
        **design_info,
    }


def binary_fit(train: pd.DataFrame, test: pd.DataFrame, model: str) -> np.ndarray:
    spec = MODEL_SPECS[model]
    x_train, x_test, _ = design_matrices(train, test, spec, intercept=False)
    fit = LogisticRegression(max_iter=2000)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        fit.fit(x_train, train["has_local"].to_numpy(dtype=int))
    return np.clip(fit.predict_proba(x_test)[:, 1], 1e-8, 1 - 1e-8)


def grouped_scores(local: np.ndarray, n: np.ndarray, probability: np.ndarray, block_weights: np.ndarray | None = None) -> dict[str, float]:
    local = np.asarray(local, dtype=float)
    n = np.asarray(n, dtype=float)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-12, 1 - 1e-12)
    weights = np.ones_like(n) if block_weights is None else np.asarray(block_weights, dtype=float)
    successful = local * weights
    unsuccessful = (n - local) * weights
    total = float((n * weights).sum())
    if total <= 0:
        return {"auc": np.nan, "brier": np.nan, "log_loss": np.nan, "observed_local_records": np.nan, "predicted_local_records": np.nan}
    y = np.concatenate([np.ones(len(n), dtype=np.int8), np.zeros(len(n), dtype=np.int8)])
    score = np.concatenate([probability, probability])
    sample_weight = np.concatenate([successful, unsuccessful])
    keep = sample_weight > 0
    auc = float(roc_auc_score(y[keep], score[keep], sample_weight=sample_weight[keep])) if len(np.unique(y[keep])) == 2 else np.nan
    brier = float(((successful * (1 - probability) ** 2 + unsuccessful * probability ** 2).sum()) / total)
    loss = float((-(successful * np.log(probability) + unsuccessful * np.log1p(-probability)).sum()) / total)
    return {
        "auc": auc,
        "brier": brier,
        "log_loss": loss,
        "observed_local_records": float((local * weights).sum()),
        "predicted_local_records": float((n * probability * weights).sum()),
    }


def binary_scores(has_local: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    y = np.asarray(has_local, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-12, 1 - 1e-12)
    return {
        "auc": float(roc_auc_score(y, p)),
        "brier": float(np.mean((y - p) ** 2)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "observed_positive_municipalities": int(y.sum()),
        "predicted_positive_probability_sum": float(p.sum()),
    }


def block_multiplicity(block_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    unique = np.unique(block_ids)
    draw = rng.integers(0, len(unique), size=len(unique))
    counts = np.bincount(draw, minlength=len(unique))
    position = np.searchsorted(unique, block_ids)
    return counts[position].astype(float)


def grouped_rate_interval(data: pd.DataFrame, replicates: int, seed: int) -> tuple[float, float]:
    local = data["local"].to_numpy(dtype=float)
    n = data["n"].to_numpy(dtype=float)
    blocks = data["block_id"].to_numpy()
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=float)
    for index in range(replicates):
        weights = block_multiplicity(blocks, rng)
        samples[index] = float((local * weights).sum() / (n * weights).sum())
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def summarize_spatial_scores(scores: pd.DataFrame, threshold_label: str, rate_rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    increment_rows: list[dict[str, Any]] = []
    group_cols = ["service", "threshold", "model"]
    metrics = ["auc", "brier", "log_loss", "predicted_local_records", "count_error"]
    for key, group in scores.groupby(group_cols, sort=True):
        service, threshold, model = key
        row: dict[str, Any] = {
            "service": service,
            "eligibility_threshold_resident_AIHS": int(threshold),
            "model": model,
            "municipalities": int(group["municipalities"].iloc[0]),
            "resident_AIHS": int(group["resident_AIHS"].iloc[0]),
            "observed_local_AIHS": int(group["observed_local_AIHS"].iloc[0]),
            "observed_local_rate": float(group["observed_local_rate"].iloc[0]),
            "n_repartitions": int(group["repartition"].nunique()),
        }
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.nanmean(values))
            row[f"{metric}_repartition_p025"] = float(np.nanquantile(values, 0.025))
            row[f"{metric}_repartition_p975"] = float(np.nanquantile(values, 0.975))
        rate_key = next((x for x in rate_rows if x["service"] == service and x["threshold"] == int(threshold)), None)
        if rate_key:
            row["observed_local_rate_block_bootstrap_p025"] = rate_key["rate_block_bootstrap_p025"]
            row["observed_local_rate_block_bootstrap_p975"] = rate_key["rate_block_bootstrap_p975"]
        summary_rows.append(row)

    return pd.DataFrame(summary_rows), pd.DataFrame(increment_rows)


def summarize_increments(scores: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_fields = ["service", "threshold"]
    for (service, threshold), group in scores.groupby(group_fields, sort=True):
        wide = group.pivot(index="repartition", columns="model", values=["auc", "brier", "log_loss", "predicted_local_records"])
        for extension, baseline in pairs:
            if extension not in wide["auc"].columns or baseline not in wide["auc"].columns:
                continue
            for metric in ["auc", "brier", "log_loss", "predicted_local_records"]:
                delta = (wide[(metric, extension)] - wide[(metric, baseline)]).dropna().to_numpy(dtype=float)
                if len(delta) == 0:
                    continue
                rows.append({
                    "service": service,
                    "eligibility_threshold_resident_AIHS": int(threshold),
                    "comparison": f"{extension} minus {baseline}",
                    "metric": metric,
                    "mean_delta": float(delta.mean()),
                    "repartition_p025": float(np.quantile(delta, 0.025)),
                    "repartition_p975": float(np.quantile(delta, 0.975)),
                    "n_repartitions": int(len(delta)),
                    "n_positive_deltas": int((delta > 0).sum()),
                    "n_negative_deltas": int((delta < 0).sum()),
                    "uncertainty_interpretation": "variation over held-out block repartitions; not a population confidence interval",
                })
    return pd.DataFrame(rows)


def block_bootstrap_spatial_deltas(predictions: pd.DataFrame, threshold: int, pairs: list[tuple[str, str]], replicates: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for service, service_frame in predictions.loc[predictions["threshold"].eq(threshold)].groupby("service", sort=True):
        by_rep = {rep: block for rep, block in service_frame.groupby("repartition", sort=True)}
        reps = sorted(by_rep)
        expected_models = set(STANDARD_MODELS + (FLEXIBLE_MODELS if threshold in [1, 10] else []))
        available_columns = {column[2:] for column in service_frame.columns if column.startswith("p_")}
        missing_columns = expected_models - available_columns
        if missing_columns:
            raise ValueError(f"missing expected bootstrap prediction columns for {service} threshold={threshold}: {sorted(missing_columns)}")
        nonfinite_models = sorted(model for model in expected_models if not service_frame[f"p_{model}"].notna().all())
        if nonfinite_models:
            raise ValueError(f"non-finite required bootstrap predictions for {service} threshold={threshold}: {nonfinite_models}")
        # Concatenating thresholds creates structurally all-NaN flexible model
        # columns at thresholds where those models were not fit. Only models
        # prespecified for this threshold enter paired bootstrap scoring.
        available_models = expected_models
        active_pairs = [(left, right) for left, right in pairs if left in available_models and right in available_models]
        models = sorted({left for left, _ in active_pairs} | {right for _, right in active_pairs})
        block_ids = by_rep[reps[0]]["block_id"].to_numpy()
        local = by_rep[reps[0]]["local"].to_numpy(dtype=float)
        n = by_rep[reps[0]]["n"].to_numpy(dtype=float)
        bootstrap_deltas: dict[tuple[str, str, str], list[float]] = {}
        for left, right in active_pairs:
            for metric in ["auc", "brier", "log_loss", "predicted_local_records"]:
                bootstrap_deltas[(left, right, metric)] = []
        for _ in range(replicates):
            weights = block_multiplicity(block_ids, rng)
            per_pair: dict[tuple[str, str, str], list[float]] = {key: [] for key in bootstrap_deltas}
            for rep in reps:
                frame = by_rep[rep]
                score_map: dict[str, dict[str, float]] = {}
                for model in models:
                    score_map[model] = grouped_scores(local, n, frame[f"p_{model}"].to_numpy(dtype=float), weights)
                for left, right in active_pairs:
                    for metric in ["auc", "brier", "log_loss", "predicted_local_records"]:
                        per_pair[(left, right, metric)].append(score_map[left][metric] - score_map[right][metric])
            for key, values in per_pair.items():
                bootstrap_deltas[key].append(float(np.nanmean(values)))
        for (left, right, metric), values in bootstrap_deltas.items():
            arr = np.asarray(values, dtype=float)
            rows.append({
                "service": service,
                "eligibility_threshold_resident_AIHS": threshold,
                "comparison": f"{left} minus {right}",
                "metric": metric,
                "block_bootstrap_replicates": replicates,
                "block_bootstrap_p025": float(np.nanquantile(arr, 0.025)),
                "block_bootstrap_median": float(np.nanmedian(arr)),
                "block_bootstrap_p975": float(np.nanquantile(arr, 0.975)),
                "uncertainty_interpretation": "paired percentile interval from resampling whole 2-degree blocks; conditional on observed municipalities and model design",
            })
    return pd.DataFrame(rows)


def binary_cv(data: pd.DataFrame, service: str, repetitions: int, seed: int, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = add_blocks(data.loc[spatial_valid_mask(data) & data["n"].ge(10)].copy().sort_values("m6").reset_index(drop=True))
    d["has_local"] = d["local"].gt(0).astype("int8")
    models = STANDARD_MODELS + FLEXIBLE_MODELS
    assignments = repartitions(d["block_id"].to_numpy(), repetitions, seed)
    prediction_frames: list[pd.DataFrame] = []
    score_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for rep in range(repetitions):
        oof = {model: np.full(len(d), np.nan, dtype=float) for model in models}
        for fold in range(5):
            test_mask = assignments[rep] == fold
            train = d.loc[~test_mask]
            test = d.loc[test_mask]
            if train["has_local"].nunique() < 2 or test["has_local"].nunique() < 2:
                raise ValueError(f"binary split lacks both outcome classes: {service} rep={rep} fold={fold}")
            for model in models:
                oof[model][test_mask] = binary_fit(train, test, model)
                fold_rows.append({
                    "validation": "binary_any_local_spatial",
                    "service": service,
                    "threshold": 10,
                    "repartition": rep,
                    "fold": fold,
                    "model": model,
                    "n_train_municipalities": len(train),
                    "n_test_municipalities": len(test),
                })
        if any(not np.isfinite(p).all() for p in oof.values()):
            raise ValueError("binary out-of-fold prediction missing")
        for model in models:
            scored = binary_scores(d["has_local"].to_numpy(dtype=int), oof[model])
            score_rows.append({
                "service": service,
                "threshold": 10,
                "repartition": rep,
                "model": model,
                "municipalities": len(d),
                "positive_municipalities": int(d["has_local"].sum()),
                "positive_fraction": float(d["has_local"].mean()),
                **scored,
            })
        pred = d[["m6", "n", "local", "has_local", "block_id"]].copy()
        pred.insert(0, "service", service)
        pred.insert(1, "repartition", rep)
        pred["fold"] = assignments[rep]
        for model in models:
            pred[f"p_{model}"] = oof[model]
        prediction_frames.append(pred)
    return pd.concat(prediction_frames, ignore_index=True), pd.DataFrame(score_rows), pd.DataFrame(fold_rows)


def denominator_cv(data: pd.DataFrame, service: str, threshold: int, repetitions: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = add_blocks(data.loc[spatial_valid_mask(data) & data["n"].ge(threshold)].copy().sort_values("m6").reset_index(drop=True))
    models = STANDARD_MODELS + (FLEXIBLE_MODELS if threshold in [1, 10] else [])
    assignments = repartitions(d["block_id"].to_numpy(), repetitions, seed)
    prediction_frames: list[pd.DataFrame] = []
    score_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for rep in range(repetitions):
        oof = {model: np.full(len(d), np.nan, dtype=float) for model in models}
        for fold in range(5):
            test_mask = assignments[rep] == fold
            train = d.loc[~test_mask]
            test = d.loc[test_mask]
            if train["local"].sum() <= 0 or (train["n"] - train["local"]).sum() <= 0:
                raise ValueError(f"grouped-binomial training split lacks successes/failures: {service} threshold={threshold} rep={rep} fold={fold}")
            for model in models:
                probability, diagnostics = grouped_binomial_fit(train, test, model)
                oof[model][test_mask] = probability
                fold_rows.append({
                    "validation": "grouped_binomial_spatial",
                    "service": service,
                    "threshold": threshold,
                    "repartition": rep,
                    "fold": fold,
                    "model": model,
                    "n_train_municipalities": len(train),
                    "n_test_municipalities": len(test),
                    "train_resident_AIHS": int(train["n"].sum()),
                    "train_local_AIHS": int(train["local"].sum()),
                    **diagnostics,
                })
        if any(not np.isfinite(p).all() for p in oof.values()):
            raise ValueError("grouped-binomial out-of-fold prediction missing")
        for model in models:
            scored = grouped_scores(d["local"].to_numpy(dtype=float), d["n"].to_numpy(dtype=float), oof[model])
            observed = int(d["local"].sum())
            total = int(d["n"].sum())
            score_rows.append({
                "service": service,
                "threshold": threshold,
                "repartition": rep,
                "model": model,
                "municipalities": len(d),
                "resident_AIHS": total,
                "observed_local_AIHS": observed,
                "observed_local_rate": observed / total,
                "count_error": scored["predicted_local_records"] - observed,
                **scored,
            })
        pred = d[["m6", "n", "local", "block_id"]].copy()
        pred.insert(0, "service", service)
        pred.insert(1, "threshold", threshold)
        pred.insert(2, "repartition", rep)
        pred["fold"] = assignments[rep]
        pred["observed_rate"] = pred["local"] / pred["n"]
        for model in models:
            pred[f"p_{model}"] = oof[model]
            pred[f"expected_local_{model}"] = pred["n"] * oof[model]
        prediction_frames.append(pred)
    return pd.concat(prediction_frames, ignore_index=True), pd.DataFrame(score_rows), pd.DataFrame(fold_rows)


def temporal_validation(features: pd.DataFrame, outcomes: pd.DataFrame, service: str, threshold: int, bootstrap_replicates: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_all = observed_outcome(features, outcomes, service, 2023)
    test_all = observed_outcome(features, outcomes, service, 2024)
    train = add_blocks(train_all.loc[spatial_valid_mask(train_all) & train_all["n"].ge(threshold)].copy().sort_values("m6").reset_index(drop=True))
    test = add_blocks(test_all.loc[spatial_valid_mask(test_all) & test_all["n"].ge(threshold)].copy().sort_values("m6").reset_index(drop=True))
    models = STANDARD_MODELS
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    prediction = test[["m6", "n", "local", "block_id"]].copy()
    prediction.insert(0, "service", service)
    prediction.insert(1, "eligibility_threshold_resident_AIHS", threshold)
    prediction["observed_rate"] = prediction["local"] / prediction["n"]
    prediction["municipality_seen_in_training"] = prediction["m6"].isin(train["m6"])
    for model in models:
        probability, fit_info = grouped_binomial_fit(train, test, model)
        prediction[f"p_{model}"] = probability
        prediction[f"expected_local_{model}"] = test["n"].to_numpy(dtype=float) * probability
        scored = grouped_scores(test["local"].to_numpy(dtype=float), test["n"].to_numpy(dtype=float), probability)
        rows.append({
            "service": service,
            "eligibility_threshold_resident_AIHS": threshold,
            "model": model,
            "train_year": 2023,
            "test_year": 2024,
            "n_train_municipalities": len(train),
            "n_test_municipalities": len(test),
            "test_municipalities_not_seen_in_training": int((~prediction["municipality_seen_in_training"]).sum()),
            "train_resident_AIHS": int(train["n"].sum()),
            "test_resident_AIHS": int(test["n"].sum()),
            "test_observed_local_AIHS": int(test["local"].sum()),
            "test_observed_local_rate": float(test["local"].sum() / test["n"].sum()),
            **scored,
            **fit_info,
        })
        diagnostics.append({"service": service, "threshold": threshold, "model": model, **fit_info})
    # Paired bootstrap intervals resample whole two-degree blocks in the 2024 test set.
    bootstrap_rng = np.random.default_rng(seed)
    blocks = test["block_id"].to_numpy()
    local = test["local"].to_numpy(dtype=float)
    n = test["n"].to_numpy(dtype=float)
    bootstrap_values: dict[tuple[str, str, str], list[float]] = {}
    pairs = [
        ("population_urban_activity", "population_urban"),
        ("population_urban_OSM", "population_urban"),
        ("strong_baseline_plus_activity", "strong_baseline"),
    ]
    for pair in pairs:
        for metric in ["auc", "brier", "log_loss", "predicted_local_records"]:
            bootstrap_values[(*pair, metric)] = []
    prob_map = {model: prediction[f"p_{model}"].to_numpy(dtype=float) for model in models}
    for _ in range(bootstrap_replicates):
        weights = block_multiplicity(blocks, bootstrap_rng)
        score_map = {model: grouped_scores(local, n, prob_map[model], weights) for model in models}
        for left, right in pairs:
            for metric in ["auc", "brier", "log_loss", "predicted_local_records"]:
                bootstrap_values[(left, right, metric)].append(score_map[left][metric] - score_map[right][metric])
    bootstrap_rows = []
    for (left, right, metric), values in bootstrap_values.items():
        arr = np.asarray(values, dtype=float)
        bootstrap_rows.append({
            "service": service,
            "eligibility_threshold_resident_AIHS": threshold,
            "comparison": f"{left} minus {right}",
            "metric": metric,
            "block_bootstrap_replicates": bootstrap_replicates,
            "block_bootstrap_p025": float(np.nanquantile(arr, 0.025)),
            "block_bootstrap_median": float(np.nanmedian(arr)),
            "block_bootstrap_p975": float(np.nanquantile(arr, 0.975)),
            "uncertainty_interpretation": "paired percentile interval from resampling whole 2-degree blocks in the 2024 test municipalities",
        })
    return prediction, pd.DataFrame(rows), pd.DataFrame(bootstrap_rows)


def inclusion_tables(feature_by_service: dict[str, pd.DataFrame], outcomes: pd.DataFrame, output_dir: Path, bootstrap_replicates: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    count_rows: list[dict[str, Any]] = []
    characteristic_rows: list[dict[str, Any]] = []
    rate_rows: list[dict[str, Any]] = []
    for service in SERVICES:
        data = observed_outcome(feature_by_service[service], outcomes, service, 2024)
        valid = spatial_valid_mask(data)
        for threshold in THRESHOLDS:
            no_record = ~data["outcome_observed"]
            below = data["outcome_observed"] & data["n"].lt(threshold)
            count_included = data["outcome_observed"] & data["n"].ge(threshold)
            missing = count_included & ~valid
            included = count_included & valid
            group_masks = {
                "no_record_outcome_unknown": no_record,
                "recorded_below_threshold": below,
                "threshold_met_missing_spatial_covariates": missing,
                "included_in_spatial_validation": included,
            }
            for label, mask in group_masks.items():
                subset = data.loc[mask]
                observed_subset = subset.loc[subset["outcome_observed"]]
                count_rows.append({
                    "service": service,
                    "threshold_resident_AIHS": threshold,
                    "group": label,
                    "municipalities": len(subset),
                    "resident_AIHS": int(observed_subset["n"].sum()) if len(observed_subset) else np.nan,
                    "local_AIHS": int(observed_subset["local"].sum()) if len(observed_subset) else np.nan,
                    "observed_local_rate": float(observed_subset["local"].sum() / observed_subset["n"].sum()) if len(observed_subset) and observed_subset["n"].sum() > 0 else np.nan,
                    "outcome_interpretation": "no record means unknown and is never assigned a zero outcome" if label == "no_record_outcome_unknown" else "observed resident procedure records",
                })
                row: dict[str, Any] = {
                    "service": service,
                    "threshold_resident_AIHS": threshold,
                    "group": label,
                    "municipalities": len(subset),
                    "median_population": float(subset["pop"].median()) if len(subset) else np.nan,
                    "median_urban_share": float(subset["urban_share"].median()) if len(subset) else np.nan,
                    "mean_urban_share": float(subset["urban_share"].mean()) if len(subset) else np.nan,
                    "median_OSM_30min_access": float(subset["D1_osm"].median()) if len(subset) else np.nan,
                    "median_historical_activity_30min_access": float(subset["D2_act_pre"].median()) if len(subset) else np.nan,
                    "median_prior_local_treatment_volume": float(subset["prior_treatment_count"].median()) if len(subset) else np.nan,
                    "prior_local_service_fraction": float(subset["prior_local_service"].mean()) if len(subset) else np.nan,
                    "no_record_is_unknown": label == "no_record_outcome_unknown",
                }
                for region in ["North", "Northeast", "Southeast", "South", "Midwest"]:
                    row[f"region_{region}_municipalities"] = int(subset["region"].eq(region).sum())
                characteristic_rows.append(row)
            observed_eligible = data.loc[included].copy()
            if len(observed_eligible):
                observed_eligible = add_blocks(observed_eligible)
                rate = float(observed_eligible["local"].sum() / observed_eligible["n"].sum())
                ci_low, ci_high = grouped_rate_interval(observed_eligible, bootstrap_replicates, seed + 1000 * THRESHOLDS.index(threshold) + SERVICES.index(service))
                rate_rows.append({
                    "service": service,
                    "threshold": threshold,
                    "municipalities": len(observed_eligible),
                    "resident_AIHS": int(observed_eligible["n"].sum()),
                    "observed_local_AIHS": int(observed_eligible["local"].sum()),
                    "observed_local_rate": rate,
                    "rate_block_bootstrap_p025": ci_low,
                    "rate_block_bootstrap_p975": ci_high,
                    "uncertainty_interpretation": "percentile interval from resampling whole 2-degree municipality blocks",
                })
    count_frame = pd.DataFrame(count_rows)
    characteristic_frame = pd.DataFrame(characteristic_rows)
    rate_frame = pd.DataFrame(rate_rows)
    count_frame.to_csv(output_dir / "inclusion_counts.csv", index=False)
    characteristic_frame.to_csv(output_dir / "inclusion_characteristics.csv", index=False)
    rate_frame.to_csv(output_dir / "observed_denominator_rates.csv", index=False)
    return count_frame, characteristic_frame, rate_frame


def annual_linkage(project_root: Path, candidate: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hospital_path = project_root / "upgrade_20261112" / "inputs" / "brazil_hospital_annual.csv"
    annual = pd.read_csv(hospital_path, dtype={"CNES": "string"})
    annual["CNES"] = normalize_code(annual["CNES"], 7)
    annual["year"] = pd.to_numeric(annual["year"], errors="coerce").astype(int)
    annual["n"] = pd.to_numeric(annual["n"], errors="coerce").astype(int)
    candidate_codes = set(candidate["CLUES"].dropna())
    annual["linked_to_CNES2025_candidate"] = annual["CNES"].isin(candidate_codes)
    summary_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    unlinked_rows: list[dict[str, Any]] = []
    selected = annual.loc[annual["year"].isin([2017, 2018, 2019, 2023, 2024])].copy()
    for (service, year), group in selected.groupby(["service", "year"], sort=True):
        linked = group["linked_to_CNES2025_candidate"]
        linked_facility = group.loc[linked, "n"]
        unlinked_facility = group.loc[~linked, "n"]
        total_volume = int(group["n"].sum())
        linked_volume = int(group.loc[linked, "n"].sum())
        summary_rows.append({
            "service": service,
            "processing_year": int(year),
            "active_hospitals": int(group["CNES"].nunique()),
            "active_hospitals_linked_to_CNES2025_candidate": int(group.loc[linked, "CNES"].nunique()),
            "hospital_linkage_fraction": float(group.loc[linked, "CNES"].nunique() / group["CNES"].nunique()),
            "resident_procedure_AIHS": total_volume,
            "AIHS_linked_to_CNES2025_candidate": linked_volume,
            "volume_weighted_linkage_fraction": float(linked_volume / total_volume) if total_volume else np.nan,
            "median_AIHS_per_linked_hospital": float(linked_facility.median()) if len(linked_facility) else np.nan,
            "median_AIHS_per_unlinked_hospital": float(unlinked_facility.median()) if len(unlinked_facility) else np.nan,
            "mean_AIHS_per_linked_hospital": float(linked_facility.mean()) if len(linked_facility) else np.nan,
            "mean_AIHS_per_unlinked_hospital": float(unlinked_facility.mean()) if len(unlinked_facility) else np.nan,
            "linkage_definition": "exact normalized CNES identifier match to the coordinate-valid common CNES2025 candidate facility table used for activity exposures",
        })
        unlinked = group.loc[~linked, ["year", "service", "CNES", "n"]].copy()
        unlinked_rows.extend(unlinked.to_dict(orient="records"))

    pre = selected.loc[selected["year"].isin([2017, 2018, 2019])]
    for service, group in pre.groupby("service", sort=True):
        years_active = group.groupby("CNES")["year"].nunique().rename("active_years")
        volumes = group.groupby("CNES")["n"].sum().rename("three_year_AIHS")
        linked_flags = group.groupby("CNES")["linked_to_CNES2025_candidate"].max().rename("linked")
        per_facility = pd.concat([years_active, volumes, linked_flags], axis=1).reset_index()
        for active_years, stratum in per_facility.groupby("active_years", sort=True):
            linked = stratum["linked"].astype(bool)
            history_rows.append({
                "service": service,
                "active_years_in_2017_2019_union": int(active_years),
                "historically_active_hospitals": len(stratum),
                "linked_hospitals": int(linked.sum()),
                "hospital_linkage_fraction": float(linked.mean()),
                "historical_resident_procedure_AIHS": int(stratum["three_year_AIHS"].sum()),
                "linked_historical_AIHS": int(stratum.loc[linked, "three_year_AIHS"].sum()),
                "volume_weighted_linkage_fraction": float(stratum.loc[linked, "three_year_AIHS"].sum() / stratum["three_year_AIHS"].sum()) if stratum["three_year_AIHS"].sum() else np.nan,
                "median_three_year_AIHS_linked": float(stratum.loc[linked, "three_year_AIHS"].median()) if linked.any() else np.nan,
                "median_three_year_AIHS_unlinked": float(stratum.loc[~linked, "three_year_AIHS"].median()) if (~linked).any() else np.nan,
            })
    annual_frame = pd.DataFrame(summary_rows)
    history_frame = pd.DataFrame(history_rows)
    unlinked_frame = pd.DataFrame(unlinked_rows)
    annual_frame.to_csv(output_dir / "annual_hospital_CNES2025_linkage.csv", index=False)
    history_frame.to_csv(output_dir / "historical_union_survivorship_by_active_years.csv", index=False)
    unlinked_frame.to_csv(output_dir / "unlinked_historical_CNES_counts.csv", index=False)
    return annual_frame, history_frame, unlinked_frame


def binary_summaries(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for (service, model), group in scores.groupby(["service", "model"], sort=True):
        row: dict[str, Any] = {
            "service": service,
            "eligibility_threshold_resident_AIHS": 10,
            "model": model,
            "municipalities": int(group["municipalities"].iloc[0]),
            "positive_municipalities": int(group["positive_municipalities"].iloc[0]),
            "positive_fraction": float(group["positive_fraction"].iloc[0]),
            "n_repartitions": int(group["repartition"].nunique()),
        }
        for metric in ["auc", "brier", "log_loss", "predicted_positive_probability_sum"]:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_repartition_p025"] = float(np.quantile(values, 0.025))
            row[f"{metric}_repartition_p975"] = float(np.quantile(values, 0.975))
        row["uncertainty_interpretation"] = "partition variation over 50 two-degree-block repartitions; not a population confidence interval"
        rows.append(row)
    pairs = [p for p in COMPARISONS if all(name in set(scores["model"]) for name in p)]
    increments = []
    for service, group in scores.groupby("service", sort=True):
        wide = group.pivot(index="repartition", columns="model", values=["auc", "brier", "log_loss"])
        for left, right in pairs:
            for metric in ["auc", "brier", "log_loss"]:
                delta = (wide[(metric, left)] - wide[(metric, right)]).to_numpy(dtype=float)
                increments.append({
                    "service": service,
                    "eligibility_threshold_resident_AIHS": 10,
                    "comparison": f"{left} minus {right}",
                    "metric": metric,
                    "mean_delta": float(delta.mean()),
                    "repartition_p025": float(np.quantile(delta, 0.025)),
                    "repartition_p975": float(np.quantile(delta, 0.975)),
                    "n_repartitions": int(len(delta)),
                    "n_positive_deltas": int((delta > 0).sum()),
                    "n_negative_deltas": int((delta < 0).sum()),
                    "uncertainty_interpretation": "variation over 50 held-out block repartitions; not a population confidence interval",
                })
    return pd.DataFrame(rows), pd.DataFrame(increments)


def run_analysis(args: argparse.Namespace) -> int:
    project_root = args.project_root
    upgrade = project_root / "upgrade_20261112"
    output = args.output_dir.resolve() / "analysis"
    output.mkdir(parents=True, exist_ok=False)
    outcome_path = upgrade / "inputs" / "brazil_municipality_annual.csv"
    hospital_path = upgrade / "inputs" / "brazil_hospital_annual.csv"
    hospital_points_path = upgrade / "inputs" / "brazil_hospital_points.csv"
    coordinates_path = args.coordinates

    feature_by_service, candidate, source_table = build_features(project_root, coordinates_path)
    outcomes = load_outcomes(outcome_path)
    inclusion_counts, inclusion_characteristics, rate_rows = inclusion_tables(
        feature_by_service, outcomes, output, args.bootstrap_replicates, args.seed + 17
    )
    linkage, history_linkage, unlinked = annual_linkage(project_root, candidate, output)

    # Main grouped-binomial validation uses 10 seeded block repartitions at each
    # prespecified threshold. Flexible spline baselines are fitted for n>=1 and
    # n>=10, including the main denominator cohort, and share the first 10 folds
    # with the 50-repeat binary analysis at n>=10.
    denominator_predictions: list[pd.DataFrame] = []
    denominator_scores: list[pd.DataFrame] = []
    denominator_folds: list[pd.DataFrame] = []
    binary_predictions: list[pd.DataFrame] = []
    binary_scores_list: list[pd.DataFrame] = []
    binary_folds: list[pd.DataFrame] = []
    temporal_predictions: list[pd.DataFrame] = []
    temporal_scores: list[pd.DataFrame] = []
    temporal_bootstrap: list[pd.DataFrame] = []

    for service in SERVICES:
        data_2024 = observed_outcome(feature_by_service[service], outcomes, service, 2024)
        binary_pred, binary_score, binary_fold = binary_cv(data_2024, service, args.binary_repetitions, args.seed, output)
        binary_predictions.append(binary_pred)
        binary_scores_list.append(binary_score)
        binary_folds.append(binary_fold)
        print(f"BINARY_50_REPARTITIONS_COMPLETE {service} n={binary_score['municipalities'].iloc[0]}", flush=True)

        for threshold in THRESHOLDS:
            pred, score, fold = denominator_cv(data_2024, service, threshold, args.denominator_repetitions, args.seed)
            denominator_predictions.append(pred)
            denominator_scores.append(score)
            denominator_folds.append(fold)
            print(f"DENOMINATOR_CV_COMPLETE {service} threshold={threshold} n={score['municipalities'].iloc[0]}", flush=True)

    den_pred = pd.concat(denominator_predictions, ignore_index=True)
    den_scores = pd.concat(denominator_scores, ignore_index=True)
    den_folds = pd.concat(denominator_folds, ignore_index=True)
    bin_pred = pd.concat(binary_predictions, ignore_index=True)
    bin_scores = pd.concat(binary_scores_list, ignore_index=True)
    bin_folds = pd.concat(binary_folds, ignore_index=True)

    # Persist fitted predictions, scores, and fold diagnostics before any
    # post-fit validation or bootstrap summaries. A failure below can then be
    # fixed from these files without repeating the model fits.
    den_pred.to_csv(output / "denominator_spatial_oof_predictions_fit_checkpoint.csv.gz", index=False)
    den_scores.to_csv(output / "denominator_spatial_scores_by_repartition_fit_checkpoint.csv", index=False)
    den_folds.to_csv(output / "denominator_glm_fold_diagnostics_fit_checkpoint.csv", index=False)
    bin_pred.to_csv(output / "binary_spatial_oof_predictions_fit_checkpoint.csv.gz", index=False)
    bin_scores.to_csv(output / "binary_spatial_scores_by_repartition_fit_checkpoint.csv", index=False)
    bin_folds.to_csv(output / "binary_spatial_fold_manifest_fit_checkpoint.csv", index=False)
    print("SPATIAL_FIT_CHECKPOINTS_WRITTEN", flush=True)

    # Confirm that threshold-10 denominator and binary analyses share their first
    # 10 geographic repartitions exactly at the municipality level.
    for service in SERVICES:
        left = den_pred.loc[den_pred["service"].eq(service) & den_pred["threshold"].eq(10) & den_pred["repartition"].lt(args.denominator_repetitions), ["m6", "repartition", "fold"]]
        right = bin_pred.loc[
            bin_pred["service"].eq(service) & bin_pred["repartition"].lt(args.denominator_repetitions),
            ["m6", "repartition", "fold"],
        ]
        check = left.merge(right, on=["m6", "repartition"], how="outer", suffixes=("_denominator", "_binary"), validate="one_to_one")
        if check[["fold_denominator", "fold_binary"]].isna().any().any() or not check["fold_denominator"].eq(check["fold_binary"]).all():
            raise ValueError(f"threshold-10 denominator and binary repartitions differ for {service}")

    # Temporal holdout is deliberately run after spatial fit products have
    # been checkpointed. Both years use the same fixed 2025 spatial inputs.
    for service in SERVICES:
        for threshold in [1, 10]:
            pred, score, boot = temporal_validation(
                feature_by_service[service], outcomes, service, threshold,
                args.bootstrap_replicates, args.seed + 500 + SERVICES.index(service) * 20 + threshold,
            )
            temporal_predictions.append(pred)
            temporal_scores.append(score)
            temporal_bootstrap.append(boot)
            print(f"TEMPORAL_DENOMINATOR_COMPLETE {service} threshold={threshold} n={len(pred)}", flush=True)
    temp_pred = pd.concat(temporal_predictions, ignore_index=True)
    temp_scores = pd.concat(temporal_scores, ignore_index=True)
    temp_boot = pd.concat(temporal_bootstrap, ignore_index=True)

    rate_rows = rate_rows.to_dict(orient="records")
    den_summary, _ = summarize_spatial_scores(den_scores, "denominator", rate_rows)
    den_increments = summarize_increments(den_scores, COMPARISONS)
    binary_summary, binary_increments = binary_summaries(bin_scores)
    den_summary.to_csv(output / "denominator_spatial_absolute_performance_fit_checkpoint.csv", index=False)
    den_increments.to_csv(output / "denominator_spatial_increment_summary_fit_checkpoint.csv", index=False)
    binary_summary.to_csv(output / "binary_spatial_absolute_performance_fit_checkpoint.csv", index=False)
    binary_increments.to_csv(output / "binary_spatial_increment_summary_fit_checkpoint.csv", index=False)
    den_boot = pd.concat([
        block_bootstrap_spatial_deltas(den_pred, threshold, COMPARISONS, args.bootstrap_replicates, args.seed + 900 + threshold)
        for threshold in [1, 10]
    ], ignore_index=True)

    # Scores are pooled over held-out municipality predictions within each split.
    den_pred.to_csv(output / "denominator_spatial_oof_predictions.csv.gz", index=False)
    den_scores.to_csv(output / "denominator_spatial_scores_by_repartition.csv", index=False)
    den_summary.to_csv(output / "denominator_spatial_absolute_performance.csv", index=False)
    den_increments.to_csv(output / "denominator_spatial_increment_summary.csv", index=False)
    den_boot.to_csv(output / "denominator_spatial_block_bootstrap.csv", index=False)
    den_folds.to_csv(output / "denominator_glm_fold_diagnostics.csv", index=False)
    bin_pred.to_csv(output / "binary_spatial_oof_predictions.csv.gz", index=False)
    bin_scores.to_csv(output / "binary_spatial_scores_by_repartition.csv", index=False)
    binary_summary.to_csv(output / "binary_spatial_absolute_performance.csv", index=False)
    binary_increments.to_csv(output / "binary_spatial_increment_summary.csv", index=False)
    bin_folds.to_csv(output / "binary_spatial_fold_manifest.csv", index=False)
    temp_pred.to_csv(output / "temporal_denominator_predictions.csv.gz", index=False)
    temp_scores.to_csv(output / "temporal_denominator_absolute_performance.csv", index=False)
    temp_boot.to_csv(output / "temporal_denominator_block_bootstrap.csv", index=False)

    input_paths = [
        outcome_path,
        hospital_path,
        hospital_points_path,
        upgrade / "derived" / "brazil_spatial_v1" / "municipality_population.csv",
        upgrade / "derived" / "brazil_spatial_v1" / "candidate_facilities_with_local_pop.csv",
        coordinates_path,
    ] + [upgrade / "derived" / "brazil_spatial_v1" / f"{service}_prior_municipality.csv" for service in SERVICES]
    manifest = {
        "objective": "resident-local treatment fraction among observed resident procedure AIH records",
        "outcome_definition": "local treatment successes / all resident procedure AIH records, where local means MUNIC_RES equals MUNIC_MOV",
        "unit_and_repeated_records": "municipality-year aggregated AIH records; AIHs are not unique patients and no unique-patient denominator is available",
        "no_record_rule": "municipalities without resident procedure records are labeled outcome unknown and excluded from outcome model fitting; they are not assigned zero outcome",
        "denominator_model": "statsmodels GLM Binomial with two-column grouped counts [local successes, resident records minus local successes], logit link, Pearson-X2 scale (quasi-binomial dispersion adjustment)",
        "uncertainty": "performance intervals from 10 seeded two-degree-block five-fold repartitions; paired percentile intervals additionally resample whole two-degree blocks (200 replicates). Repartition and block-bootstrap intervals are not individual-AIH confidence intervals.",
        "thresholds": THRESHOLDS,
        "flexible_spline_thresholds": [1, 10],
        "denominator_repartitions": args.denominator_repetitions,
        "binary_repartitions": args.binary_repetitions,
        "seed": args.seed,
        "spatial_fold_definition": "2-degree latitude/longitude blocks shuffled with NumPy default_rng(seed=20261111), assigned round-robin to 5 folds; one held-out prediction per municipality/repartition",
        "binary_outcome_definition": "any local AIH among municipalities with at least 10 resident AIHs; original logistic specification is retained and flexible cubic B-spline terms are added as prespecified sensitivity",
        "temporal_split": "train 2023 grouped counts, test 2024 grouped counts; 2017-2019 features and common 2025 coordinates/population are fixed in both years",
        "features": {
            "population_urban": "log1p(GHSL2025 municipality population) and urban population / total municipality population",
            "population_urban_OSM": "population_urban plus population share within 30 minutes of an OSM hospital",
            "population_urban_activity": "population_urban plus population share within 30 minutes of the service-specific 2017-2019 facility union",
            "strong_baseline": "population_urban plus prior local service presence and log1p historical procedure volume grouped by the registered municipality of linked CNES2025 candidate hospitals",
            "strong_baseline_plus_activity": "strong_baseline plus historical 2017-2019 activity-based 30-minute access",
            "flexible_baseline": "cubic B-spline transformations (4 quantile knots, degree 3, fitted within each training fold) of log1p population and urban fraction, with listed additional linear terms",
            "access_formula": "sum of GHSL2025 population cells in a municipality reachable within 30 minutes of at least one service-active hospital / sum of GHSL2025 population cells in that municipality; travel may cross municipality boundaries",
            "historical_active_facility_union": "CNES hospital is active for a procedure if its official procedure-code AIH count summed over processing years 2017, 2018 and 2019 is greater than zero; the feature uses the any-year union, not persistence in all three years",
        },
        "absolute_scores": "AUC, AIH-weighted Brier and AIH-weighted log loss; expected local-event count is sum of municipality resident AIHs times predicted probability",
        "input_manifest": file_manifest(input_paths),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "statsmodels": sm.__version__,
            "patsy": patsy.__version__,
        },
        "reconciliation_note": "annual hospital linkage and all-year volume results are reported separately; raw date/identifier audit is limited to the five staged cholecystectomy extracts",
    }
    (output / "analysis_design.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    source_table.to_csv(output / "feature_source_paths.csv", index=False)
    print("ANALYSIS_COMPLETE", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--coordinates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--denominator-repetitions", type=int, default=10)
    parser.add_argument("--binary-repetitions", type=int, default=50)
    parser.add_argument("--bootstrap-replicates", type=int, default=BLOCK_BOOTSTRAPS)
    parser.add_argument("--seed", type=int, default=20261111)
    args = parser.parse_args()
    if args.denominator_repetitions < 1 or args.binary_repetitions < 1 or args.bootstrap_replicates < 20:
        parser.error("repartitions must be positive and bootstrap replicates at least 20")
    return run_analysis(args)


if __name__ == "__main__":
    raise SystemExit(main())
