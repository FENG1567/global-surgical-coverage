"""Auditable re-run of the municipal spatial-block comparison.

Requires urban population counts. The legacy urban_share column is never used.
No raster data, participant records, or frozen project outputs are modified.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REQUIRED = (
    "m6", "latitude", "longitude", "has_local", "pop",
    "urban_population", "D1_osm", "D2_act_pre",
)


class InvalidAnalysis(ValueError):
    """An input or validation failure that must prevent a performance claim."""


def validate_table(table: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(REQUIRED) - set(table.columns))
    if missing:
        raise InvalidAnalysis(f"Missing required columns: {missing}")
    d = table.loc[:, REQUIRED].copy()
    if d.empty:
        raise InvalidAnalysis("Input has no municipalities")
    if d[list(REQUIRED)].isna().any().any():
        raise InvalidAnalysis("Required values are missing; no rows were silently removed")
    d["m6"] = d.m6.astype(str)
    if not d.m6.str.fullmatch(r"\d{6}").all() or d.m6.duplicated().any():
        raise InvalidAnalysis("m6 must contain unique six-digit municipality identifiers")
    for col in REQUIRED[1:]:
        try:
            d[col] = pd.to_numeric(d[col], errors="raise")
        except (ValueError, TypeError) as exc:
            raise InvalidAnalysis(f"Non-numeric values in {col}") from exc
        if not np.isfinite(d[col].to_numpy(dtype=float)).all():
            raise InvalidAnalysis(f"Nonfinite values in {col}")
    if not d.latitude.between(-90, 90).all() or not d.longitude.between(-180, 180).all():
        raise InvalidAnalysis("Invalid geographic coordinates")
    if set(d.has_local.unique()) != {0, 1}:
        raise InvalidAnalysis("has_local must contain both binary classes 0 and 1")
    if (d["pop"] <= 0).any():
        raise InvalidAnalysis("Population must be positive")
    if (d.urban_population < 0).any() or (d.urban_population > d["pop"]).any():
        raise InvalidAnalysis("Urban population must be between zero and total population")
    for col in ("D1_osm", "D2_act_pre"):
        if not d[col].between(0, 1).all():
            raise InvalidAnalysis(f"{col} must be a coverage proportion in [0, 1]")
    d["urban_share"] = d.urban_population / d["pop"]
    d["logpop"] = np.log1p(d["pop"])
    d["block"] = (
        np.floor(d.latitude / 2).astype(np.int64) * 1000
        + np.floor(d.longitude / 2).astype(np.int64)
    )
    return d.reset_index(drop=True)


def _distribution(values: list[float] | np.ndarray) -> dict:
    v = np.asarray(values, dtype=float)
    return {
        "mean": float(v.mean()), "lo": float(np.percentile(v, 2.5)),
        "hi": float(np.percentile(v, 97.5)), "n_repartitions": len(v),
    }


def run_comparison(table: pd.DataFrame, repetitions: int = 50,
                   seed: int = 20261111, minimum_test_size: int = 50):
    if repetitions < 1 or minimum_test_size < 1:
        raise InvalidAnalysis("Repetitions and minimum test size must be positive")
    d = validate_table(table)
    blocks = d.block.to_numpy()
    unique_blocks = np.unique(blocks)
    if len(unique_blocks) < 5:
        raise InvalidAnalysis("At least five distinct spatial blocks are required")
    y = d.has_local.to_numpy(dtype=int)
    models = {
        "baseline": d[["logpop", "urban_share"]].to_numpy(),
        "baseline+OSM": d[["logpop", "urban_share", "D1_osm"]].to_numpy(),
        "baseline+activity": d[["logpop", "urban_share", "D2_act_pre"]].to_numpy(),
    }
    rng = np.random.default_rng(seed)
    scores = {name: [] for name in models}
    prediction_tables, diagnostics = [], []
    for rep in range(repetitions):
        perm = rng.permutation(unique_blocks)
        fold_of_block = {b: i % 5 for i, b in enumerate(perm)}
        folds = np.array([fold_of_block[b] for b in blocks])
        oof = {name: np.full(len(d), np.nan) for name in models}
        counts = {name: np.zeros(len(d), dtype=np.int16) for name in models}
        for fold in range(5):
            train, test = folds != fold, folds == fold
            diagnostic = {
                "repartition": rep, "fold": fold,
                "n_train": int(train.sum()), "n_test": int(test.sum()),
                "train_positive": int(y[train].sum()), "test_positive": int(y[test].sum()),
            }
            if test.sum() < minimum_test_size or len(np.unique(y[train])) < 2:
                raise InvalidAnalysis(
                    "Legacy skip condition encountered; no AUC produced: "
                    + json.dumps(diagnostic)
                )
            for name, matrix in models.items():
                model = LogisticRegression(max_iter=2000)
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model.fit(matrix[train], y[train])
                if any(issubclass(w.category, ConvergenceWarning) for w in captured):
                    raise InvalidAnalysis(
                        f"Nonconvergence: repetition={rep}, fold={fold}, model={name}; no AUC produced"
                    )
                probability = model.predict_proba(matrix[test])[:, 1]
                if not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
                    raise InvalidAnalysis("Invalid held-out probabilities")
                oof[name][test] = probability
                counts[name][test] += 1
            diagnostics.append(diagnostic)
        predictions = d[["m6", "has_local", "block", "urban_share"]].copy()
        predictions["repartition"], predictions["fold"] = rep, folds
        for name in models:
            if not (counts[name] == 1).all() or not np.isfinite(oof[name]).all():
                raise InvalidAnalysis(f"Incomplete or duplicate held-out predictions: {name}")
            scores[name].append(float(roc_auc_score(y, oof[name])))
            predictions[name] = oof[name]
            predictions[f"{name}_prediction_count"] = counts[name]
        prediction_tables.append(predictions)
    summary = {
        "status": "computed_on_supplied_municipality_table",
        "n_municipalities": len(d), "n_blocks": len(unique_blocks),
        "n_positive": int(y.sum()), "n_repartitions": repetitions, "seed": seed,
        "minimum_test_size": minimum_test_size, "block_size_degrees": 2,
        "model": "LogisticRegression(max_iter=2000), other installed-version defaults",
        "urban_share_definition": "urban_population / pop; legacy urban_share ignored",
        "interval_definition": "2.5th–97.5th percentiles across repartitions, NOT a population confidence interval",
        "auc_definition": "pooled out-of-fold AUC for each complete repartition",
        "all_oof_predictions_exactly_once": True,
        "models": {name: _distribution(v) for name, v in scores.items()},
        "paired_increments": {},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__},
    }
    for label, lhs, rhs in (
        ("+OSM - baseline", "baseline+OSM", "baseline"),
        ("+activity - baseline", "baseline+activity", "baseline"),
        ("+activity - +OSM", "baseline+activity", "baseline+OSM"),
    ):
        delta = np.asarray(scores[lhs]) - np.asarray(scores[rhs])
        summary["paired_increments"][label] = {
            **_distribution(delta), "n_positive": int((delta > 0).sum()),
            "n_nonpositive": int((delta <= 0).sum()),
        }
    return summary, pd.concat(prediction_tables, ignore_index=True), pd.DataFrame(diagnostics)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--urban-population-definition", required=True,
                        help="Document the source, year and urban classification of the supplied counts")
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261111)
    args = parser.parse_args(argv)
    try:
        if args.output_dir.exists():
            raise InvalidAnalysis("Output directory already exists; refusing overwrite")
        if not args.urban_population_definition.strip():
            raise InvalidAnalysis("An urban population source definition is required")
        raw_input = args.input.read_bytes()
        input_hash = hashlib.sha256(raw_input).hexdigest()
        table = pd.read_csv(io.BytesIO(raw_input), dtype={"m6": str})
        summary, predictions, folds = run_comparison(table, args.repetitions, args.seed)
        summary["input"] = {"path": str(args.input.resolve()),
                            "sha256": input_hash, "bytes": len(raw_input),
                            "urban_population_source": args.urban_population_definition}
        # Stage on the same filesystem. Only publish the final directory after
        # every artifact has been written successfully; failures remove staging.
        parent = args.output_dir.resolve().parent
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{args.output_dir.name}.staging-", dir=parent) as tmp:
            stage = Path(tmp)
            predictions.to_csv(stage / "oof_predictions.csv.gz", index=False)
            folds.to_csv(stage / "fold_diagnostics.csv", index=False)
            (stage / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            if args.output_dir.exists():
                raise InvalidAnalysis("Output directory appeared during computation; refusing overwrite")
            stage.rename(args.output_dir)
        print(json.dumps({"status": summary["status"], "n": len(table),
                          "output_dir": str(args.output_dir)}, ensure_ascii=False))
        return 0
    except (InvalidAnalysis, OSError, ValueError) as exc:
        print(json.dumps({"status": "INVALID_NO_PERFORMANCE_CLAIM", "reason": str(exc)},
                         ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
