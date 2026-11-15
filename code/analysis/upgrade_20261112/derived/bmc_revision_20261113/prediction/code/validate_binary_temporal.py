#!/usr/bin/env python3
"""Synthetic smoke check for the binary temporal extension."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).with_name("binary_temporal_strong.py")
SPEC = importlib.util.spec_from_file_location("binary_temporal_strong", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODEL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODEL)


def main() -> None:
    rng = np.random.default_rng(20261113)
    n_municipalities = 50
    m6 = [f"{100000 + index:06d}" for index in range(n_municipalities)]
    population = rng.integers(1_000, 100_000, n_municipalities)
    features = pd.DataFrame({
        "m6": m6,
        "pop": population,
        "urban_population": population * rng.uniform(0.2, 0.9, n_municipalities),
        "D1_osm": rng.uniform(0.0, 1.0, n_municipalities),
        "D2_act_pre": rng.uniform(0.0, 1.0, n_municipalities),
        "latitude": rng.uniform(-30.0, 5.0, n_municipalities),
        "longitude": rng.uniform(-70.0, -35.0, n_municipalities),
        "logpop": np.log1p(population),
        "urban_share": rng.uniform(0.2, 0.9, n_municipalities),
        "prior_local_service": rng.integers(0, 2, n_municipalities),
        "log_prior_treatment_count": rng.uniform(0.0, 5.0, n_municipalities),
    })
    outcomes = pd.DataFrame([
        {
            "year": year,
            "service": "chole",
            "MUNIC_RES": code,
            "n": total,
            "local": int(rng.integers(1, total + 1)) if index % 3 else 0,
        }
        for year in [2023, 2024]
        for index, code in enumerate(m6)
        for total in [int(rng.integers(10, 70))]
    ])
    outcomes["m6"] = outcomes["MUNIC_RES"]

    predictions, scores, bootstrap = MODEL.run_service(features, outcomes, "chole", 20, 20261113)
    assert len(predictions) == n_municipalities
    assert scores["model"].tolist() == MODEL.TEMPORAL_MODELS
    assert predictions.filter(regex=r"^p_").shape[1] == len(MODEL.TEMPORAL_MODELS)
    assert len(bootstrap) == len(MODEL.TEMPORAL_COMPARISONS) * 4
    assert bootstrap["valid_block_bootstrap_replicates"].max() <= 20
    print({
        "status": "binary_temporal_smoke_passed",
        "test_municipalities": len(predictions),
        "models": scores["model"].tolist(),
        "paired_bootstrap_rows": len(bootstrap),
    }, flush=True)


if __name__ == "__main__":
    main()
