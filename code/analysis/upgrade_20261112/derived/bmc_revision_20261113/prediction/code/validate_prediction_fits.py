#!/usr/bin/env python3
"""One-fold smoke validation for grouped-binomial linear/spline specifications."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("reanalyse_prediction.py")
SPEC = importlib.util.spec_from_file_location("reanalyse_prediction", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODEL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODEL)

PROJECT = Path("__PROJECT_ROOT__")
COORDINATES = Path("__BRAZIL_SOURCE_ROOT__/01_data/interim/municipios_master.csv")


def main() -> None:
    feature_sets, _, _ = MODEL.build_features(PROJECT, COORDINATES)
    outcomes = MODEL.load_outcomes(PROJECT / "upgrade_20261112/inputs/brazil_municipality_annual.csv")
    data = MODEL.observed_outcome(feature_sets["chole"], outcomes, "chole", 2024)
    data = MODEL.add_blocks(data.loc[MODEL.spatial_valid_mask(data) & data["n"].ge(10)].copy().sort_values("m6").reset_index(drop=True))
    folds = MODEL.repartitions(data["block_id"].to_numpy(), 10, 20261111)[0]
    train, test = data.loc[folds != 0], data.loc[folds == 0]
    output = []
    for name in [
        "population_urban",
        "strong_baseline_plus_activity",
        "flexible_population_urban",
        "flexible_strong_baseline_plus_activity",
    ]:
        probabilities, diagnostics = MODEL.grouped_binomial_fit(train, test, name)
        if not np.isfinite(probabilities).all():
            raise ValueError(f"non-finite probabilities from {name}")
        output.append({"model": name, "test_municipalities": len(test), **diagnostics})
    print(json.dumps({"status": "smoke_check_passed", "models": output}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
