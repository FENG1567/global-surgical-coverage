#!/usr/bin/env python3
"""Regression check for threshold-specific prediction columns in bootstrap summaries."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).with_name("reanalyse_prediction.py")
SPEC = importlib.util.spec_from_file_location("reanalyse_prediction", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODEL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODEL)


def main() -> None:
    rows = []
    for threshold in [1, 10]:
        for repartition in [0, 1]:
            for index in range(10):
                row = {
                    "service": "chole",
                    "threshold": threshold,
                    "repartition": repartition,
                    "block_id": index // 2,
                    "local": 1 if index % 2 == 0 else 0,
                    "n": 10 + index,
                }
                for model_index, model in enumerate(MODEL.STANDARD_MODELS):
                    row[f"p_{model}"] = 0.2 + 0.1 * ((index + repartition + model_index) % 5)
                for model_index, model in enumerate(MODEL.FLEXIBLE_MODELS):
                    row[f"p_{model}"] = (
                        np.nan if threshold == 1
                        else 0.2 + 0.1 * ((index + repartition + model_index) % 5)
                    )
                rows.append(row)
    predictions = pd.DataFrame(rows)

    threshold_1 = MODEL.block_bootstrap_spatial_deltas(predictions, 1, MODEL.COMPARISONS, 20, 1)
    threshold_10 = MODEL.block_bootstrap_spatial_deltas(predictions, 10, MODEL.COMPARISONS, 20, 1)
    assert len(threshold_1) == 3 * 4
    assert len(threshold_10) == 5 * 4
    assert not threshold_1["comparison"].str.contains("flexible").any()
    print({
        "status": "prediction_bootstrap_smoke_passed",
        "threshold_1_rows": len(threshold_1),
        "threshold_10_rows": len(threshold_10),
    }, flush=True)


if __name__ == "__main__":
    main()
