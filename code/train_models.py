"""Model A (sequence only) versus Model B (sequence plus modifications).

Reproduces the test-set metrics reported in Table 1, including the one-sided
paired t-test on per-transcript squared prediction errors.

Usage
    python code/train_models.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_rel
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (LGB_PARAMS, RESULTS_DIR, RANDOM_STATE, TEST_SIZE,  # noqa: E402
                    feature_groups, load_features)


def main():
    X, y, table = load_features()
    groups = feature_groups()
    seq_features = groups["seq"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model_a = lgb.LGBMRegressor(**LGB_PARAMS).fit(X_train[seq_features], y_train)
    model_b = lgb.LGBMRegressor(**LGB_PARAMS).fit(X_train, y_train)

    pred_a = model_a.predict(X_test[seq_features])
    pred_b = model_b.predict(X_test)

    metrics = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "r2_train_A": float(r2_score(y_train, model_a.predict(X_train[seq_features]))),
        "r2_train_B": float(r2_score(y_train, model_b.predict(X_train))),
        "r2_A": float(r2_score(y_test, pred_a)),
        "r2_B": float(r2_score(y_test, pred_b)),
        "rho_A": float(spearmanr(y_test, pred_a)[0]),
        "rho_B": float(spearmanr(y_test, pred_b)[0]),
    }
    metrics["p_gain"] = float(
        ttest_rel((y_test - pred_a) ** 2, (y_test - pred_b) ** 2, alternative="greater")[1]
    )
    metrics["relative_gain_pct"] = (metrics["r2_B"] - metrics["r2_A"]) / abs(metrics["r2_A"]) * 100

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "metrics_model_A_vs_B.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    predictions = pd.DataFrame({
        "transcript_index": X_test.index,
        "observed_y": y_test.values,
        "predicted_y_model_A": pred_a,
        "predicted_y_model_B": pred_b,
    })
    predictions.to_csv(os.path.join(RESULTS_DIR, "test_set_predictions.csv"), index=False)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
