"""Ablation analysis of the three feature groups under 10-fold cross-validation.

Evaluates the seven non-empty combinations of sequence features (Seq),
modification statistics (Mod) and modification interaction features (Inter),
using one shared fold assignment for every configuration, and compares each
configuration with the full model by a two-sided paired t-test on fold-wise R2.

Usage
    python code/ablation.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import ttest_rel
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import LGB_PARAMS, RANDOM_STATE, RESULTS_DIR, feature_groups, load_features  # noqa: E402

CONFIGURATIONS = [
    ("Seq", ["seq"]),
    ("Mod", ["mod_stat"]),
    ("Inter", ["mod_inter"]),
    ("Mod+Inter", ["mod_stat", "mod_inter"]),
    ("Seq+Mod", ["seq", "mod_stat"]),
    ("Seq+Inter", ["seq", "mod_inter"]),
    ("Full", ["seq", "mod_stat", "mod_inter"]),
]


def main():
    X, y, _ = load_features()
    groups = feature_groups()

    folds = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    fold_index = list(folds.split(X))

    rows = []
    for name, parts in CONFIGURATIONS:
        columns = [c for part in parts for c in groups[part]]
        for fold, (train_idx, valid_idx) in enumerate(fold_index, start=1):
            model = lgb.LGBMRegressor(**LGB_PARAMS)
            model.fit(X.iloc[train_idx][columns], y.iloc[train_idx])
            r2 = r2_score(y.iloc[valid_idx], model.predict(X.iloc[valid_idx][columns]))
            rows.append({"configuration": name, "fold": fold, "r2": r2})

    scores = pd.DataFrame(rows)
    summary = scores.groupby("configuration")["r2"].agg(["mean", "std"])
    summary = summary.reindex([name for name, _ in CONFIGURATIONS])

    full = scores.loc[scores["configuration"] == "Full", "r2"].values
    p_values = {}
    for name, _ in CONFIGURATIONS[:-1]:
        values = scores.loc[scores["configuration"] == name, "r2"].values
        p_values[name] = float(ttest_rel(full, values)[1])
    p_values["Full"] = None

    os.makedirs(RESULTS_DIR, exist_ok=True)
    scores.to_csv(os.path.join(RESULTS_DIR, "ablation_fold_scores.csv"), index=False)

    report = {
        name: {
            "mean_r2": float(summary.loc[name, "mean"]),
            "sd_r2_sample": float(summary.loc[name, "std"]),
            "sd_r2_population": float(scores.loc[scores["configuration"] == name, "r2"].std(ddof=0)),
            "p_vs_full": p_values[name],
        }
        for name, _ in CONFIGURATIONS
    }
    with open(os.path.join(RESULTS_DIR, "ablation_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
