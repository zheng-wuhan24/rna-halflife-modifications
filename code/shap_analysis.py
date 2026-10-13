"""TreeSHAP values for Model B on the held-out test set.

Writes the full SHAP matrix and the mean absolute SHAP ranking, which underlie
Fig. 2 and the dependence plots.

Usage
    python code/shap_analysis.py
"""
import os
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import LGB_PARAMS, RANDOM_STATE, RESULTS_DIR, TEST_SIZE, load_features  # noqa: E402


def main():
    X, y, table = load_features()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = lgb.LGBMRegressor(**LGB_PARAMS).fit(X_train, y_train)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X_test)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    shap_matrix = pd.DataFrame(values, columns=X.columns)
    shap_matrix.insert(0, "transcript_index", list(X_test.index))
    shap_matrix.to_csv(os.path.join(RESULTS_DIR, "shap_values_testset.csv"), index=False)

    ranking = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": np.abs(values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    ranking.to_csv(os.path.join(RESULTS_DIR, "shap_mean_abs_ranking.csv"), index=False)

    print(ranking.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
