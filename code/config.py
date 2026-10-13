"""Shared settings and helpers for the mRNA half-life analysis.

Feature matrix, train/test split and feature list are the files released with
the manuscript.  Everything downstream (model training, SHAP, ablation) is
reproduced from those files alone.
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# The feature matrix and the feature list are not distributed with this
# repository. Point HL_DATA_DIR at the directory that holds them.
DATA_DIR = os.environ.get("HL_DATA_DIR", os.path.join(ROOT, "data"))
RESULTS_DIR = os.environ.get("HL_RESULTS_DIR", os.path.join(ROOT, "results"))

FEATURE_MATRIX = os.path.join(DATA_DIR, "feature_matrix.csv")
SPLIT_FILE = os.path.join(DATA_DIR, "train_test_split.csv")
FEATURE_LIST = os.path.join(DATA_DIR, "feature_list.csv")

RANDOM_STATE = 42
TEST_SIZE = 0.2

# LightGBM settings used for every model in the manuscript.
LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "n_estimators": 2500,
    "learning_rate": 0.004,
    "max_depth": 8,
    "num_leaves": 15,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.6,
    "bagging_freq": 5,
    "lambda_l1": 5,
    "lambda_l2": 10,
    "verbosity": -1,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

ID_COLUMNS = ["transcript_index", "Accession", "gene_symbol", "tx_id", "half_life_h", "y"]


def load_features(path=FEATURE_MATRIX):
    """Return the 96-feature matrix X (DataFrame) and the target y (Series)."""
    table = pd.read_csv(path)
    feature_names = feature_names_from_list()
    missing = [c for c in feature_names if c not in table.columns]
    if missing:
        raise ValueError("features missing from the matrix: %s" % missing[:5])
    return table[feature_names].copy(), table["y"].copy(), table


def feature_names_from_list(path=FEATURE_LIST):
    return list(pd.read_csv(path)["feature"])


def feature_groups(path=FEATURE_LIST):
    """Return the Seq / Mod / Inter groups used in the ablation analysis."""
    table = pd.read_csv(path)
    seq = list(table.loc[table["group"] == "seq", "feature"])
    mod_stat = list(table.loc[table["group"] == "mod_stat", "feature"])
    mod_inter = list(table.loc[table["group"] == "mod_inter", "feature"])
    return {"seq": seq, "mod_stat": mod_stat, "mod_inter": mod_inter}
