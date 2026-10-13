"""Build the 96-feature matrix from the transcript-level input table.

Input table columns required
    sequence    full-length spliced transcript sequence (A/C/G/T or U)
    T           half-life in hours, on the scale used in the manuscript
    annotation  modification sites as "pos(type);pos(type);..." with positions
                given as 1-based offsets on the spliced transcript

Usage
    python code/feature_extraction.py --input input_table.xlsx --output data/feature_matrix.csv
"""
import argparse
import os
import re
from collections import Counter
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize

TRINUCLEOTIDES = ["".join(p) for p in product("AUCG", repeat=3)]


def parse_modifications(value):
    """Return a list of (position, modification type) pairs."""
    if not isinstance(value, str):
        return []
    return [(int(p), t) for p, t in re.findall(r"(\d+)\(([^)]*)\)", value)]


def build_features(df):
    """Sequence features, modification statistics and modification interactions."""
    modification_types = sorted(
        {t for value in df["annotation"].dropna() for _, t in parse_modifications(value)}
    )

    records = []
    keep = []
    for idx, row in df.iterrows():
        sequence = str(row["sequence"]).upper().replace("T", "U")
        length = len(sequence)
        if length < 50:
            continue
        keep.append(idx)

        record = {
            "log_seq_length": np.log1p(length),
            "gc_content": (sequence.count("G") + sequence.count("C")) / length,
            "max_polyA": max((len(m) for m in re.findall(r"A+", sequence)), default=0),
            "max_polyC": max((len(m) for m in re.findall(r"C+", sequence)), default=0),
            "max_polyU": max((len(m) for m in re.findall(r"U+", sequence)), default=0),
            "max_polyG": max((len(m) for m in re.findall(r"G+", sequence)), default=0),
        }
        for base in "AUCG":
            record["nt_freq_" + base] = sequence.count(base) / length

        sites = parse_modifications(row["annotation"])
        for m_type in modification_types:
            own = [p for p, t in sites if t == m_type]
            other = [p for p, t in sites if t != m_type]
            record["CrossMod_Interaction_" + m_type] = sum(
                1 for p in own if any(abs(p - q) <= 10 for q in other)
            )

        types_present = [t for _, t in sites]
        record["mod_types_diversity"] = len(set(types_present))
        for m_type in modification_types:
            record[m_type + "_density"] = types_present.count(m_type) / length

        records.append(record)

    counts = Counter()
    for idx in keep:
        sequence = str(df.loc[idx, "sequence"]).upper().replace("T", "U")
        counts.update(sequence[i:i + 3] for i in range(len(sequence) - 2))
    total = max(1, sum(counts.values()))
    kmer = pd.DataFrame(
        [{("k3_" + k): counts.get(k, 0) / total for k in TRINUCLEOTIDES} for _ in keep]
    )

    base = pd.DataFrame(records).fillna(0)
    features = pd.concat([base.reset_index(drop=True), kmer.reset_index(drop=True)], axis=1)
    return features, keep


def filter_features(features):
    """Two unsupervised filtering steps that do not use the target variable."""
    keep_variance = [
        c for c in features.columns
        if features[c].value_counts(normalize=True).iloc[0] < 0.99
    ]
    features = features[keep_variance]

    correlation = features.corr(method="spearman").abs()
    upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] >= 0.95)]
    return features.drop(columns=to_drop)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="transcript-level input table")
    parser.add_argument("--output", default=os.path.join("data", "feature_matrix.csv"))
    args = parser.parse_args()

    if args.input.lower().endswith((".xlsx", ".xls")):
        raw = pd.read_excel(args.input)
    else:
        raw = pd.read_csv(args.input)

    raw = raw.dropna(subset=["sequence"])
    raw = raw[raw["T"] > 0].reset_index(drop=True)
    raw["T"] = winsorize(raw["T"], limits=[0.1, 0.1])

    features, keep = build_features(raw)
    features = filter_features(features)
    target = np.log1p(raw.loc[keep, "T"]).reset_index(drop=True)

    out = pd.DataFrame({"transcript_index": keep})
    for column in ("Accession", "gene symbol", "tx_id"):
        if column in raw.columns:
            out[column] = raw.loc[keep, column].values
    out["half_life_h"] = raw.loc[keep, "T"].values
    out["y"] = target.values
    out = pd.concat([out, features.reset_index(drop=True)], axis=1)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out.to_csv(args.output, index=False)
    print("wrote %s with %d rows and %d features" % (args.output, len(out), features.shape[1]))


if __name__ == "__main__":
    main()
