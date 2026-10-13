# RNA modification annotations and mRNA half-life in HepG2 cells

Analysis code for the manuscript *RNA modification annotations add predictive
information beyond sequence features for mRNA half-life in HepG2 cells*.

This repository contains the code used for feature construction, model
training, the SHAP analysis and the ablation analysis. Input data, processed
feature matrices, train/test splits and result files are not distributed here.

## Contents

```
code/
  config.py               shared settings (LightGBM parameters, seed, feature groups)
  feature_extraction.py   builds the 96-feature matrix from the input table
  train_models.py         Model A vs Model B on the held-out test set
  shap_analysis.py        TreeSHAP values and the mean absolute SHAP ranking
  ablation.py             seven feature-group configurations under 10-fold CV
```

## Data

The transcript-level input table, the processed feature matrix and the
train/test split are available from the corresponding author on reasonable
request.

The input table combines three sources. mRNA decay rates measured in HepG2
cells were taken from Supplemental Table 9 of Yang et al. (*Genome Research*
13:1863, 2003). Full-length transcript sequences were taken from Ensembl
release 116 (GRCh38), using the Ensembl canonical transcript of each gene.
RNA modification sites were compiled at transcript level.

The feature matrix expected by the scripts has one row per transcript and the
following columns.

| column | description |
| --- | --- |
| `transcript_index` | row index in the input table |
| `Accession` | GenBank accession of the decay-rate measurement |
| `gene_symbol`, `tx_id` | gene symbol and Ensembl transcript identifier |
| `half_life_h` | half-life in hours, winsorised at the 10th and 90th percentiles |
| `y` | regression target, `ln(1 + half_life_h)` |
| remaining 96 columns | the features listed in `feature_list.csv` |

A companion file `feature_list.csv` lists the 96 retained features together
with their category and their group in the ablation analysis.

Sequence features are transcript length, GC content, the longest homopolymer
run of each nucleotide, nucleotide frequencies and 3-mer frequencies.
Modification features are the number of annotated modification types, the site
density of each modification type expressed as sites per nucleotide, and, for
each type, the number of its sites that lie within 10 nucleotides of a site of
another type.

## Running the code

```
pip install -r requirements.txt

# point the scripts at the directory holding feature_matrix.csv and feature_list.csv
export HL_DATA_DIR=/path/to/feature_matrix          # PowerShell: $env:HL_DATA_DIR="..."
export HL_RESULTS_DIR=/path/to/output               # optional, defaults to ./results

python code/feature_extraction.py --input input_table.xlsx --output feature_matrix.csv
python code/train_models.py    # Model A vs Model B on the test set
python code/shap_analysis.py   # TreeSHAP values and the feature ranking
python code/ablation.py        # the seven configurations under 10-fold CV
```

`code/train_models.py` and `code/shap_analysis.py` use `train_test_split` with
`test_size=0.2` and `random_state=42`, which reproduces the split used in the
manuscript. `code/ablation.py` uses a single `KFold` with `n_splits=10`,
`shuffle=True` and `random_state=42`, shared by all seven configurations. All
models use the LightGBM settings listed in `code/config.py`.

`feature_extraction.py` expects an input table with the columns `sequence`,
`T` and `annotation`, where `annotation` stores modification sites as
`pos(type);pos(type);...` with 1-based positions on the spliced transcript.

## Environment

Python 3.12 with `lightgbm`, `shap`, `scikit-learn`, `pandas`, `numpy` and
`scipy`. The versions used for the final analysis are documented in the
manuscript.

## Licence

A licence still has to be chosen. MIT is a common choice for analysis code.
