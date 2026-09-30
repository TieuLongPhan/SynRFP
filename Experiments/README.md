# SynRFP experiments

Everything needed to regenerate the results of the SynRFP paper from the
SynRXN data. Each numbered folder is one study and keeps its script next to
its outputs in `results/`. Shared code lives in `synrfp_exp/`, settings in
`config.yaml`, and rebuildable caches in `work/` (git-ignored).

| Folder | Question | Paper section |
| --- | --- | --- |
| `01_data` | Dataset tables, all splits, and a data audit (duplicates, label conflicts, train/test leakage) | Experiments, Table 1 |
| `02_featurize` | Fingerprints: SynRFP (default and 162-point grid), DRFP, RDKit structural/difference, RXNFP (external) | – |
| `03_classification` | Supervised benchmark (3 split schemes × 5 seeds × 3 models) and the factorial runs | Reaction classification |
| `04_factorial` | ANOVA of the SynRFP design space (partial η², crossed radius × bits) | Factor analysis |
| `05_clustering` | k-means with k = #labels, plus k-free BitBIRCH and HDBSCAN, scored against every hierarchy level | Clustering |
| `06_retrieval` | Nearest-neighbour search by native similarity (P@k, mAP) per hierarchy level | Similarity search |
| `07_robustness` | Balanced/unbalanced stability and transfer, SMILES randomisation, reversal, fingerprint resolution | Robustness and resolution |
| `08_reaction_center` | Share of signed-delta tokens on the mapped reaction centre; class-enriched atom environments | Reaction-center localization |
| `09_report` | Tables (CSV + LaTeX), Friedman/Nemenyi and Wilcoxon–Holm statistics, CD diagrams, granularity plots | all |
| `external/rxnfp` | Pretrained and split-wise fine-tuned RXNFP embeddings (separate environment) | baselines |

## Setup

```bash
conda activate synrfp                  # Python >= 3.11
pip install -e ..                      # this SynRFP checkout (corrected sketches)
pip install bblean drfp statsmodels scikit-posthocs xgboost
```

SynRXN data are loaded with the `synrxn` package (`synrxn` section of
`config.yaml`): by default the pinned Zenodo release (v1.2.1, cached under
`work/synrxn_cache`), or a local checkout with `source: local`
(`local_dir`, default `../../SynRXN/Data`). The mechanism table still comes
from `paths.mech_raw`. Cached tables record their source in
`work/tables/SOURCE.json`; after changing the source or version, clear
`work/tables`, `work/splits`, `work/features` and `work/export`.

The public SynCat Mech31k export supplies both the 63-class mechanism labels
and its nine-class `labels_lv1` grouping. The benchmark reads this source from
`data/mech31k.csv.gz`; `data/mech_level1_map.DRAFT.csv` remains unused.

## Running

```bash
bash run_all.sh                        # full suite in order; restartable
# or step by step, e.g.
python 01_data/prepare_data.py
python 02_featurize/featurize.py --study benchmark --datasets schneider_b
python 03_classification/classify.py --study benchmark --datasets schneider_b --models knn
```

Every step caches its outputs and skips finished runs (one JSON per run), so a
crashed or interrupted job can be restarted with the same command. Scripts
accept `--datasets`, `--fingerprints`, `--models`, `--schemes`, `--seeds`
where meaningful, which makes it easy to spread the work over several machines.

RXNFP needs its own environment:

```bash
conda env create -f external/rxnfp/environment.yml
conda run -n rxnfp python external/rxnfp/rxnfp_embed.py       # pretrained
conda run -n rxnfp python external/rxnfp/rxnfp_finetune.py    # per split, GPU
```

Fine-tuning uses only the training (and validation) records of each split, so
the embeddings of test records carry no test-label information.

## Protocol decisions

* **Splits.** `published` (SynRXN test set; 10 % of train as validation),
  `random` (stratified 8:1:1) and `grouped` (StratifiedGroupKFold on the Murcko
  scaffold of the largest product) — five seeds each. Classes with fewer than
  10 records are removed and logged in `work/tables/ds_<name>.json`.
* **Models.** 5-NN uses Tanimoto similarity for bit vectors and cosine for
  embeddings. The primary comparison uses kNN, XGBoost (`multi:softprob`, CPU
  histogram training), and MLP (GPU queue). XGBoost was restored on 2026-09-21;
  the former 200-class limit is disabled. The linear probe remains optional,
  and EHreact retains its separate kNN/logistic protocol.
* **Factorial design.** Radius and bits are crossed (162 configurations), each
  with 5-NN, XGBoost, and MLP on five matched random splits. Datasets larger than
  50,000 records are stratified-subsampled to 50,000 for this study only.
* **Clustering without k.** The BitBIRCH threshold is chosen per fingerprint on
  `uspto50k_b` by AMI and then frozen for all other datasets; the per-dataset
  optimum is reported separately as a label-tuned upper bound.
* **Timing pilots.** GPU MLP and CPU XGBoost timing pilots are recorded separately
  under `03_classification/results/`; short XGBoost probes are not full results.

## Differences from the archived runs

The archived factorial results were produced with a SynRFP version in which

1. the CWS and SRP sketches drew random variables per reaction and indexed them
   by a reaction-local position, so signatures of different reactions were not
   comparable (two reactions without a shared token received identical CWS
   signatures);
2. MinHash/CWS signatures were folded into bits with `h % bits`, which lets
   samples collide;
3. radius and bit length were varied together (r=1/512, r=2/1024, r=3/2048),
   so their effects could not be separated; and
4. XGBoost used a binary objective on multiclass tasks.

All four are fixed here (see the commit history of `synrfp/sketchers` and the
tests in `tests/sketchers/test_consistency.py`). CWS and XGBoost results from
the archive should not be reused.

## Data notes to check

* `mech-USPTO-31k.csv` in SynRXN has 31,364 records and 63 classes; the paper
  table lists 31,673. The reaction column is configurable
  (`mech.reaction_column`, default `original_reactions`).
* SynRXN `ecreact.csv.gz` has 8 EC1 classes (the paper lists 7); the rare-class
  filter decides which remain and logs it.
* `syntemp.csv.gz` has no split column. Its Track B reference point uses the
  configured deterministic stratified 8:1:1 split (seed 42), rather than a
  supposedly comparable external published split.

## Smoke test

A small configuration with a few hundred reactions per dataset exercises every
script end to end in a few minutes; point `paths` (including `results_root`)
to a scratch directory so the real `results/` folders stay untouched.
