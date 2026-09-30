# SynRFP experiments — plan, decisions and handoff

Single source of truth for the benchmark (replaces `SESSION_LOG.md`).
Last updated 2026-09-18. Branch `exp/reproducible-benchmark` (from `dev-lo`);
nothing pushed. Anything marked **VERIFY** has not been checked against the
primary source and must be before it enters the paper.

---

## 1. State of the repository

### Commits on the branch
- `fbf564d` CWS/SRP cross-reaction consistency fix, b-bit encoding,
  `fingerprint_from_tokens`, WL token provenance, agents handling, tests.
- `e2260ef` `Experiments/` suite (01–09 + external/rxnfp).

### Uncommitted (working tree, reviewed, lint + 115 tests pass)
- SynRXN loaded through the `synrxn` package, pinned v1.2.1
  (`config.yaml → synrxn`); datasets use `name:` instead of `file:`.
- Per-scheme seeds (`benchmark.schemes`) and `scheme_seeds()`.
- kNN/linear/MLP train on train+val (only xgb uses val); kNN runs once on the
  fixed published split.
- SynTemp R1/R2 added as datasets.
- `10_enzyme_activity/activity.py` (EHreact case study).
- RXNFP env reads CSV exports (`work/export/`) instead of pickles.
- `Experiments/environment.yml` (env `synrfp-bench`).
- Package cleanup: dead `synrfp/encoder.py` + test removed, `GraphData` =
  `Molecule` alias, Sphinx docstrings, `synrfp.__version__`, new
  `pyproject.toml`/`requirements.txt`/`env.yml`, `scripts/{lint,pytest,build_doc}.sh`
  + file-size/docstring/package checks, CI rewritten.
- README figure `doc/figure/synrfp.{tex,pdf,png}` (graphical abstract).
- **Review fixes (this session):**
  - `synrfp/tokenizers/map.py` (MAP tokenizer — user's own work; commit
    separately): with `edge_attrs=[]` the bond label hashed node ids, so tokens
    changed with atom numbering → constant label now. Isolated atoms (ions,
    water, halides) produced no tokens (`[Cl-]>>[Br-]` was all zeros) → one
    `("map_atom", r, env)` token per radius. Bonded atoms unchanged. Tests in
    `tests/tokenizers/test_map.py`.
  - `synrfp_exp/data.py::check_data_source()` writes
    `work/tables/SOURCE.json` and raises when cached tables come from another
    SynRXN source/version or predate the stamp. Called from `reaction_table`,
    `dataset_table`, `activity.py`. **An old `work/` from e2260ef stops at
    step 01 on purpose:** delete `work/{tables,splits,features,export}`.
  - `Experiments/README.md` data section updated.

### Environments
- `synrfp-bench` (Python 3.11): SynRFP (editable), synrxn 1.2.1, synkit
  1.6.2, DRFP 0.3.7, bblean, stats. DRFP does not conflict with SynRFP.
- `rxnfp` (`external/rxnfp/environment.yml`, Python 3.8): the only
  conflicting baseline. Same branch, separate env; reads
  `work/export/*.csv.gz` and `work/splits/*.npy`, writes `.npz` features.
  Not installed on this machine; its scripts compile but were never executed.

---

## 2. Data (SynRXN 1.2.1)

| dataset(s) | n | classes | split in SynRXN | notes |
|---|---|---|---|---|
| schneider_u / _b | 50,000 | 50 | train 10,000 / test 40,000 | same split as SynCat (checked row by row for _b) and RXNFP |
| tpl_u / _b | 445,115 | 1,000 | train 400,604 / test 44,511 | train = RXNFP's `bert_class_1k_tpl` training set (400,000 + 604 val) |
| uspto50k_u / _b | 50,016 | 10 | train 45,014 / test 5,002 | |
| ecreact_1/2/3 | 185,734 | 7 / 63 / 175 | train 166,918 / test 18,816 | from CLAIRE; test n equals CLAIRE's |
| syntemp_r0/r1/r2 | 43,441 | 143 / 356 / 680 | **none** | deterministic stratified 8:1:1 Track B split, seed 42 |
| mech_1 / mech_2 | 31,673 | 9 / 63 | none (not in SynRXN) | public SynCat Mech31k export; source `labels_lv1` supplies level 1 |
| ehreact_activity | 1,943 rows | binary per assay | none | small-data SI case study |

- Validation: 10 % of train, stratified, for every published split.
- **SynTemp and Mech Track B split:** use the configured stratified 8:1:1
  split with seed 42. Label these Track B results as our reference points only,
  rather than direct replications of an external published split.
- Classes with < 10 records are removed (logged in `work/tables/ds_<name>.json`).
- Zenodo bug in `synrxn`: `DataLoader(source="zenodo")` fails with HTTP 400
  (`resolve_record_id` requests `size=200`, Zenodo allows 25 without a
  token). Workaround: `synrxn.record_id: 22045145`. Fix upstream in SynRXN.
- EHreact: 9–65 reactions per assay, 199 duplicated (system, assay, rxn)
  rows (grouped in CV). Use assays with ≥ 10 active and ≥ 10 inactive
  (3 tasks: csp2–csp3 coupling Class, alcohol dehydrogenase E-2,
  dehalogenase E-1). **`config.yaml` still has `min_positives: 5` (14 tasks)
  → set to 10.**

---

## 3. Protocol

Two classification tracks, never mixed in one table or test.

### Track A — fingerprint comparison (statistics)
- Grouped **5×5 CV** on all 14 datasets: `StratifiedGroupKFold` on the
  Murcko scaffold of the largest product, 5 repeats (shuffle seed r) × 5
  folds; fold f → test, fold f+1 → validation, rest → train.
- Fingerprints (fixed, not trained on the target labels):
  `synrfp_default` (WL r2, 1024-bit parity, delta), `drfp` (2048, r3),
  `rdkit_struct`, `rdkit_diff`, `rxnfp` (= `bert_ft`), `rxnfp_mlm`
  (= `bert_pretrained`, label-free control).
- Models: 5-NN (Tanimoto for bits, cosine otherwise), XGBoost (CPU histogram,
  multiclass objective), MLP (512 hidden, GPU). XGBoost restored to the primary
  comparison on 2026-09-21 at the author's request; no class-count exclusion.
  Earlier linear-probe results are retained separately from the configured comparison.
- Statistics per dataset, see §6.
- If TPL/ECReact MLP runs are too slow: 5×2 or 3×5 CV for those four
  datasets, stated in the methods.

### Track B — reference comparison on one fixed split (no significance test)
- Split: SynRXN `split` column; SynTemp and Mech fixed stratified 8:1:1, seed 42.
- Our six fingerprints with the same heads: kNN once, XGBoost/MLP 3 seeds
  on the same fixed partition. With full row/column sampling, XGBoost seed
  repeats can be identical; these are reference runs, not independent data replicates.
- RXNFP fine-tuned rows, **no fine-tuning of our own**: use the published
  dataset-specific encoders where their training split equals ours —
  `bert_class_1k_tpl` for tpl_u/b, `bert_ft_10k_25s` for schneider_u/b.
  Other datasets get no fine-tuned RXNFP row.
- Trained models (published numbers where the test set is identical; one
  retraining run only where a dataset is missing):
  - SynCat: published on the 9 shared datasets (§5); retrain once on
    ecreact_1/2/3 and mech_1/2 (5 GPU runs) or skip those.
  - CLAIRE: published weighted F1 on ECReact; no retraining.
  - T5Chem: optional, 14 runs, heavy.
- Report with hyperparameters taken from each paper, stated as such.

### Other studies
| # | study | scope | protocol |
|---|---|---|---|
| 04 | factorial + ANOVA | 162 SynRFP configs (tokenizer wl/morgan/path × radius 1–3 × bits 512/1024/2048 × sketch parity/minhash/cw × mode delta/union), 10 datasets (≤ 50k subsample) | 5 matched random splits; kNN, XGBoost, MLP ×5 seeds |
| 05 | clustering | 7 datasets | full data, no split; see below |
| 06 | retrieval | 5 datasets × 6 fingerprints, k = 1/5/10 | published split (random seed 0 otherwise) |
| 07 | robustness | balanced/unbalanced pairs, SMILES randomisation, reversal, resolution (bits, sketch, mode, radius) | as in `config.yaml → robustness` |
| 08 | reaction centre | mapped Mech reactions, radius 3 | share of signed-delta tokens on the mapped centre |
| 10 | EHreact activity | 3 assays × 6 fingerprints × kNN/logistic | 5 × stratified group 5-fold CV, pooled OOF: MCC, AP, ROC-AUC |

### Clustering (05)
- Already implemented: k-means (k = #labels, 10 external seeds with 10 internal
  restarts per fit), BitBIRCH (bblean, Tanimoto threshold sensitivity sweep
  0.05–0.90, bit fingerprints only), HDBSCAN
  (PCA-50, min_cluster_size 5/10/25/50).
- Metrics: ARI, AMI, NMI, homogeneity, completeness, V-measure,
  Hungarian-matched MCC (extra clusters count as errors), n_clusters,
  singleton and largest-cluster fraction — against every hierarchy level
  (SynTemp R0/R1/R2, Mech 1/2) → granularity profile.
- Unsupervised, so no train/test split and no 5×5. K-means seed variation only
  measures optimizer initialization on a fixed dataset. For population-level
  uncertainty, use paired group resampling shared across fingerprints; do not
  describe a small collection of overlapping 80% subsamples as a bootstrap CI.
- Fingerprints: label-free only — `synrfp_default`, `drfp`, `rdkit_struct`,
  `rdkit_diff`, `rxnfp_mlm`. BitBIRCH uses only the three binary fingerprints.
  `bert_ft` saw class labels (flag it if kept). Never cluster
  SynCat/CLAIRE embeddings (trained on the labels being scored → circular).
- **Bug to fix:** the BitBIRCH threshold is chosen by AMI on `uspto50k_b`,
  which is also reported in `bitbirch_frozen.csv` → that row is label-tuned.
  Mark or exclude it, or calibrate on a dataset that is not reported.

---

## 4. RXNFP: which weights

The PyPI wheel `rxnfp==0.1.0` ships only `bert_pretrained`, `bert_ft` and
`bert_ft_10k_25s` (in `rxnfp/models/transformers/`; checked by listing the
wheel). `bert_mlm_1k_tpl` and `bert_class_1k_tpl` exist only in the GitHub
repo and must be downloaded into that folder
(<https://github.com/rxn4chemistry/rxnfp/tree/master/rxnfp/models/transformers>):

| folder | trained on | labels? | use |
|---|---|---|---|
| `bert_pretrained` | masked-LM only | no | `rxnfp_mlm`, Track A/B, clustering |
| `bert_mlm_1k_tpl` (GitHub only) | masked-LM on USPTO 1k TPL | no | not used |
| `bert_ft` (package default) | fine-tuned on reaction classes (Pistachio per our README — **VERIFY**) | yes | `rxnfp`, Track A/B |
| `bert_class_1k_tpl` (GitHub only) | `bert_mlm_1k_tpl` fine-tuned on TPL templates, 400,000 train + 604 val = our TPL train split, 5 epochs | yes (TPL) | Track B TPL only |
| `bert_ft_10k_25s` | fine-tuned on the 10k Schneider training reactions ("RXNFP ft 10k") = our Schneider train split | yes (Schneider) | Track B Schneider only |

- Decision: no fine-tuning by us anywhere; drop `rxnfp_ft` and
  `rxnfp_finetune.py` from the plan.
- Do not use `bert_class_1k_tpl`/`bert_ft_10k_25s` outside their own dataset:
  TPL is a USPTO superset, so USPTO-50k/Schneider test reactions may be in
  their training data.
- **Check before the run** (rxnfp env): compare `bert_ft` and
  `bert_class_1k_tpl` encoder weights (max abs diff of
  `embeddings.word_embeddings.weight`). If identical, `bert_ft` was trained on
  TPL labels — say so and rely on `rxnfp_mlm` as the fair row.
- Disclose contamination: the SynCat paper measured 72.58 % (Schneider
  unbalanced) and 99.97 % (TPL unbalanced) of test reactions in RXNFP's
  training corpus; this favours RXNFP.
- RXNFP docs, USPTO 1k TPL: BERT classifier acc 0.9893 / MCC 0.9893;
  fine-tuned FP + 5-NN 0.989; MLM FP + 5-NN 0.340.

---

## 5. Published results for the reference table (Track B)

**Direct** (same test set; we only compute our side):

SynCat (Van Nguyen et al. 2025, *Digital Discovery*, DOI 10.1039/D5DD00367A).
Values from `../SynCat/paper/old_syncat/SI.tex` (table
`s_tab:results_SynCat_only_GINE`) — **VERIFY** against the published version.
Single run per dataset.

| model | Schn. U | Schn. B | TPL U | TPL B | 50k U | 50k B | R0 | R1 | R2 |
|---|---|---|---|---|---|---|---|---|---|
| SynCat acc | 0.981 | 0.973 | 0.991 | 0.988 | 0.995 | 0.996 | 0.996 | 0.992 | 0.982 |
| SynCat MCC | 0.980 | 0.973 | 0.991 | 0.988 | 0.995 | 0.995 | 0.996 | 0.992 | 0.978 |
| RXNFP+MLP MCC (their run) | 0.985 | 0.956 | 0.988 | 0.971 | 0.993 | 0.989 | 0.994 | 0.954 | 0.758 |
| DRFP+MLP MCC (their run) | 0.956 | 0.925 | 0.977 | 0.962 | 0.898 | 0.910 | 0.971 | 0.958 | 0.957 |

Splits verified identical: schneider_b (row by row), syntemp (row-aligned
file with its own splits). **VERIFY** schneider_u, tpl_u/b, uspto50k_u/b.

CLAIRE (Zeng et al. 2025, *J. Cheminform.*, DOI 10.1186/s13321-024-00944-8):
weighted-average F1 on the ECReact test set (n = 18,816), one model per EC
level, single run; input = rxnfp (256) + DRFP (256).

| model | EC 1st | EC 2nd | EC 3rd |
|---|---|---|---|
| CLAIRE | 0.958 | 0.944 | 0.861 |
| Theia (ECREACT-trained, from CLAIRE paper) | 0.387 | 0.333 | 0.228 |

**VERIFY** the test reactions are identical (via `orig_index`), not only n.

**Indirect** (different split or protocol; context only, never in a table
with direct numbers):
- Rxn Hypergraph (Tavakoli et al. 2022, arXiv 2201.01196): 0.928 accuracy on
  a Schneider-style resampled split (200 train / 800 test per class).
- RXNFP BERT classifier (Schwaller et al. 2021, *Nat. Mach. Intell.*, DOI
  10.1038/s42256-020-00284-w): TPL 0.989 — direct if the TPL test set is
  ours (train split already matches).
- T5Chem (Lu & Zhang 2022, *JCIM*, DOI 10.1021/acs.jcim.1c01467):
  USPTO_TPL classification — number and split **VERIFY**.
- SynRXN paper (arXiv 2601.01943): DRFP + random forest baselines (weighted
  F1, MCC) on the same splits — sanity check for our DRFP rows. Numbers came
  from an automated summary (**VERIFY**).

---

## 6. Metrics and statistics

- Classification: MCC (primary), accuracy, macro-F1, **weighted-F1 (add to
  `synrfp_exp/metrics.classification`; CLAIRE and SynRXN report it)**,
  balanced accuracy.
- Statistics follow Ash et al. 2025, "Practically Significant Method
  Comparison Protocols for ML in Small Molecule Drug Discovery", *JCIM* 65,
  9398–9411, DOI 10.1021/acs.jcim.5c01609 (full text not accessible here —
  **VERIFY** details against the paper):
  - 5×5 repeated CV per dataset (Track A).
  - Per dataset: repeated-measures ANOVA (or Friedman) over folds, then
    Tukey HSD.
  - Effect size = mean MCC difference vs SynRFP with simultaneous 95 % CI.
  - Practical-significance threshold fixed in `config.yaml` before the run
    (proposal ΔMCC = 0.01 — author decides).
  - Plots: extended sign plot (significance + effect size) in the main text,
    CI plot of pairwise differences in the SI.
  - No bold/underline "best" tables (`09_report/report.py:latex_table` does
    this now) → mean ± CI with better/equivalent/worse marks.
  - Friedman + Nemenyi CD diagrams and Holm–Wilcoxon only as an
    across-dataset summary in the SI (current implementation).
- Track B and published numbers are single runs: reference points, no tests.

---

## 7. Run budget (6 fingerprints incl. `rxnfp_mlm`)

| block | formula | runs |
|---|---|---|
| A: 5×5 CV | 14 datasets × 25 splits × 6 fp × 3 models | 6,300 |
| B: our fingerprints | 14 × 6 fp × (1 kNN + 3 XGBoost + 3 MLP) | 588 |
| B: RXNFP dataset-specific encoders | 4 datasets × 7 | 28 |
| B: SynCat retrain | ecreact ×3 + mech ×2 (or all 14) | 5 (14) GPU |
| B: T5Chem (optional) | 14 | 14 GPU |
| 04 factorial | 10 × 162 × 5 splits × 3 models | 24,300 |
| 05 clustering | 7 × [4 fp × (3 k-means + 4 HDBSCAN) + 3 bit fp × 13 BitBIRCH] | 469 |
| 06 retrieval | 5 × 6 | 30 |
| 10 EHreact | 3 tasks × 6 fp × 2 models × 5 repeats | 180 |
| **total** | | **≈ 18,900 CPU runs + 5–28 GPU trainings** |

Featurisation is once per reaction table. RXNFP embedding = inference only
(5 encoders at most, all reaction tables). MLP now runs on GPUs. XGBoost on
TPL (400k × 1,000 classes) is a potential CPU bottleneck; measure its cost
before scheduling the full workload. CPU timing probes are isolated in
`03_classification/results/xgb_cost_pilot` and are not benchmark results.

---

## 8. To do (in order)

1. ✅ Done (bundled into `877984d "exp"`, not a separate commit as originally
   planned). MAP tokenizer: `synrfp/tokenizers/map.py`, `core.py` hook,
   `tests/tokenizers/test_map.py`.
2. ✅ Done. Config: `benchmark.fingerprints` →
   `synrfp_default, drfp, rdkit_struct, rdkit_diff, rxnfp, rxnfp_mlm`;
   `rxnfp_ft` removed; `rxnfp_pretrained` → `rxnfp`; `activity.min_positives:
   10`; `clustering.fingerprints` → `synrfp_default, drfp, rdkit_struct,
   rxnfp_mlm`; `statistics.practical_threshold: 0.01` added (author should
   confirm the value).
3. ✅ Done, **not run** (no `rxnfp` env on this machine; `bert_class_1k_tpl`/
   `bert_ft_10k_25s` are not downloaded either). `external/rxnfp/rxnfp_embed.py`
   rewritten to write `rxnfp` (`bert_ft`), `rxnfp_mlm` (`bert_pretrained`),
   `rxnfp_tpl` (`bert_class_1k_tpl`, tpl_ub/tpl_b only), `rxnfp_schneider`
   (`bert_ft_10k_25s`, schneider_ub/schneider_b only; dataset scoping via the
   new `rxnfp_dataset_fingerprints` config section), plus `--check-weights`
   for the `bert_ft` vs `bert_class_1k_tpl` diff. `rxnfp_finetune.py` deleted;
   `rxnfp_ft` paths removed from `classify.py`/`features.py`.
4. ✅ Done. `synrfp_exp/metrics.py::classification` now returns `weighted_f1`.
5. ✅ Done. `synrfp_exp/splits.py`: `published`/`random`/`grouped` replaced by
   `cv` (Track A, seed = repeat*5+fold, 25 seeds) and `fixed` (Track B).
   `common.py`'s dead copy of the old `scheme_seeds`/`split_codes` removed
   (only used by the now-deleted finetune script). `06_retrieval`,
   `07_robustness`, `01_data/audit_data.py` updated to the new scheme names.
6. ⚠️ Code done, **data blocked**: `synrfp_exp/data.py::syncat_syntemp_split`
   reads SynCat's `Split_R{0,1,2}` (new `paths.syncat_syntemp_split`,
   row-aligned per §2) and `splits.py::fixed_split` uses it for
   syntemp_r0/r1/r2. No `../SynCat` checkout exists on this machine, so this
   raises `FileNotFoundError` until one is provided; `classify.py`/
   `01_data/prepare_data.py` catch it and skip just that (dataset, scheme)
   pair rather than crashing the run.
7. ✅ Done, **statistics unverified** (Ash et al. full text still not
   accessible — VERIFY before results enter the paper). `09_report/report.py`
   rewritten: per-dataset one-way RM-ANOVA (subject = cv seed) + Tukey HSD
   simultaneous CIs vs `synrfp_default` (folds/repeats treated as independent
   replicates for the post-hoc step — not exact repeated-measures Tukey),
   practical-significance marks (+/=/-), extended sign plot (`sign_<model>.pdf`)
   and CI plot (`ci_<model>.pdf`, SI); `plain_latex_table` drops bold/underline
   in favour of mean ± SD + mark; Track B is a separate, unmarked reference
   table (`track_b_<model>.csv/.tex`) plus the literal SynCat/CLAIRE numbers
   from §5 as `track_b_reference_{syncat,claire}.csv` (still VERIFY).
   Friedman/Nemenyi/Holm-Wilcoxon kept as the SI cross-dataset summary.
8. ✅ Done (marked, not excluded/recalibrated). `bitbirch_frozen.csv` now has
   a `calibration_leak` column, `True` for the calibration dataset's own row
   (a `log.warning` is also emitted); every other row is unaffected.
9. ❌ Not done, **blocked**: needs `../SynCat/paper/old_syncat/SI.tex` and
   `../SynRXN`, neither present on this machine.
10. ❌ Not done (author): `data/mech_level1_map.csv` — only the DRAFT exists.
11. ❌ Not done: `Experiments/README.md` still describes the old scheme design.
12. ⚠️ Workaround only (`synrxn.record_id` pin in `config.yaml`, pre-existing);
    the upstream `synrxn` fix is not done.
13. ❌ Not started: no smoke test or full run (needs SynRXN data, not
    downloaded on this machine as part of this session).
14. ❌ Not started (paper repo, separate branch).

Session note (2026-09-18, this pass): a `synrfp` conda env was built from
`Experiments/environment.yml` + `rdkit` (`env.yml`'s only addition) and
`pip install -e ".[dev]"`; versions satisfy `pyproject.toml`; 115 tests pass.
Items 2–8 above were implemented and smoke-tested against synthetic/in-memory
data (no real SynRXN table was downloaded, so none of the actual experiment
scripts have been run end-to-end yet — that's item 13).

---

## 9. How to run

```bash
conda env create -f Experiments/environment.yml
conda activate synrfp-bench && pip install -e .
bash Experiments/run_all.sh                     # restartable; skips finished runs
# RXNFP (separate env; after 01_data and the activity export):
conda env create -f Experiments/external/rxnfp/environment.yml
conda run -n rxnfp python Experiments/external/rxnfp/rxnfp_embed.py
```

---

## 10. Sources

- Ash et al. 2025, JCIM: https://pubs.acs.org/doi/10.1021/acs.jcim.5c01609
- CLAIRE: https://pmc.ncbi.nlm.nih.gov/articles/PMC11707929/
- SynRXN: https://arxiv.org/html/2601.01943v1
- RXNFP docs: https://rxn4chemistry.github.io/rxnfp/ (fine-tuning on 1k TPL,
  results 1k TPL, results Schneider 50k)
- RXNFP weights: https://github.com/rxn4chemistry/rxnfp/tree/master/rxnfp/models/transformers
- Rxn Hypergraph: https://arxiv.org/pdf/2201.01196
- T5Chem: https://pubmed.ncbi.nlm.nih.gov/35266390/
- SynCat numbers: `../SynCat/paper/old_syncat/SI.tex`
