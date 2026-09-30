# CPU XGBoost cost pilot — 2026-09-21

The author restored the primary benchmark and factorial estimators to kNN,
XGBoost, and MLP. MLP continues in the existing two-GPU benchmark queue.
EHreact keeps its separate kNN/logistic protocol. Existing linear-probe results
are preserved, but the main benchmark report and factorial ANOVA now select
the configured estimators.

XGBoost uses CPU histogram training, depth 6, learning rate 0.1, a maximum of
500 boosting rounds, and early stopping after 20 rounds without validation
improvement. The previous 200-class exclusion is disabled, including for TPL.

## Measured cost

All probes used 16 CPU threads, default SynRFP (1024-bit WL2 parity delta),
and the complete cached CV seed-0 training/validation/test partitions.
No dataset subsampling was introduced. MLP continued on both GPUs during
measurement. Times below cover fitting, not loading features.

| Dataset | Classes | Training rows | Measured rounds | Fit time | Projected 500-round fit |
|---|---:|---:|---:|---:|---:|
| USPTO-50k unbalanced | 10 | 30,005 | 500 | 18.28 s | 18.28 s, observed |
| Schneider unbalanced | 50 | 30,636 | 20 | 3.18 s | About 70 s |
| TPL unbalanced | 1,000 | 267,295 | 3 | 44.48 s | About 119 min |

TPL took 14.28, 14.41, and 14.24 seconds per round; peak process resident
memory was about 7.03 GiB. The projection assumes later rounds cost the
median of the measured rounds after the first. It is **not a completed-fit
runtime or a queue ETA**. Early stopping can reduce the number of rounds;
later tree growth and different fingerprints can change per-round cost.

With the default multiclass strategy, each boosting round builds one tree
per class: TPL can produce up to 500,000 trees per fit at the configured cap.
See the [XGBoost parameter documentation](https://xgboost.readthedocs.io/en/stable/parameter.html).

## Workload implications

- Available-input benchmark: 2,022 XGBoost fits, including 336 across TPL
  balanced and unbalanced (2 datasets × 6 fingerprints × 28 splits/seeds).
- Available-input factorial: 3,888 XGBoost fits, with a 50,000-row cap per
  dataset. These require separate timing; the full-dataset TPL estimate does
  not apply to the factorial subsample.
- No full XGBoost queue has been launched. A sensible next step is to schedule
  smaller datasets first and budget TPL separately. Do not silently reduce
  folds, trees, or datasets based on these short probes.
- Budgeting concurrent CPU workers must leave CPU and memory headroom for
  the running GPU MLP workers and their input preparation.

## Reproduction and validation

The script is `xgb_cost_pilot.py`; raw results and logs are under
`results/xgb_cost_pilot/`. Example:

```bash
OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 \
  ~/anaconda3/envs/synrfp/bin/python Experiments/03_classification/xgb_cost_pilot.py \
  --dataset tpl_ub --rounds 3 --seconds 120 \
  --out Experiments/03_classification/results/xgb_cost_pilot/tpl_ub.json
```

The time guard is checked between rounds, so a long single round can exceed
the requested duration. Partial models are never written to benchmark result
paths. The USPTO probe completed the full protocol, but also remains isolated.

`test_xgb_protocol.py` passed: actual 201-class training with the limit disabled,
original-label restoration, optional finite class limit, and rejection of a
validation set containing no training classes. Modified scripts also passed
syntax compilation and `git diff --check`.
