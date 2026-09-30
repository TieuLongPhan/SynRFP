# Isolated MLP GPU pilot

`mlp_pilot.py` separates MLP evaluation from the normal classification outputs.
It is deliberately scoped to `schneider_ub` and
`synrfp-wl-r2-b1024-parity-delta`, using the existing 25 Track A splits
(five repeats of five product-scaffold-grouped folds). It does not change
the main benchmark configuration or replace completed results.

## Matched training protocol

- The existing classifier merges outer train and validation rows, then reserves
  an internal random 10% validation set. The pilot preserves this behavior.
- Float32, 512 ReLU hidden units, multiclass softmax, Adam at learning rate
  0.001, batch size 256, alpha 0.0001, maximum 200 epochs.
- Early stopping monitors validation accuracy with tolerance 0.0001 and
  patience 10; best validation weights are restored.
- Preparation uses the installed sklearn initializer and exports its RNG state
  after the validation split. GPU uses the same initial weights, internal split,
  and minibatch permutations as CPU.
- The GPU worker implements the sklearn gradient and Adam equations explicitly,
  including batch-normalized L2 on weights only and sklearn's epsilon placement.
  It does not substitute PyTorch's default Adam regularization behavior.
- TF32 is disabled. Small floating-point differences can still change convergence
  or stopping epochs. CPU and GPU versions and per-epoch validation scores are saved.

CPU uses sklearn in `synrfp` with four BLAS threads; CUDA uses the existing
PyTorch in `rxnfp` on GPU 1. No packages need to be installed. The GPU script
uses numpy and torch only, so the old sklearn in `rxnfp` is not used.
All final metrics are computed in `synrfp` using the project's metrics code.

## Commands (from repository root)

```bash
OUT=Experiments/03_classification/results/mlp_gpu_pilot
~/anaconda3/envs/synrfp/bin/python Experiments/03_classification/mlp_pilot.py prepare --out "$OUT"

OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  ~/anaconda3/envs/synrfp/bin/python -u Experiments/03_classification/mlp_pilot.py cpu --out "$OUT" > "$OUT/cpu.log" 2>&1

CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  ~/anaconda3/envs/rxnfp/bin/python -u Experiments/03_classification/mlp_pilot.py gpu --out "$OUT" > "$OUT/gpu.log" 2>&1

~/anaconda3/envs/synrfp/bin/python Experiments/03_classification/summarize_mlp_pilot.py "$OUT"
```

CPU and GPU commands can run concurrently. Each processes folds sequentially.
Use `--seeds 0` for a single fold. Completed folds are skipped on resume.
`--max-epochs` is intended for smoke checks in a separate prepared output directory.
Do not reuse results across changes to the implementation or configuration.

## Reading the results

`folds.csv` contains fold timings, epochs, and metrics. `repeats.csv` computes
metrics from all pooled out-of-fold predictions within each repeat; its five
rows per backend are the basis for reported means and standard deviations.
`summary.json` includes aggregate runtime and paired prediction agreement.

Timing covers per-fold data materialization/transfers, training, and test
prediction. Imports, reading prepared files, initial CUDA context setup, result
serialization, and metric computation are outside the timed region. The GPU
fit time includes transfers of test features; total time is the primary fair
comparison. This is single-worker throughput, not a comparison against using
all CPU cores for concurrent folds. First-fold timings include kernel/library
warm-up effects. The result should not be extrapolated directly to TPL or to
the complete mixed-model benchmark.

The `smoke/` directory records the initial one-epoch parity check separately
from the full early-stopped 5x5 experiment.

## Completed pilot, 2026-09-20

Both backends completed all 25 folds on 50,000 Schneider reactions.

| Measurement | CPU (4 threads) | GPU (one RTX A6000) |
|---|---:|---:|
| Total measured fit + prediction time | 273.47 s | 40.85 s |
| Total training epochs | 622 | 622 |
| Pooled-repeat MCC, mean ± SD | 0.975920 ± 0.000663 | 0.975920 ± 0.000663 |
| Pooled-repeat accuracy, mean ± SD | 0.976400 ± 0.000649 | 0.976400 ± 0.000649 |
| Pooled-repeat macro-F1, mean ± SD | 0.976329 ± 0.000657 | 0.976329 ± 0.000657 |

GPU was **6.70 times faster**, reducing measured compute time by 85.1%.
Every paired fold stopped at the same epoch count. Mean per-fold prediction
agreement was 99.99835%. The one-epoch smoke check had 100% prediction agreement.

Recommendation: use a separately identified GPU MLP backend for this workload.
The pilot supports a broader integration but does not establish the speedup on
other datasets, fingerprints, or against a fully parallel CPU scheduler.
For CPU-only models, the existing runner can be invoked with
`--models knn linear`; this pilot runs only MLP. The main benchmark defaults
have not been changed.
