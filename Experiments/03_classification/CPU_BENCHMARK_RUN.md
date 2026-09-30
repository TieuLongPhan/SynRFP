# CPU benchmark sequence — 2026-09-21

Started at the user's request: finish benchmark kNN, then automatically start
benchmark XGBoost. No CPU benchmark process was running before this launch.

| Stage | Available fits | Already complete at launch | Queued |
|---|---:|---:|---:|
| kNN | 1,874 | 319 | 1,555 |
| XGBoost | 2,022 | 0 | 2,022 |

Expanded at the user's request to **two kNN workers, 16 threads each** around
00:58 on 2026-09-21. Dataset assignments are disjoint and balanced using
remaining-fit count × dataset rows squared, an approximate kNN cost.
Completed fits are retained; the in-flight fit was restarted during transition.
The first session's state/logs are archived under `single_worker_session/`.

XGBoost remains **one 16-thread worker**, starting only after both kNN workers
exit successfully and all expected kNN results are validated. MLP continues
on its separate GPU queue. XGBoost keeps the configured
500-round cap, depth 6, learning rate 0.1, validation early stopping, per-fit
seeds, and no class-count exclusion. Factorial is outside this sequence.

`cpu_benchmark_queue.py` launches `classify.py --study benchmark --models knn`
with a separate dataset list for each worker, waits for both to exit successfully,
and verifies every expected available-input
kNN result before launching the equivalent XGBoost command. A process failure
or missing/invalid result blocks the next stage. Missing datasets/features are
listed separately in the inventory; they are not treated as completed fits.

The detached supervisor continues when the interactive session closes.
All existing result files are retained. New results use the normal benchmark
paths, so the existing classifier can resume after interruption.

## Monitoring

Under `results/cpu_benchmark_queue/`:

- `status.json`: active stage, PIDs, completed/remaining counts, updated every
  15 seconds while the child process is running.
- `knn_worker0.log` and `knn_worker1.log`: current kNN starts, completions, and errors.
- `knn.log`: historical single-worker log.
- `xgb_worker0.log`: created when the XGBoost stage starts.
- `queue.log`: stage transitions and failures.
- `inventory.json`: initial counts and unavailable inputs.
- `config.yaml`: pinned configuration with absolute input paths and 16 threads.
- `launch.json`: exact detached launch command and initial PID.

The control-directory lock prevents a second copy of this supervisor from
running against the same control folder. Do not run another CPU classifier
for the same model/splits concurrently. The GPU MLP queue writes different
model result files and continues independently.

To resume after the existing supervisor and child have exited:

```bash
OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
  ~/anaconda3/envs/synrfp/bin/python -u Experiments/03_classification/cpu_benchmark_queue.py \
  --threads 16 --knn-workers 2 --control Experiments/03_classification/results/cpu_benchmark_queue
```

The full XGBoost stage includes TPL, whose CPU cost can be substantial; see
`XGB_COST.md`. This sequence does not reduce folds, rounds, or dataset sizes.
