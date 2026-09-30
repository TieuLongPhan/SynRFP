# Benchmark MLP GPU queue

Launched 2026-09-20 at 23:37 Vietnam time on GPU 1, then expanded at the
user's request to **GPU 0 and GPU 1** (two RTX A6000s) around 23:42.
The first worker finished its active fit before the queue resumed with two
workers. All 14 newly completed fits were preserved; **1,662 fits remained**
at the two-GPU launch. Factorial work is still outside this queue.

At launch: 346 benchmark MLP results already existed; **1,676 fits remained**
with available inputs. MechFinder datasets and six TPL-specific RXNFP fits
remain unavailable. The queue starts with the largest datasets (TPL).

## Status and results

- `results/mlp_gpu_benchmark/status.json`: active job, finished/failed counts,
  remaining count, supervisor PID, and status. Updated at job boundaries.
- `results/mlp_gpu_benchmark/queue.log`: completed jobs and errors.
- `results/mlp_gpu_benchmark/gpu1.log`: epoch progress of the active fit.
- `results/mlp_gpu_benchmark/gpu0.log`: epoch progress on the second GPU.
- `results/mlp_gpu_benchmark/completed.jsonl`: timing history.
- `results/mlp_gpu_benchmark/inventory.json`: planned jobs and blocked inputs.
- `results/mlp_gpu_benchmark/launch.json`: exact command and original PID.
- `results/mlp_gpu_benchmark/single_gpu_session/`: archived first-session
  state and logs. Current status counts start from the two-GPU resume;
  the main completion log retains both sessions.
- `results/benchmark/`: normal per-fit result location. Existing CPU results
  are retained; new results explicitly record `backend: torch_cuda`, protocol,
  training parameters, source hash, torch version, GPU device, epochs, and timing.

The detached supervisor continues after the interactive session ends. It runs
one persistent worker per GPU and prepares inputs in reusable temporary directories
under `/tmp` (the root filesystem has substantially more room than `/home`).
Temporary arrays are removed after each fit and the directory at normal exit.
Completed fits are published atomically without overwriting existing results.
Failures are logged and remaining jobs continue where the worker remains usable.

## Resume

From repository root, after confirming no queue is already running:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  ~/anaconda3/envs/synrfp/bin/python -u Experiments/03_classification/mlp_gpu_queue.py \
  --gpu-python ~/anaconda3/envs/rxnfp/bin/python \
  --devices 0 1 --studies benchmark \
  --control Experiments/03_classification/results/mlp_gpu_benchmark
```

To stop scheduling after the active fit completes, create
`results/mlp_gpu_benchmark/STOP`. Remove it before resuming.
A control-directory lock rejects a second supervisor using the same directory.
Do not run the original classification script with MLP concurrently.

## Validation

The earlier Schneider 5x5 pilot showed 6.70x GPU speedup versus a four-thread
CPU worker, identical stopping epochs on every fold, and only four differing
predictions out of 250,000. Before launch, the general worker passed three-epoch
binary and multiclass checks with identical CPU/GPU validation histories and
100% test prediction agreement.

The general runner preserves the existing train+validation merge, train-only
standardization for real-valued fingerprints, sklearn weight initialization,
internal validation split, minibatch permutations, batch-normalized L2,
sklearn Adam epsilon placement, early stopping, and best-weight restoration.
TF32 is disabled. No mixed precision or larger batches were introduced.

The pilot speedup is not an ETA for the full queue. TPL has 445,115 rows and
1,000 classes, substantially larger than Schneider. Use completion timings
across datasets/fingerprints to update ETA once enough runs have finished.
