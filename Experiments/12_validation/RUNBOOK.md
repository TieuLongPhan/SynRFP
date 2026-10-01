# Validation suite: execution runbook

For the agent (or person) running the suite on the workstation. Read
[PLAN.md](PLAN.md) first for what each step is meant to show. This file is the
order of work, the exact commands, what to check after each one, and what to
hand back.

All commands run from `Experiments/`. Nothing here writes into the numbered
experiment folders `01_…`–`11_…` or their `results/`; do not edit those.

## 0. Ground rules

1. **Do not change the protocol.** `study.yaml` (panels, seeds, payloads,
   gates) was frozen before any run. If a setting looks wrong, stop and report
   it; do not tune. Fixing a crash is fine; changing what is measured is not.
2. **XGBoost is out.** Only 5-NN and MLP are run. Do not start
   `cpu_benchmark_queue.py` or any `--models xgb` command.
3. **Record every deviation** (a bug fix, a skipped dataset, a restarted step)
   as a dated line in `12_validation/runs/DECISIONS.md`.
4. **The retrieval `final` part is scored exactly once**, after step 7 is
   complete and reviewed on the development part.
5. **Report failures as they are.** A failed encoding stays in the
   denominator; a gate that fails is reported as failed.

## 1. What has and has not been tested

Written on a laptop without the full data or GPUs.

| Script | Status |
| --- | --- |
| `00_setup.py`, `vlib.py` | Ran end to end on a 4,000-row subset of four endpoints |
| `01_input_audit.py` | Ran on two endpoints |
| `featurize.py` / `classify.py --models knn` with the generated configs | Ran on two endpoints, policies raw and n1 |
| `02_attribution.py` | 5-NN path ran on one endpoint; the MLP path was reorganized afterwards and is untested |
| `03_retrieval_proxy.py`, `03b_…`, `04_locality.py`, `05_cost.py`, `05b_…`, `06_report.py`, `run_cpu.sh` | Syntax-checked only |
| `mlp_gpu_queue.py` with the generated configs | Not run |

So **step 3 (smoke run) is mandatory**. Expect to fix small bugs in the
syntax-checked scripts there.

## 2. Environment

```bash
cd <SynRFP checkout>/Experiments
git status                      # expect a clean tree apart from 12_validation/
git rev-parse HEAD              # record it; the archived runs used 4b1e944

conda activate synrfp           # Python >= 3.11
pip install -e ..               # this checkout
pip install -r 12_validation/requirements.txt
export PY=$(which python)
export RXNFP_PY=<path to the rxnfp env python>    # external/rxnfp/environment.yml
export GPU_PY=<python with CUDA torch + numpy>    # may equal RXNFP_PY
```

Check: `$PY -c "import synrfp, drfp, rdkit, synrxn, synkit, tabulate"` succeeds
and `$GPU_PY -c "import torch; print(torch.cuda.device_count())"` prints the
number of GPUs.

## 3. Smoke run (about 20–40 minutes, mandatory)

```bash
$PY 12_validation/00_setup.py --smoke 4000 \
    --datasets mech_1 mech_2 uspto50k_b ecreact_1 --rxnfp-python $RXNFP_PY
export VALIDATION_SMOKE=1
DATASETS="mech_2 uspto50k_b ecreact_1" POLICIES="raw n1" \
  ATTRIBUTION_ARGS="--datasets mech_2" LOCALITY_ARGS="--sample 500" \
  bash 12_validation/run_cpu.sh
$RXNFP_PY external/rxnfp/rxnfp_embed.py --config 12_validation/smoke/configs/n1.yaml \
    --fingerprints rxnfp rxnfp_mlm
$RXNFP_PY 12_validation/03b_retrieval_rxnfp.py
$PY 03_classification/mlp_gpu_queue.py --config 12_validation/smoke/configs/n1.yaml \
    --gpu-python $GPU_PY --devices 0 --studies benchmark \
    --control 12_validation/smoke/runs/n1/mlp_gpu_queue --limit 8
$PY 12_validation/05_cost.py && $RXNFP_PY 12_validation/05b_cost_rxnfp.py
$PY 12_validation/03_retrieval_proxy.py --stage score --part development
$PY 12_validation/06_report.py
unset VALIDATION_SMOKE
```

Pass criteria:

- `00_setup.py` logs `input-policy self-test passed`.
- `smoke/runs/input_audit/serialization.csv`: SynRFP and both RDKit
  fingerprints have `identical_fraction` 1.0 under both policies; DRFP is 1.0
  under `n1` (it is expected to be well below 1.0 under `raw`).
- `02_attribution.py` logs `default payload reproduces library fingerprints`
  and writes `smoke/runs/attribution/n1/mech_2__wl.csv` with `model` values
  `knn` and `mlp`.
- `smoke/runs/retrieval_proxy/coverage.json` exists and
  `summary__development.csv` has one row per fingerprint.
- `smoke/runs/report/SUMMARY.md` and `gates.json` are written.

Smoke numbers mean nothing scientifically; do not report them. Fix crashes,
note each fix in `DECISIONS.md`, and rerun the failing command. The retrieval
tables are small, so the smoke retrieval run uses the full query set; that is
fine because the smoke tree is discarded.

## 4. Full setup (about 30–60 minutes)

```bash
$PY 12_validation/00_setup.py --rxnfp-python $RXNFP_PY
```

Check `runs/setup.json`:

- `splits_origin` says `copied from archived work/splits` for a dataset when
  the old `work/splits/<dataset>` still exists. That is the best case: the new
  runs use the archived partitions. `regenerated in this environment` is
  acceptable; all three policies still share identical splits.
- `runs/setup_policy_report.csv`: `status_fallback_raw` should be a small
  count per table. Report the counts; do not drop those rows.
- `runs/splits/*.npz` exist. These are the split codes of record.

## 5. Features

CPU features for all policies (restartable, hours for the TPL tables):

```bash
for P in raw n1 n2; do
  $PY 02_featurize/featurize.py --config 12_validation/configs/$P.yaml --study benchmark
done
```

RXNFP embeddings (GPU; `raw` is skipped automatically when the archived
feature files were linked by setup):

```bash
for P in raw n1; do
  $RXNFP_PY external/rxnfp/rxnfp_embed.py --config 12_validation/configs/$P.yaml \
      --fingerprints rxnfp rxnfp_mlm
done
```

Check: every `work/validation/<policy>/features/<table>/` has one `.npz` per
fingerprint (`n2` needs no RXNFP files; it is a 5-NN sensitivity arm).

## 6. Classification

5-NN, all policies. Result files are claimed with lock files, so several
processes can share the work; start one per group of datasets:

```bash
for P in raw n1 n2; do
  C=12_validation/configs/$P.yaml
  OMP_NUM_THREADS=8 $PY 03_classification/classify.py --config $C --models knn \
      --datasets schneider_ub schneider_b uspto50k_ub uspto50k_b syntemp_r0 syntemp_r1 syntemp_r2 mech_1 mech_2 &
  OMP_NUM_THREADS=8 $PY 03_classification/classify.py --config $C --models knn \
      --datasets ecreact_1 ecreact_2 ecreact_3 &
  OMP_NUM_THREADS=8 $PY 03_classification/classify.py --config $C --models knn --datasets tpl_ub &
  OMP_NUM_THREADS=8 $PY 03_classification/classify.py --config $C --models knn --datasets tpl_b &
  wait
done
```

MLP on the GPUs, policies `raw` and `n1` (one queue at a time; each uses both
GPUs):

```bash
for P in raw n1; do
  $PY 03_classification/mlp_gpu_queue.py --config 12_validation/configs/$P.yaml \
      --gpu-python $GPU_PY --devices 0 1 --studies benchmark \
      --control 12_validation/runs/$P/mlp_gpu_queue
done
```

`--dry-run` prints pending, completed, and blocked counts without training.
`blocked` must be 0 before a real run (a blocked job means a missing feature
file). To pause, `touch <control>/STOP`; remove the file to resume.

Expected cells per policy: 14 endpoints × 25 splits × 6 fingerprints = 2,100
per learner (`n2`: 5-NN only, and without RXNFP rows 1,400). `06_report.py`
writes the exact found/expected table.

## 7. Attribution, locality, retrieval proxy, cost

```bash
$PY 12_validation/02_attribution.py            # 5-NN + MLP, 4 endpoints x 2 vocabularies
$PY 12_validation/04_locality.py
$PY 12_validation/03_retrieval_proxy.py --stage fetch
$PY 12_validation/03_retrieval_proxy.py --stage build
$PY 12_validation/03_retrieval_proxy.py --stage featurize
$RXNFP_PY 12_validation/03b_retrieval_rxnfp.py
$PY 12_validation/03_retrieval_proxy.py --stage score --part development
```

`02_attribution.py` is the long CPU step: per endpoint and vocabulary, 360
5-NN runs, 30 exact-set runs, and 30 CPU MLP fits on up to 50,000 reactions.
It writes one CSV per (endpoint, vocabulary) and skips finished ones; split it
across shells with `--datasets` / `--vocabularies` if useful.

Checks:

- Attribution log shows `default payload reproduces library fingerprints` for
  every endpoint. If that assertion fails, stop: the payload code and the
  library disagree and the contrasts would be meaningless.
- `runs/retrieval_proxy/coverage.json`: report every count in `steps`.
- `runs/locality/bounds_summary.csv`: for `original_reactions`, radius 2,
  `mean_root_order` should be close to 0.875 and `mean_union_fraction` close
  to 0.213 (the archived audit); the bounds were 0.853 and 0.900 on the laptop.

Cost, **alone on an otherwise idle machine**, after everything above:

```bash
$PY 12_validation/05_cost.py
$RXNFP_PY 12_validation/05b_cost_rxnfp.py
```

## 8. Report and gates

```bash
$PY 12_validation/06_report.py
```

Read `runs/report/SUMMARY.md` and `gates.json`. Then, and only if the
development retrieval results and the classification inventory are complete,
score the final retrieval part **once** and regenerate the report:

```bash
$PY 12_validation/03_retrieval_proxy.py --stage score --part final
$PY 12_validation/06_report.py
```

## 9. Optional, after the gates

Clustering and the archived perturbation audit on canonical input (5-NN
geometry without labels):

```bash
C=12_validation/configs/n1.yaml
$PY 05_clustering/cluster.py --config $C --methods kmeans
$PY 07_robustness/robustness.py --config $C --parts perturb
```

## 10. What to hand back

1. `12_validation/runs/report/` (all CSVs, `SUMMARY.md`, `gates.json`).
2. `12_validation/runs/manifests/`, `runs/environment/`, `runs/setup.json`,
   `runs/splits/`, `runs/DECISIONS.md`.
3. The per-run JSON trees `runs/<policy>/03_classification/results/`.
4. A short written summary with, in this order:
   - anything that failed or was skipped, with the error;
   - the classification inventory (found / expected);
   - each gate: pass, fail, or not evaluable, with the numbers from
     `gates.json`;
   - whether splits were copied from the archive or regenerated;
   - every bug fix made to the scripts.

Suggested archive:

```bash
tar czf validation_runs_$(date +%Y%m%d).tar.gz -C 12_validation \
    runs study.yaml configs external/SOURCES.json
```

## 11. If something goes wrong

| Symptom | Likely cause and action |
| --- | --- |
| `cached tables … were built from SynRXN …` | The base `work/tables` come from another data source. Do not delete them blindly; report which source the archived runs used (`work/tables/SOURCE.json`) and set `synrxn.source` in `config.yaml` to match before rerunning setup. |
| `missing features …; run 02_featurize.py` | Step 5 not finished for that policy/table. |
| MLP queue prints `blocked_entries > 0` | RXNFP feature files missing for that policy; run the embed command in step 5. |
| DRFP identity under `n1` below 1.0 in the input audit | Unexpected. Save `serialization.csv`, report the tables affected, and continue; do not alter the policy. |
| A retrieval fingerprint has a low `encoded_fraction` | Expected for some encoders on Rhea strings with generic atoms (`*`). It is part of the result; do not filter queries. |
| Out of memory in 5-NN on TPL | Lower `OMP_NUM_THREADS`, run the two TPL endpoints one at a time. |
