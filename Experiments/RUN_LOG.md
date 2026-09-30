# SynRFP Benchmark Run Log
**Date:** 2026-09-20
**Session:** Started 2026-09-18 23:13:00

## Environment
- **Machine:** 64-core, 128GB RAM
- **Conda env:** synrfp (Python 3.11) + rxnfp (Python 3.8)
- **Config:** `n_jobs: 64`, RAM-capped at 100GB (benchmark) / 40GB (factorial)
- **Fingerprints:** synrfp_default, drfp, rdkit_struct, rdkit_diff, rxnfp, rxnfp_mlm, rxnfp_schneider

## Data Summary
- **Available datasets:** 12/14 (mech_1/2 blocked — raw MechFinder CSV missing)
- **Datasets:** schneider_u/b, tpl_u/b, uspto_50k_u/b, syntemp_r0/r1/r2, ecreact_1/2/3, ehreact_activity
- **Total reactions featurized:** ~1.3M across 9 reaction tables + EHreact export

## Running Jobs (as of 2026-09-20 04:21 UTC)

### 1. Benchmark Classification (Track A CV + Track B fixed)
- **Status:** ACTIVE — 5h 8m elapsed
- **Progress:** 287 / ~5,400 expected runs (5.3%)
- **Current dataset:** schneider_ub (50k rows, 50 classes) — 287/450 complete (~64%)
- **Rate:** ~57 runs/hour
- **Remaining datasets:** 11 (schneider_b, tpl_u/b, uspto_50k_u/b, syntemp_r0/r1/r2, ecreact_1/2/3)
- **ETA:** 30-40+ hours total (MLP on TPL is the bottleneck)
- **Memory:** 2.16 GiB / 100 GiB capped
- **CPU:** ~35-36 cores active

### 2. Factorial Classification (SynRFP ablation)
- **Status:** ACTIVE — 3h 0m elapsed
- **Progress:** 116 / 15,552 expected runs (0.7%)
- **Rate:** ~39 runs/hour (bias: mostly fast kNN phase)
- **Datasets:** 8 (schneider_ub/b, tpl_b, uspto_50k_ub/b, syntemp_r0, ecreact_1/3)
- **ETA:** 200-400+ hours (depends on MLP slowdown on large datasets)
- **Memory:** 0.45 GiB / 40 GiB capped
- **CPU:** ~33-34 cores active

### 3. Completed Studies
- **EHreact activity (10_enzyme_activity):** ✅ DONE
  - 3 assays (csp2–csp3 coupling, alcohol dehydrogenase E-2, dehalogenase E-1)
  - 5 fingerprints × 2 models × 5 repeats = 30 results
  - Outputs: tasks.csv, runs.csv, summary.csv

## Fingerprint Status
- **Local (CPU):** synrfp_default, drfp, rdkit_struct, rdkit_diff ✅
- **RXNFP (GPU):** rxnfp, rxnfp_mlm ✅ (all 9 reaction tables)
- **RXNFP dataset-specific:** rxnfp_schneider ✅ (tpl_ub/b, schneider_ub/b)
- **RXNFP missing:** rxnfp_tpl (bert_class_1k_tpl not downloaded — GitHub-only)

## Key Notes
- No errors in either job (logger noise from RDKit SMILES sanitization is benign)
- System load ~92-95 on 64 cores (healthy 50% oversubscription via scheduler)
- Benchmark job has MLP training as primary bottleneck; TPL datasets (445k rows × 1000 classes) will dominate runtime
- exp.md proposes fallback: reduce TPL/ECReact to 5×2 or 3×5 CV if full 5×5 becomes infeasible

## Next Steps
- Wait for benchmark to complete all 12 datasets
- Factorial will continue independently (expected to finish last)
- Once benchmark complete: run `09_report/report.py` for statistics + plots
- If time permits, run remaining studies (05_clustering, 06_retrieval, 07_robustness, 08_reaction_center)

---
**Last updated:** 2026-09-20 04:21 UTC
**Session host:** Tieu Long Phan <tieuphanphan.work@gmail.com>
