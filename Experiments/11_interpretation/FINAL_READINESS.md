# Current readiness checklist

This checklist is generated from the current worktree state and is intended to
prevent partial results from being presented as final.

## Complete and validated

- kNN factorial: 8,100/8,100 cells.
- Clustering ablation, hierarchy profiles, large-data summaries, and 20-pair
  stability subsamples.
- Cached-feature validity audit for five label-free clustering fingerprints.
- Representation controls for signed delta, unsigned delta, and DRFP.
- Paired bootstrap/sign-test summaries for the completed clustering stability
  contrasts and kNN mode contrast.
- Claim-level audit tying headline clustering and kNN statements to saved
  tables.
- Label-free classification summaries for all currently represented endpoints.
- Mechanistic L2 reaction-level assignment replay for all five label-free
  fingerprints, plus a Schneider B replay for SynRFP and DRFP, with
  deterministic representative examples and integrity checks.
- Schneider B shared-valid-row/L2-normalization sensitivity and four-seed
  SynRFP hash-seed sensitivity control.
- Corrected full reaction-centre localization: 31,673/31,673 Mech31k rows,
  original mapped reactions, radius 2, and occurrence-level coverage.
- Cached feature payload/cost summary and a presentation-ready localization
  figure.

## Provisional

- Main XGBoost benchmark: 2,257/2,364 cells complete. The remaining 107 are
  TPL UB (49) and TPL B (58), so XGBoost rankings are provisional.
- MLP benchmark records include benchmark-specific extra runs; use the
  coverage files before pooling them with the primary comparison.
- Factorial XGBoost: 562/8,100 cells (live snapshot; see `results/classification_progress.json`).
- Factorial MLP: 1,620/8,100 cells.

## Running resources

- TPL UB and TPL B XGBoost workers are active on separate GPUs.
- CPU factorial XGBoost workers are active.
- An ownership-safe deferred GPU wrapper is waiting for the active TPL
  benchmark. Once those GPUs are free, it will stop only the still-running CPU
  factorial workers, preserve completed JSON cells, replay missing XGBoost
  cells on both GPUs, then run the queued MLP stage. `anova.py` withholds
  output unless the full matrix is present.

## Reproducible commands

Run the read-only interpretation suite with:

```bash
Experiments/11_interpretation/run_audit.sh
```

The suite also writes `results/classification_family_leave_one_out.csv`, a
family-balanced sensitivity table for completed classification endpoints.

The machine-readable progress and ETA files are
`results/classification_progress.json` and `results/tpl_tail_eta.json`.
