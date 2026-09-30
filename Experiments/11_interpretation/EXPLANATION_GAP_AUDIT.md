# SynRFP explanation-gap audit

Audited 2026-09-26 against the current result files and live queues.

## Decision

The paper does not need another broad benchmark or another model search to
explain SynRFP. The central mechanism is supported by matched classification
contrasts, clustering ablations, representation controls, reaction-centre
localization, and cost/validity measurements. Additional training now would
mainly increase compute and post-hoc selection risk.

## Evidence already sufficient for the main explanation

- Signed delta versus union: the complete 8,100-cell kNN factorial is positive
  on all ten endpoints; mean MCC is 0.9262 versus 0.7475.
- Mechanistic meaning: reversal changes SynRFP, unchanged spectators cancel,
  and the unsigned-delta control removes reversal sensitivity.
- Cross-task support: SynRFP leads the completed matched kNN and MLP summaries;
  clustering support is strongest on the seven small K-means endpoints and is
  explicitly qualified on large-data exceptions.
- Robustness: paired subsamples, shared-valid-row/L2 checks, hash-seed
  sensitivity, hierarchy profiles, validity rates, and AMI plus ARI are all
  available.
- Chemical localization: the corrected radius-2/original-reaction E8 run has
  31,673 valid reactions and mean occurrence-level center fraction 0.8749.
- Practical trade-off: feature size, density, validity, and estimated payload
  are summarized for all five label-free representations.
- Serialization/reversal robustness: a new 8,000-reaction perturbation control
  confirms SynRFP's order invariance and direction sensitivity across four
  datasets, with validity recorded explicitly.

## The only high-value additions still worth making

1. Add one paragraph explaining the large-data exceptions (TPL UB and the
   ECReact L1 ARI result) and the SynTemp fine-resolution margin narrowing.
2. Use the new applicability/failure-mode table in
   `APPLICABILITY_ANALYSIS.md` to report model-specific losses. It shows that
   RDKit difference is the closest XGBoost comparator, while TPL and fine
   SynTemp are recurring stress cases. The MLP convergence audit (median
   stopping epoch 28 for SynRFP) gives no evidence that its current result is
   simply under-trained.
3. After the queued classification matrices finish, regenerate the final
   completeness-gated leaderboard and replace all provisional XGBoost wording.

The fixed-rule chemical panels are now rendered at
`results/cluster_examples_mech2_panel.png` and
`results/cluster_examples_schneider_b_panel.png`. They show the highest-purity
examples for each available fingerprint plus a low-purity SynRFP counterexample;
they are illustrative assignments, not additional independent benchmark runs.

Do not start another full clustering grid, another MLP search, or a broad
stereo/hash sweep for explanatory purposes. Those are secondary sensitivity
analyses and are not needed for the current mechanism claim.

## Current reproducibility gate

The interpretation audit currently reports `all_ok: true`, with zero duplicate
logical cells, zero locks, and all protocol checks passing. Classification is
still incomplete: benchmark XGBoost is 2,258/2,364, factorial XGBoost is
562/8,100, and factorial MLP is 1,620/8,100. Therefore the final model-ranking
claim remains provisional until those queues complete.
