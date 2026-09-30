# Should SynRFP receive more explanatory analysis?

## Decision

Yes, but only a compact mechanism-and-limitations package is still needed. The
existing evidence is already sufficient to explain the central representation
idea; more broad benchmarking or another large hyperparameter search would add
little and would risk turning exploratory results into post-hoc optimization.

## What is already explained

- Signed reaction differences are useful: delta beats union in all ten kNN
  factorial endpoints and in the completed clustering ablation endpoints.
- The intended invariances are visible: unchanged spectators cancel in SynRFP,
  while reversal changes SynRFP and leaves DRFP unchanged in the controlled
  examples. The unsigned-delta control separates cancellation from direction.
- The signal is not limited to one task: SynRFP leads the matched small-scale
  clustering comparisons and remains positive against DRFP in the completed
  label-free classification summaries.
- The story includes robustness and limits: paired subsample stability,
  shared-valid-row/L2 sensitivity, hash-seed sensitivity, failed-encoding
  rates, hierarchy profiles, AMI plus ARI, and the large-data exceptions.

## Highest-value additions

1. **Chemical interpretation panel.** Use the saved Mechanistic L2 and
   Schneider B assignments to show a few representative clusters, changed
   reaction motifs, cluster purity, and one counterexample. This connects the
   bit-level mechanism to chemistry without refitting the benchmark.
2. **Reaction-centre localization.** This is now complete for all 31,673
   mechanistic reactions using the main radius-2/original-reaction convention
   and occurrence-level coverage. Present it as supporting pre-sketch evidence,
   not as a claim that every hashed bit is uniquely interpretable.
3. **One practical-cost table.** On fixed hardware, report encoding time,
   feature size, valid-row rate, and nonzero density for SynRFP and the four
   label-free references. This makes the compact binary representation's
   trade-off explicit.

## Keep as limitations, not new headline experiments

- Stereochemistry is currently absent from the graph attributes; document this
  limitation unless a stereo-aware parser is added consistently to every
  representation.
- Hash-seed variation is already bounded on 2,000 Schneider B reactions; do
  not fold it into the performance uncertainty.
- Large clustering has real exceptions (TPL UB and ECReact L1 ARI), and
  XGBoost versus RDKit difference is task-dependent. Show these exceptions.
- RXNFP-MLM remains the label-free neural reference; TPL-1K/RXNFP-FT must be
  labelled supervised references.

## Defer

Do not launch another full clustering grid, more K-means seeds, or a new MLP
search for explanatory purposes. Finish the queued XGBoost/MLP matrices first,
then regenerate the completeness-gated audits and final tables. The added
analyses above explain *why* SynRFP helps and where it fails; another search
would mainly select favorable configurations after seeing the results.
