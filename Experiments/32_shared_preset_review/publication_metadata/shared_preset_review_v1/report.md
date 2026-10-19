# Exploratory shared SynRFP shortlist

All 72 constructions cover all seven tasks. The three global candidates are:

1. `synrfp-wl-r2-b2048-parity-delta`
2. `synrfp-morgan-r2-b2048-parity-delta_union`
3. `synrfp-morgan-r2-b2048-parity-delta`

Ordering: grouped_main macro rank, macro normalized regret, worst task rank, then configuration ID. Each task has equal weight after source-family balancing. Native-reference sensitivity is reported separately.

No final preset is selected. These are exploratory diagnostics after test results were inspected; no confidence intervals, significance claims or SOTA claims follow from this shortlist.

The completed 72-construction factorial already contains these three candidates on every task. This review starts zero fits and does not repeat those cells. Raw endpoint metrics remain separate; different metrics or regression units are never averaged together.

Final baseline comparison remains a separate protocol: audit exact features, row populations, splits, learner settings and retained artifacts before any reuse. Classification factorials contain one grouped five-fold MLP evaluation; the main comparison includes repeated 5×5 grouped CV and kNN. The factorial does not establish completion of those additional fits. Clustering uses five initializations with row-count-based K, distinct from the main known-class-count and adaptive unknown-K studies. Yield keeps the original 64 native partitions; external feasibility uses a fixed expert test. Condition folds average optimizer seeds before fold diagnostics.

After user review, freeze a shared preset or a clearly defined three-candidate confirmation protocol. Extend only genuinely missing matched evaluations, preserve published OOD/external tests and account for dependent folds. New seeds on historically viewed data do not erase selection bias. Learned-reference completion is independent of this review.
