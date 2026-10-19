# Common SynRFP representative diagnostics

All seven task factorials are complete. This table supports choosing one common preset; no preset is selected automatically.

Scores are exploratory because historical test outcomes were viewed. Ranks and regrets are balanced within source families and then across seven tasks.

The grouped_main view uses grouped reagent/condition evaluations. The native_reference view substitutes their author splits; those regimes are never averaged together.

Yield retains all four original regimes with separate rows; random and OOD regimes inside each source family receive equal weight. Barrier families link GDB7/RDB7 and E2/SN2/E2SN2.

Study22 clustering fixes K from the endpoint row count using min(512,max(2,round(sqrt(n/2)))); class labels and class counts do not choose K. These factorial cells remain separate from the main known-class-count comparison and adaptive unknown-K clustering experiments.

Condition fold means average five optimizer seeds first. Native/FREA repeated seeds use one fixed test set. Fold training populations and repeated holdouts overlap; repeated_test_rows_sum is not an independent sample size.

Encoding coverage is descriptive:study22 classification/yield use held-out coverage, study22 clustering uses all input rows, and study30 uses the frozen task population. Its macro is a mean reported fraction, not a new pooled coverage estimate.

Cost:the common candidate table does not invent missing per-fit durations. Study30's cell table currently omits those columns; source-specific encoding and training receipts retain cost evidence for a separate measured cost analysis.

NCS BH includes some source material related to classic BH HTE; source-family balance is a transparent weighting convention, not a claim of dataset independence. EHReact is excluded.

Read common_candidates.csv together with endpoint_summary.csv, family_summary.csv and task_regime_summary.csv. Different metrics or regression units are never directly pooled.
