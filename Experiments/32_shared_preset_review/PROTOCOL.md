# Shared preset review v1

This isolated report waits for the completed experiment 30 seven-task common
analysis. It reads no partially completed cells, starts no GPU or CPU fits and
does not select a final preset. All historical results, encoders and weights
remain untouched.

Verify the completed common receipt, its producer SHA, its own output completion
ledger, every bound output, all four canonical source ledgers and their pinned
plans/cell tables. Require 20,448 logical input cells, all seven tasks, exactly
72 unchanged configurations and both grouped_main/native_reference views.

Freeze one global shortlist of three configurations using grouped_main macro
rank ascending, macro normalized regret ascending, worst task rank ascending,
then configuration ID lexically. Source-family and optimizer/fold aggregation
comes exclusively from the completed common analysis. Every task receives equal
weight. Do not select a different configuration per task. Show all 72 ranks,
native-reference sensitivity, per-task ranks/regrets/coverage and separate raw
endpoint metrics. Never average different task metrics or regression units.

The existing full 72-configuration factorial already evaluates all three
candidates on all seven tasks. No rerun is necessary to construct this table.
These exploratory results follow prior inspection of test outcomes. Neither
the ranking nor additional random seeds remove selection bias. No independent
fold counts, p-values, confidence intervals or SOTA claim are produced.

A final matched baseline protocol is separate. Audit feature identity, original
populations and splits, model settings and artifacts before reusing fits. The
classification factorial has one grouped five-fold MLP evaluation; its main
benchmark has repeated 5×5 CV and kNN. Clustering's five row-count-K factorial
initializations are distinct from the main known-class-count and adaptive
unknown-K analyses. Native yield partitions and fixed expert/OOD tests retain
their original meaning. Condition optimizer seeds are averaged within folds.
Any further confirmation must freeze its candidate set and protocol first,
run only missing matched cells and describe dependence and historical selection.
User review determines whether one shared preset or a three-candidate
confirmation comparison is appropriate. This process does not launch that work.

`shortlist.py --once` writes a waiting status if the common receipt is absent.
`--watch` polls every 60 seconds (optionally 30) and exits after a verified
shortlist. Changes to its own source, protocol or test file while waiting are
fatal. Completion binds input receipts/tables, the producer/protocol/test SHAs
and all derived outputs. A changed completed result is refused, never silently
overwritten. Learned-reference completion is outside this receipt's scope.
