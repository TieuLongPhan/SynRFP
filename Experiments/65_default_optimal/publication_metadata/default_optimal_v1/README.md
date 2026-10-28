# Completed default and task-optimal fingerprint comparison

All seven task tables closed on 27 October 2026 at 18:42 UTC. This release
contains 9,691 main result cells, plus a separate 840-score full-row
class-count-blind clustering comparison. It adds 1,089 matched fits and reuses
compatible completed results with their source identities retained.

| Task | Main cells | Task-optimal SynRFP construction |
| --- | ---: | --- |
| Classification | 4,900 | WL / radius 2 / parity / delta_union |
| Clustering, known-K | 1,680 | Path / radius 2 / parity / delta |
| Yield | 528 | WL / radius 3 / parity / original roles |
| Barrier | 2,268 | WL / radius 2 / parity / delta_union |
| Feasibility | 35 | Path / radius 2 / parity / delta |
| Reagent | 70 | Path / radius 2 / CW / original roles |
| Condition | 210 | Path / radius 2 / parity / delta |

The shared experimental default is WL / radius 2 / parity / delta. Every
construction uses 2,048 total bits and fingerprint seed 1. Library defaults
are unchanged. Each task has one selected construction across its datasets;
selection is exploratory after viewed outcomes. Classification selection uses
MLP MCC and transfers to kNN; it is not a separate kNN optimum. See
[the protocol](../../PROTOCOL.md) and [frozen decisions](DECISIONS.json).

Each task directory contains per-cell results and native-protocol summaries.
Classification retains the original 25 saved splits and both MLP and kNN.
Known-K clustering and `clustering_blind/cells.csv` are separate evaluations.
Yield includes the full 27,497-row NCS cohort and 143,535-row USPTO cohort;
the yield-wide selected construction is transferred to USPTO. Native and
grouped evaluation panels remain separate. Summary variation follows each
native protocol and should not be interpreted as independent dataset samples.

The accompanying [corrected factorial release](../../../56_factorial_refresh/publication_metadata/corrected_full_cohort_v1/README.md)
contains all 20,448 primary cells and the separately corrected kNN summaries.
Architecture references continue independently and are outside this completion.
The legacy RGD1 D-MPNN cohort has 44 fewer rows than the corrected FP cohort;
its results cannot be presented as a full-row matched reference without repair.

`COMPLETE.json` records the native completion receipt's hash and hashes for the
exported scalar files. Those hashes were checked during export. Full model
artifact verification remains documented in the native local receipts. Weights,
raw data, features and prediction arrays remain local and are not included.
