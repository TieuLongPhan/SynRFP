# r2 nested MLP follow-up — do not promote as the primary benchmark

This is an r2-only, post hoc model-selection experiment. For each of the 25 existing
scaffold-grouped outer folds, three inner scaffold-grouped folds selected among a small
MLP candidate pool; the untouched outer test fold was then scored once.

| Fingerprint | Mean outer MCC | SD |
|---|---:|---:|
| SynRFP | 0.94394 | 0.00370 |
| DRFP | 0.95295 | 0.00300 |

The matched mean difference, SynRFP minus DRFP, is -0.00901 MCC (SynRFP wins 4 of
25 outer folds). Although both representations improve over the original fixed-MLP
configuration, the experiment does not establish an r2 advantage for SynRFP.

**Decision:** retain the original fixed-configuration benchmark as the primary,
cross-dataset result. Treat this directory as a documented negative follow-up and do
not merge these r2-only scores into the main comparison table.

See `final_summary.json` for per-metric aggregates and `selection.json` for the
per-fold inner choices.
