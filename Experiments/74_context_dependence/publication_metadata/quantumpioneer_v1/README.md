# QuantumPioneer fixed-width context results

All 30 fits completed on 24 November 2026. The study pairs five fingerprint
arms and three fitting seeds on two targets, using 16,329 held-out records
across 13,349 reactant-pair groups. The [protocol](../../PROTOCOL.md) defines the
contrast. The training-only diagnostic met its coverage thresholds and ranked
reaction enthalpy above forward barrier for relative benefit from pooled context.

| Target or contrast | Gain | 95% paired group-bootstrap interval |
|---|---:|---:|
| Barrier | +0.0165 | +0.0099 to +0.0234 |
| Reaction enthalpy | +0.0138 | +0.0089 to +0.0188 |
| Enthalpy gain minus barrier gain | −0.0027 | −0.0108 to +0.0052 |

Gain is the reduction in group-averaged squared error versus ordinary
2,048-bit delta, normalized by the training target variance. The prespecified
ordering remains inconclusive. Both targets benefited relative to ordinary
delta. The pooled-versus-reactant-context intervals both span zero.

The release contains per-fit scores, all paired comparator summaries and the
10,000 paired bootstrap draws. Each draw uses the same groups across all
arms and targets. The frozen prediction, split hashes, overlap and coverage
audits, and development and synthetic controls accompany the comparison.
Checkpoint and prediction hashes were verified against the local fit receipts.
Row-level artifacts and model checkpoints are stored on the workstation.

The feature audit recorded that the pinned WL node attributes omit explicit
radical counts. All QuantumPioneer rows contain radicals.
