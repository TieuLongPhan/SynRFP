# Cross-task representation consistency

This is a descriptive audit of the frozen label-free fingerprints.
Classification uses scaffold-CV MCC summaries; clustering uses
small-endpoint K-means AMI summaries. Shared datasets are related
views, so margins are not independent evidence or a pooled test.

## Summary

- classification / knn vs drfp: SynRFP wins 6/6 endpoints (100.0%); mean margin +0.0502.
- classification / knn vs rdkit_diff: SynRFP wins 5/6 endpoints (83.3%); mean margin +0.1183.
- classification / knn vs rdkit_struct: SynRFP wins 6/6 endpoints (100.0%); mean margin +0.3493.
- classification / knn vs rxnfp_mlm: SynRFP wins 6/6 endpoints (100.0%); mean margin +0.1724.
- classification / mlp vs drfp: SynRFP wins 6/6 endpoints (100.0%); mean margin +0.0381.
- classification / mlp vs rdkit_diff: SynRFP wins 6/6 endpoints (100.0%); mean margin +0.0200.
- classification / mlp vs rdkit_struct: SynRFP wins 6/6 endpoints (100.0%); mean margin +0.0416.
- classification / mlp vs rxnfp_mlm: SynRFP wins 6/6 endpoints (100.0%); mean margin +0.0680.
- classification / xgb vs drfp: SynRFP wins 6/6 endpoints (100.0%); mean margin +0.0279.
- classification / xgb vs rdkit_diff: SynRFP wins 2/6 endpoints (33.3%); mean margin +0.0003.
- classification / xgb vs rdkit_struct: SynRFP wins 6/6 endpoints (100.0%); mean margin +0.0250.
- classification / xgb vs rxnfp_mlm: SynRFP wins 6/6 endpoints (100.0%); mean margin +0.1481.
- clustering / kmeans vs drfp: SynRFP wins 5/5 endpoints (100.0%); mean margin +0.1036.
- clustering / kmeans vs rdkit_diff: SynRFP wins 5/5 endpoints (100.0%); mean margin +0.2264.
- clustering / kmeans vs rdkit_struct: SynRFP wins 5/5 endpoints (100.0%); mean margin +0.4290.
- clustering / kmeans vs rxnfp_mlm: SynRFP wins 5/5 endpoints (100.0%); mean margin +0.2394.

## Interpretation

A positive margin in both tasks on the same endpoint supports a
representation-level signal that is not tied to one estimator.
A sign change identifies a task/protocol-dependent comparison and
should be reported as such. This audit does not replace paired
split or initialization inference.
