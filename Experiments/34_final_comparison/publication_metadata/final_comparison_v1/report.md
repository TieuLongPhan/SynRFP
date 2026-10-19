# Final shared-preset fingerprint comparison

Status: COMPLETE

Primary construction: WL, radius 2, 2,048 total bits, parity sketch, signed delta, fingerprint seed 1.

Six fingerprints: selected SynRFP, DRFP, RDKit structural, RDKit difference, RXNFP-MLM and crxnfp. The historical 1,024-bit SynRFP is excluded.

| Task | Audited cells | Expected |
|---|---:|---:|
| classification | 4200 | 4200 |
| clustering | 1440 | 1440 |
| yield | 384 | 384 |
| barrier | 1944 | 1944 |
| feasibility | 30 | 30 |
| reagent | 60 | 60 |
| condition | 180 | 180 |

Classification includes both kNN and MLP on the original five repetitions of five grouped folds. Clustering here is known-K K-means with ten initializations, separately reporting all rows and the original common valid population. It does not replace the label-blind unknown-K analyses.

Yield keeps original published random, OOD and clustered-OOD partitions. Barrier keeps random CV, learning fractions, grouped folds and grouped holdouts separate; extension15 adds seven activation-energy endpoints. GDB7/RDB7 and E2/SN2/combined source families are related and are not independent datasets.

Reagent uses five native initializations and one grouped five-fold evaluation. Condition uses one fixed grouped five-fold partition with five optimizer seeds per fold, plus five native initializations. Feasibility uses five training seeds on one fixed external expert test. None of these is described as five redrawn CV repetitions.

Summary SD uses repeat means for repeated CV; condition grouped SD uses fold means over optimizer seeds. Raw-cell SD is retained separately. These are descriptive dispersions, not confidence intervals or independent-fold significance tests.

Selection is exploratory: these test outcomes were already viewed when the common preset was chosen. Re-fitting with new seeds does not remove selection bias. This report makes no SOTA or untouched-confirmation claim. Metrics remain separate by task and protocol; MCC and R² are not averaged together.

Learned yield references, supervised RXNFP, D-MPNN and SynCat remain separately labelled context in reference_inventory.json. The completed queue does not include newly adapted feasibility SynCat, Parrot, T5Chem or a full SynCluster comparison.
