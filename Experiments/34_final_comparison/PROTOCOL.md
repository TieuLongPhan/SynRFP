# Final comparison using the frozen shared SynRFP preset

The selected experiment preset is **WL, radius 2, parity, signed delta, 2,048
bits, fingerprint seed 1**. It is the exact construction frozen by experiment33,
not the historical 1,024-bit default. The public library default is unchanged.

The six fingerprint panel comprises that preset, DRFP, RDKit structural, RDKit
difference, RXNFP-MLM and crxnfp. Supervised RXNFP, Chemprop D-MPNN, SynCat and the
three learned yield references remain separately labelled context. They are not
included in a fingerprint average. No XGBoost is introduced.

The final panel contains **8,238 cells**:

| Task | Cells | Protocol |
|---|---:|---|
| Classification | 4,200 | Fourteen endpoints, kNN and MLP, original five repetitions of five grouped folds |
| Clustering | 1,440 | Twelve endpoints, ten K-means initializations, original all-row and all-six-valid populations |
| Yield | 384 | Three datasets, original 64 published random/OOD/clustered-OOD partitions |
| Barrier | 1,944 | Original110 two-dataset partitions plus214 activation-energy partitions across seven extension datasets, six fingerprints |
| Feasibility | 30 | Five training seeds, one fixed751-row expert test |
| Reagent | 60 | Five native initialization seeds and one grouped five-fold evaluation |
| Condition | 180 | One fixed grouped five-fold partition with five initialization seeds, plus five native initialization seeds |

Existing compatible primary results are retained:700 classification,214 barrier
extension, and45 auxiliary cells. All compatible baseline results are retained.
The missing selected-preset cells are **64 yield,110 barrier and240 clustering**.
Only these414 replacements need new execution. Original results, source rows,
feature caches, raw data, models and historical reports are never overwritten.

`report.py --prepare` validates the frozen construction and canonical reused
results. It seals canonical CSVs, original result JSON files, scientific plans,
original row-selector partitions, worker metadata and the selected-preset receipt
under `~/.local/share/synrfp-runs/final_comparison_v1/report/plan.json`. Individual
completed model JSONs retain their artifact catalogs. A seal verifies canonical
metadata identity; it does not claim a new full weight replay for historical
classification models whose old receipts lack one. Existing replay/audit scope
remains explicit in the source protocols.

The regression component performs fresh CUDA fits using the original13/14
native worker arithmetic and exact shared rows/splits. Its completed receipt
requires retained-weight prediction replay, full artifact hashes and metric
replay. The clustering component verifies the five original baselines and
replaces the old SynRFP assignment with the selected2,048-bit construction,
keeping the original known-K method, seed and population per endpoint. This is
a matched known-K comparison; separate label-blind unknown-K analyses remain
separate evidence.

`report.py --once` always writes an explicitly **partial** report while either
required component is missing. It writes overall `COMPLETE.json` only when all
8,238 scientific cell identities and both component receipts, plan bindings,
canonical result hashes and output hashes pass. Six matching method grids,
original split identities, seed/fold keys and training/test row counts are
checked; counts alone do not authorize completion. All outputs are local to
experiment34. Weights remain local and are not published.

Raw cells and mean/sample-SD summaries are written separately for each task.
Classification and random repeated CV summarize repeat means and retain raw-fold
SD separately. NCS yield summarizes its ten iteration means over four cluster
holdouts. Condition grouped SD uses five fold means after averaging its five
optimizer seeds. Fixed native and external tests report optimizer-seed variation.
Clustering reports initialization variation. Native, random, grouped, OOD,
learning-fraction and row-population panels never mix. MCC, R² and unlike task
metrics are not pooled. Related GDB7/RDB7 and E2/SN2/combined source families are
not independent datasets. No independent-fold confidence or significance claim
is made.

The selected preset was chosen after earlier test results had been viewed.
Consequently this final table is an **exploratory matched comparison**, not an
untouched confirmation. New seeds cannot remove this selection bias. There is
no global SOTA claim. Extra redrawn CV repeats for auxiliary tasks and new
specialist reference adaptations are outside this414-cell completion scope.

Run the meaningful bounded report checks with:

```bash
python -m unittest discover -s Experiments/34_final_comparison -p test_report.py
python Experiments/34_final_comparison/report.py --prepare
python Experiments/34_final_comparison/report.py --once
```

The parent controller owns CPU/GPU admission and persistence. The report is CPU
only and never creates a CUDA context or changes a device lease.
