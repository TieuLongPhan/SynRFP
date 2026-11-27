# Request 92 prediction and condition-coverage preflight

The saved predictions reproduce the full-row Study 65 yield release for standard
SynRFP, yield-selected SynRFP, DRFP and RDKit structural. The 180 method-by-split
checks cover 40 expanded Buchwald–Hartwig cluster holdouts and five grouped
USPTO folds. All saved row selectors agree with the frozen plan. The largest
absolute R² difference is 9.02 × 10⁻¹⁷. The published MAE and RMSE also agree within
10⁻¹⁰ on every split. The checks reused saved predictions.

The published table uses the full-row Study 65 release because the earlier
continuous-yield report covers a smaller shared-valid cohort. The condition
coverage audit groups exact author aryl and amine fields and counts distinct
catalyst, base and solvent triples on test rows. Across 40 holdouts there are
6,213 split-by-substrate-pair instances with at least eight test recipes. Of
these, 609 pairs are absent from both train and validation in their respective
split. Counts are per split, so the same pair can contribute in multiple holdouts.

This release covers prediction replay and condition coverage. Distance bins,
condition-ranking scores and examples are pending. The archived author CSV
abbreviates the DRFP vectors. The distance analysis requires the full vectors
from the author deposit and a row-identity check.
[Author dataset](https://zenodo.org/records/19636649)
