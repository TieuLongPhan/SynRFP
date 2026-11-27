# Request 92 yield analysis

## Error versus training similarity

The analysis uses the four released model predictions and the exact archived
fit-train and test selectors. The expanded Buchwald–Hartwig panel contributes
274,970 test-row occurrences across 40 holdouts. The USPTO panel contributes
143,535 across five grouped folds. Each prediction is compared against the
maximum binary Tanimoto similarity to a fit-train row. Validation rows are
excluded from the search. Two empty fingerprints have similarity one.

The common Buchwald–Hartwig reference uses the deposited full author `DRFP`
vectors, the column named by the author's clustering code. The deposit is
verified against the publisher's checksum. Row identity, visible bit ends,
cluster assignments and all 40 held-out selectors are checked. The USPTO
common reference is the released benchmark DRFP, alongside the four
representation-specific spaces. Maximum similarity uses exact integer
comparisons and is stored rounded to float32. Independent RDKit checks cover
all nine dataset-space combinations.

Similarity cutoffs are pooled within each panel and space. Identical
similarities stay together at a quantile boundary. The table gives pooled
counts, shares, MAE and R², along with the split-level means and sample standard
deviations. Empty split-bin cells count as zero for occupancy and have missing
errors. Split-bin R² is undefined for fewer than two observations or constant
observed yield. The 40 holdouts reuse rows, so variation across them is descriptive.
The released predictions use a single fitted learner per split. Confidence
bins would require extra seed fits and are outside this release.

## Condition ranking and examples

The ranking includes 609 split-by-substrate-pair instances with at least eight
test recipes and with the aryl–amine pair absent from train and validation.
A recipe consists of the released catalyst, base and solvent fields. Replicate
assays under the same recipe are averaged before scoring. Success requires
observed yield strictly above 10%. Ties are resolved by the lexicographic recipe
order. Spearman summaries exclude constant observed or predicted vectors and
report the defined count. The training recipe mean baseline uses only fit-train
yields and falls back to the global fit-train mean on unseen recipes. A constant
training global mean control is also included.

Reaction examples require a gap of at least ten yield percentage points between
the yield configuration's absolute error and the lower of the DRFP and RDKit
structural absolute errors. Eligible reactions are ordered by heavy-atom count,
source position and split. Selected cases use distinct largest-component
Bemis–Murcko scaffolds across cases, with canonical largest-component identity
for acyclic cases. Ranking examples display the top recipe chosen by each
method on the same pair. The table reports assay counts when a displayed yield
and its predictions are averages.

## Reproduction

Run `rank92.py`, `distance92.py own`, `validate92.py own`, `distance92.py author`,
`validate92.py author`, then `aggregate92.py` in the workstation SynRFP environment.
The author analysis reads the checksummed pickle and pinned clustering code
from the local Request 92 artifact directory. The manifest records source and
split hashes, control results and table hashes.

[Author data deposit](https://zenodo.org/records/19636649) ·
[Pinned author clustering code](https://github.com/schwallergroup/bh-hte-ood/blob/3a44e398d9fed3f08cf396f1153be2960193f581/utils/drfp_cluster.py)
