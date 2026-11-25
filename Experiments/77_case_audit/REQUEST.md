# Request 77: audit the case descriptions and the path-radius-2 overlap

Written 26 November 2026 by the paper agent. The manuscript is in
`~/Documents/TACsy/paper/synrfp_paper`; do not edit it. Export the files named here and
report the branch and commit. No model fitting is needed.

## Why

The Supplementary Information now describes the request 75 selections in words:
- EnzymeMap losses are carbohydrate epimerizations (EC 5.1) that D gets right and D+U gets wrong,
  and gains include a transamination (EC 2.6), an ammonia addition (EC 4.3) and a peptide hydrolysis (EC 3.4).
- QuantumPioneer rows are hydrogen abstractions with large barrier errors under both encodings.
These statements come from reading ten rows each. They must be checked against all eligible
cases before they stay in the paper.
The request 74 ceilings separate gaining from losing endpoints for WL r2, WL r3 and Morgan r2
but overlap for path r2 (0.27 to 16.6 against 0.03 to 12.3 points). The paper says so and gives no reason.

## Experiment A (required): population statistics behind the selected cases

1. EnzymeMap, request 75 Case F rule (paired seed votes): over all eligible gains (546) and losses (236)
   report the distribution of the true EC main class (first digit) and EC subclass (first two digits), the
   share of each class among gains and among losses, and a chi-square or Fisher test of gain versus loss
   composition. Also report the share of reactions whose bond-edit label differs between gains and losses.
2. QuantumPioneer, same rule, 7,467 gains and 6,509 losses: report the distribution of the reacting radical
   species (the attacking radical, for example HO, HOO, CH3O), the median and interquartile range of the
   true barrier and enthalpy among gains and losses, and the median absolute error of both arms in each group.
3. A one-line statement per dataset of whether the paper's sentence above is representative of the eligible
   population (yes or no, with the numbers).

Outputs: `enzymemap_composition.csv`, `quantumpioneer_composition.csv`, `case_descriptions_check.md`.

## Experiment B (required): why path radius 2 overlaps

Using the request 74 count classes for path radius 2 and WL radius 2 on all 14 endpoints, report per endpoint:
number of exact D classes, number of classes of D under path r2 that merge two or more WL r2 classes,
the share of reactions affected, and the label purity of the merged classes. State whether the large
ceiling gains on loss endpoints under path r2 (up to 12.3 points) come from classes that merge distinct
labels (which pooled counts then separate) or from singleton classes. A short note should say which of the
two explanations the numbers support.

Outputs: `path_overlap_summary.csv`, `path_overlap_note.md`.

## Constraints

Same as requests 74 to 76: pinned commits and hashes, package versions, no weights, no per-row predictions
beyond the cases already exported, state anything that cannot be reproduced. Push on a new branch and report the commit.
