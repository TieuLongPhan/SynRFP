# Request 78: why pooled context hurts clustering, and a barrier version of the mixed-label test

Written 26 November 2026 by the paper agent. The manuscript is in
`~/Documents/TACsy/paper/synrfp_paper`; do not edit it. Export the files named here and
report the branch and commit. Experiment A refits k-means only. Experiment B needs no fitting.

## Why

Supplementary Section S3.4 now reports two results from requests 74 and 77.
- In classification, the share of reactions in exact signed-count classes with more than one
  label predicts where pooled context helps the MLP (Spearman rho = 0.63, p = 0.016, 14 endpoints).
- In clustering it does not (rho = 0.13, p = 0.68, 12 endpoints). Pooled context lowers k-means AMI
  on 10 of 12 endpoints, including TPL B (-0.093) and Schneider B (-0.070), where it raises MCC.

The paper offers an untested reading: a supervised learner can weight the pooled counts only where
they separate classes, whereas k-means adds them to every distance next to the change at the reaction
center. Experiment A tests this. Experiment B asks the same question as request 74 for activation
barriers, which have no class labels.

## Experiment A (required): does the pooled channel dilute k-means distances?

Use the factorial setting of the paper: WL radius 2, parity, 2,048 bits, fingerprint seed 1, the 12
clustering endpoints, all rows, k = reference class count, the same initialisation seeds as the
published clustering factorial.

1. Channel weighting. Build D+U with D and U in separate 1,024-bit blocks as published, then scale the
   U block by w in {0, 0.25, 0.5, 0.75, 1} before k-means (w = 1 is the published D+U, w = 0 removes U
   but keeps the 1,024-bit D block). Report AMI per endpoint and w, mean and SD over seeds. Also report
   D alone at 2,048 bits as the published reference.
2. Distance decomposition, no fitting. For each endpoint and each block (D, U), report the mean squared
   Euclidean distance between reactions of the same reference class and of different classes, from at
   most 2,000 sampled pairs per type with a fixed seed, and their ratio (between over within). A block
   whose ratio is close to 1 adds distance without separating classes.
3. One short note: does AMI rise as w falls on the endpoints where U lowers AMI, and is the U block's
   between-to-within ratio lower than the D block's? Yes or no per endpoint, with the numbers.

Outputs: `clustering_weight_scan.csv`, `clustering_distance_ratio.csv`, `clustering_note.md`.

## Experiment B (required): exact signed-count groups for activation barriers

For the nine barrier datasets of the benchmark (B97XD3, Cyclo, E2, E2/SN2, GDB7, RDB7, RGD1, SN2, SNAr),
group all reactions by their exact WL radius 2 signed count vector (no sketching), as in request 74.
Report per dataset:
- number of groups, share of reactions in groups with two or more reactions;
- within-group standard deviation of the barrier (pooled over groups with two or more reactions) and
  its ratio to the dataset standard deviation, in the dataset's units;
- the same two numbers after splitting each group by its exact D+U count vector;
- the reduction in pooled within-group variance from D to D+U, as a share of the dataset variance.

Then report the Spearman correlation across the nine datasets between that reduction and the published
mean RMSE reduction from D to D+U over the 18 matched constructions (paper Fig. S3, source data
`context_score_changes.csv`, column `barrier_rmse_reduction_pct`). This is the regression analogue of the
mixed-label share. One short note should say whether it predicts where context helps barriers.

Outputs: `barrier_groups.csv`, `barrier_note.md`.

## Constraints

Same as requests 74 to 77: pinned commits and hashes, package versions, no weights, no per-row predictions,
state anything that cannot be reproduced. Push on a new branch named `exp/78-...` and report the commit.
