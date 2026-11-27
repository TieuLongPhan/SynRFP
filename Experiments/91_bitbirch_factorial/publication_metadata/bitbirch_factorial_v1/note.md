# Full BitBIRCH factorial and benchmark

The study completed 49,400 CPU fits across ten full populations, with
59,280 endpoint scores. ECReact levels share each clustering fit,
with separate scores for their reference labels. The three repaired TPL B rows use the
updated released caches. All other frozen cohort rows, including refusal placeholders,
are included. Morgan uses the corrected graph-native profile.

The sweep uses Tanimoto, branching factor 50, diameter merging, thresholds 0.20 to
0.80 and matched input-order seeds 20260924–20260928. The first operating point uses
threshold 0.35. The second minimizes the absolute difference between mean occupied
clusters and the k-means row-count rule, with ties resolved by the lower threshold.
In 1.5% of the reported second operating points, the occupied count is within
10% of that rule. The full signed differences are in `operating_points.csv`.
No class labels enter threshold selection. The operating-point table also includes
the released row-count k-means AMI on the same rows for the factorial and the two
native-bit baselines.

The baseline rows record their conversion explicitly. DRFP and RDKit structural use
native bits. RDKit difference maps nonzero counts to one. RXNFP-MLM uses positive
coordinates. The binarized results describe those projections. Singleton share is the
fraction of observations that are alone in a cluster. Standard deviations describe
five input orders of each fixed cohort.

## Skipped representation

- crxnfp. Every valid continuous embedding maps to the same all-one vector under the nonzero rule.

## Runtime and reproducibility

The workstation uses a AMD Ryzen Threadripper 7970X 32-Cores. The main service used 22 CPU workers and took
4.76 wall-clock hours. The initial pilot results are reused
and timed in the cells. The full report contains 11,856 setting summaries.
The 60 overlapping Request 79 cells were replayed, with a
maximum absolute score difference of 5.5e-11. The variance
table uses the same balanced decomposition as the k-means factorial and includes all
interaction orders. The release manifest pins cache files, source code and software versions.
The grouped AMI calculation sums the same hypergeometric terms by cluster-size
multiplicity. Full-cohort controls verify agreement with the native sklearn metric.
