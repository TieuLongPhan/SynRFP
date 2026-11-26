# Exact count classes and activation barriers

The fraction of total barrier variance resolved by adding pooled context had a
positive rank association with the published RMSE benefit across the nine
datasets (Spearman ρ = 0.770, exact two-sided permutation p = 0.0195).
The degrees-of-freedom-adjusted pooled-variance contrast gave a weaker
association (ρ = 0.636, p = 0.0718). The whole-cohort statistic identifies
where pooled context helped in this benchmark. Its sensitivity to the variance
definition and the nine-dataset sample limit predictive claims.

| Dataset | Rows in repeated D classes | Rows in repeated D+U classes | Total variance resolved | Published mean RMSE reduction |
| --- | ---: | ---: | ---: | ---: |
| B97XD3 | 0.220% | 0.141% | 0.0097% | −1.501% |
| Cyclo | 3.492% | 3.492% | 0% | −0.001% |
| E2 | 72.943% | 72.943% | 0% | −0.577% |
| E2/SN2 | 78.786% | 75.338% | 6.613% | +15.424% |
| GDB7 | 0.034% | 0.017% | 0.0006% | −0.459% |
| RDB7 | 1.736% | 1.719% | 0.0005% | −3.475% |
| RGD1 | 49.736% | 45.225% | 5.310% | +4.431% |
| SN2 | 81.914% | 76.620% | 10.796% | +27.334% |
| SNAr | 67.992% | 44.732% | 6.028% | +11.745% |

For both exact WL r2 partitions, the pooled within-class SD is
√[SSE/(N−G)], where SSE sums squared deviations from each class mean, N
is the row count, and G is the number of classes. Singleton classes
contribute zero to SSE and to N−G. The CSV includes the ratio to the
dataset sample SD. The main reduction statistic divides the decrease in
SSE by the total dataset sum of squares. This measures the portion of
dataset variation resolved by refining D with U, keeping the denominator
fixed. The sensitivity analysis differences the two pooled variance estimates
and divides by the dataset sample variance.

The RMSE comparison averages the 18 matched construction contrasts from Fig. S3
on each paper endpoint. Positive values favor D+U. The permutation tests
enumerate all 9! permutations of those nine endpoint ranks and are descriptive.
All 419,149 rows reproduce both archived WL parity feature arrays, including the
44 RGD1 reactions recovered by omitting the unused optional stereo registry.
Raw SDs use the release's native units. The variance fractions are dimensionless.

[Group and dispersion table](barrier_groups.csv) · [Source hashes and correlations](barrier_AUDIT.json)
