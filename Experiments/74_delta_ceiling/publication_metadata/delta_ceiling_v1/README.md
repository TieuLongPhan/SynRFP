# Exact count ceilings for classification

Study 65's native cleaned classification panels supply the rows and labels.
The analysis uses full occurrence counts before any sketching. The frozen
encoder refusals are excluded from the ceilings and retain their original
classification benchmark status. Four tokenizer/radius profiles are reported.

| Endpoint | Encoded / study 65 | D mixed-label row share | D oracle accuracy | D+U oracle accuracy | D+U minus D oracle (pp) | Published MLP MCC change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ecreact_1 | 185,720 / 185,734 | 8.749% | 98.538% | 98.649% | +0.111 | +0.013 |
| ecreact_2 | 185,720 / 185,734 | 11.584% | 97.562% | 97.764% | +0.201 | +0.013 |
| ecreact_3 | 185,720 / 185,734 | 14.691% | 96.071% | 96.514% | +0.443 | +0.019 |
| mech_1 | 31,673 / 31,673 | 0.038% | 99.991% | 100.000% | +0.009 | -0.010 |
| mech_2 | 31,667 / 31,667 | 0.013% | 99.994% | 100.000% | +0.006 | -0.007 |
| schneider_b | 49,975 / 50,000 | 1.867% | 99.290% | 100.000% | +0.710 | +0.021 |
| schneider_ub | 50,000 / 50,000 | 0.010% | 99.996% | 100.000% | +0.004 | -0.004 |
| syntemp_r0 | 43,439 / 43,441 | 0.000% | 100.000% | 100.000% | +0.000 | -0.005 |
| syntemp_r1 | 43,439 / 43,441 | 0.000% | 100.000% | 100.000% | +0.000 | -0.011 |
| syntemp_r2 | 43,439 / 43,441 | 0.269% | 99.906% | 100.000% | +0.094 | -0.011 |
| tpl_b | 445,043 / 445,115 | 7.322% | 98.539% | 99.984% | +1.445 | +0.033 |
| tpl_ub | 445,111 / 445,115 | 3.096% | 99.197% | 99.984% | +0.787 | +0.026 |
| uspto50k_b | 49,840 / 50,016 | 0.102% | 99.974% | 100.000% | +0.026 | -0.010 |
| uspto50k_ub | 50,016 / 50,016 | 0.086% | 99.976% | 100.000% | +0.024 | -0.013 |

The MCC differences average the 18 tokenizer, radius and sketch settings
in the pinned paper source table. The oracle values above use WL radius 2.
These are two different quantities. A near-perfect full-cohort oracle can
arise from singleton groups and offers little information about prediction
on an exact class absent from fitting.

For each of the 25 saved grouped partitions, the seen-class ceiling uses
only test rows whose exact class appears among the retained MLP optimizer
fitting rows. It sums the majorities of their held-out label histograms.
The summary pools eligible test events over the 25 partitions and reports
their fraction of all encoded test events. The fold file gives denominators
and ceilings separately. These values use held-out labels to define an oracle.

D+U and D+S induce identical exact partitions because U = |D| + 2S.
The implementation independently checks that equivalence for every endpoint
and tokenizer. Their finite-width sketches can behave differently.

Group IDs are ordinals scoped to endpoint, tokenizer and channel set.
The ECReact examples use the smallest distinct-label pair per exact D class
and rank those pairs by total heavy atoms on both reaction sides.

See [endpoint_summary.csv](endpoint_summary.csv), [fold_summary.csv](fold_summary.csv),
[group_table.csv.gz](group_table.csv.gz), [examples.csv](examples.csv), and [AUDIT.json](AUDIT.json).
