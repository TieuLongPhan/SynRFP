# ECReact target pairs

The full EC-level comparison holds out the same 185,734 rows across five repeats of five grouped folds. The EC level-3 schedule supplied the common outer partitions and seeds. We reused 50 level-3 fits and trained 100 level-1 and level-2 fits with the published 512-unit MLP recipe and the original target-dependent inner-validation procedure. The new fits use the archived CPU implementation of the same training arithmetic. The reused level-3 fits ran on CUDA. All three levels share the same outer test rows.

| Target | D → D+U MCC gain, 95% CI | D → D+U error reduction, 95% CI |
|---|---:|---:|
| L1 | +0.0214 [+0.0036, +0.0419] | +0.0139 [+0.0022, +0.0300] |
| L2 | +0.0155 [+0.0075, +0.0248] | +0.0139 [+0.0066, +0.0238] |
| L3 | +0.0171 [+0.0108, +0.0239] | +0.0158 [+0.0100, +0.0223] |
| L3_minus_L1 | -0.0043 [-0.0248, +0.0127] | +0.0019 [-0.0143, +0.0137] |
| L2_minus_L1 | -0.0059 [-0.0190, +0.0065] | -0.0000 [-0.0082, +0.0071] |

The training-only rule uses the unchanged thresholds of 5% of training rows, 30 eligible query groups and ten exact signed-count classes. Bootstrap intervals resample scaffold groups together across arms, targets and repetitions.

- L1_vs_L2 met the rule thresholds in 25/25 fits, with L1 in 4, L2 in 21.
- L1_vs_L3 met the rule thresholds in 25/25 fits, with L3 in 25.
- L2_vs_L3 met the rule thresholds in 25/25 fits, with L3 in 25.

The strict bond-edit analysis reused the full ECReact mapping archive from pinned RXNMapper v2. The original mapping accepted 185,343 rows under its own looser contract. The EnzymeMap classifier requires all atoms to map bijectively across the exact original sides. That leaves **4,017 rows (2.16%)**, or **3,414** after removing exact and reverse-direction matches to the full EnzymeMap source. Against the EnzymeMap author's maps, the bond-edit class agreed for **544/595 (91.4%)** rows with a unique reference label. Two additional rows have conflicting author edit labels. The frozen experimental EnzymeMap cohort contained no exact shared row.

Both strict-label cohorts use the original EnzymeMap feature arms, three optimizer seeds, scaffold grouping, split rule and paired group-weighted bootstrap. Group assignments were constructed on the full mapped set and held fixed when overlap was removed.

| Cohort | Test rows | EC2 error reduction | Edit error reduction | EC2 minus edit, 95% CI |
|---|---:|---:|---:|---:|
| all_mapped | 334 | +0.0474 | +0.0033 | +0.0441 [-0.0108, +0.1021] |
| without_enzymemap | 302 | +0.0570 | +0.0254 | +0.0316 [-0.0296, +0.0914] |

The training-only ordering was all_mapped → ec_level_2, without_enzymemap → ec_level_2. All_mapped is inconclusive. Without_enzymemap is inconclusive. The strict mapping subset defines the scope of this comparison.
