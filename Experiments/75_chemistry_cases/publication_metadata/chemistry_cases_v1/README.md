# Chemistry cases for the Supplementary Information

[Selection rules](selection_rules.md) give the caps, tie breaks and scaffold checks.
[AUDIT.json](AUDIT.json) records input hashes, versions and candidate counts.

## A and B. Classification

| Case | Endpoint | Majority-vote qualified | Encoded | With required A witness | Exported |
| --- | --- | ---: | ---: | ---: | ---: |
| A | ecreact_1 | 3,670 | 3,670 | 181 | 12 |
| A | ecreact_2 | 4,802 | 4,802 | 154 | 12 |
| A | ecreact_3 | 5,942 | 5,942 | 484 | 12 |
| A | tpl_b | 12,667 | 12,646 | 5155 | 12 |
| A | schneider_b | 1,229 | 1,221 | 387 | 12 |
| B | mech_1 | 170 | 170 | — | 12 |
| B | mech_2 | 116 | 116 | — | 12 |
| B | syntemp_r1 | 408 | 408 | — | 12 |
| B | syntemp_r2 | 905 | 905 | — | 12 |
| B | uspto50k_b | 295 | 295 | — | 12 |
| B | uspto50k_ub | 414 | 414 | — | 12 |

[Case A](case_a.csv) contains 60 rescues with training witnesses. [Case B](case_b.csv) contains 72 failures.
The 1,024-only D screen found 0 width-specific collisions among 2,308 encoded failures.
See the [fold collision rates](fold_collision_summary.csv).

## C. DRFP and signed-count disagreements

| Dataset | Direction | Eligible unordered pairs | Representative buckets | Exported pairs |
| --- | --- | ---: | ---: | ---: |
| ecreact | drfp_equal_D_different | 29,838,483 | 521 | 20 |
| ecreact | D_equal_drfp_different | 98,351,881 | 3,439 | 20 |
| uspto50k | drfp_equal_D_different | 48 | 19 | 10 |
| uspto50k | D_equal_drfp_different | 95,808 | 3,635 | 20 |

[Case C](case_c.csv) contains both rows of every pair, decoded DRFP fragments,
and rooted WL environments for the differing signed tokens.
Among the 30 equal-DRFP/different-D pairs, 29 have identical pre-fold DRFP fragment sets and 1 share only the folded bits.

## D–F. Yield, condition ranking, and external tasks

- [Case D](case_d.csv) contains 2,368 recipe-matched pairs across 9 E3 holdouts and 4,736 assay appearances.
- [Case E](case_e.csv) contains all 3 alcohol gains and all 3 losses among the 36 substrates with saved scores. Each selected substrate has all 20 condition rankings.
- [Case F](case_f.csv) contains five gains and five losses per dataset. EnzymeMap had 546 eligible gains and 236 losses. QuantumPioneer had 7,467 eligible gains and 6,509 losses.

See the [alcohol stereochemistry probe](stereo_probe.md) for Request 74's bit-vector check.
