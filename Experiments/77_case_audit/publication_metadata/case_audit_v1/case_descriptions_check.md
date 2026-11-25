# Audit of the Request 75 case descriptions

- **EnzymeMap — No.** All five illustrated losses carry EC 5.1.3. In the full eligible loss population, only 79/236 (33.5%) carry that annotation, and 82/236 (34.7%) carry the broader EC 5.1 label. The featured gain subclasses EC 2.6, 4.3 and 3.4 account for 4, 3 and 2 of 546 gains, respectively. EC 5.1.3 covers enzymes acting on carbohydrates and derivatives. [IUBMB enzyme nomenclature](https://iubmb.qmul.ac.uk/enzyme/EC5/1/)
- **QuantumPioneer — Yes for hydrogen abstraction. No for large barrier errors as a population description.** All 7,467 gains and 6,509 losses satisfy the structural hydrogen-transfer check. Median barrier absolute errors for D and D+U are 1.89 and 1.80 kcal/mol on gains and 1.84 and 1.87 on losses. Among the illustrated cases, the corresponding medians are 9.47 and 4.87 on gains and 1.93 and 2.77 on losses.

## EnzymeMap

A gain requires D+U alone to predict the correct EC subclass in at least two
of three paired seeds. A loss applies the same rule to D alone. The audit
recovered 546 gains and 236 losses among 9,631 held-out reactions. EC 5.1.3
uses the third level of the original EC annotation on the frozen representative
source row.

| EC main class | Gains | Losses |
| --- | ---: | ---: |
| 1 | 3 (0.5%) | 0 (0.0%) |
| 2 | 436 (79.9%) | 59 (25.0%) |
| 3 | 70 (12.8%) | 46 (19.5%) |
| 4 | 8 (1.5%) | 24 (10.2%) |
| 5 | 3 (0.5%) | 98 (41.5%) |
| 6 | 26 (4.8%) | 9 (3.8%) |

| Label resolution | Pearson χ² | df | Conditional Monte Carlo p | Cramér’s V |
| --- | ---: | ---: | ---: | ---: |
| EC main class | 329.62 | 5 | 0.00002 | 0.649 |
| EC subclass | 412.42 | 27 | 0.00002 | 0.726 |

The conditional Monte Carlo calculation fixes the row and column margins.
It uses 49,999 draws with seed 77. No draw reached the observed statistic for
either table, so the listed p-values are at the simulation resolution.
These tests describe a contrast between model-selected groups. Structural
dependence among rows can make the row-wise p-values overly precise.

| Bond-edit quantity | Gains | Losses |
| --- | ---: | ---: |
| True label absent from the opposite group | 113 (20.7%) | 34 (14.4%) |
| True no-edit label | 58 (10.6%) | 82 (34.7%) |
| Arms predict different bond-edit labels in at least two seeds | 286 (52.4%) | 71 (30.1%) |
| D+U alone correct on bond edits in at least two seeds | 95 (17.4%) | 3 (1.3%) |
| D alone correct on bond edits in at least two seeds | 61 (11.2%) | 54 (22.9%) |

The two groups contain different reactions. A true label counts as shared
when it occurs anywhere in both groups. The CSV gives all EC subclasses
and true bond-edit labels, counts and shares. [EnzymeMap composition](enzymemap_composition.csv)

## QuantumPioneer

The frozen selection target is reaction enthalpy. Eligibility uses normalized
squared-error changes with the same sign in at least two of three seeds and in
the seed mean, using the training target variance to normalize. Barrier and
enthalpy summaries below describe those same eligible groups.

All 16,329 held-out rows contain a closed-shell component and a monoradical on
each side. Adding H at each radical centre gives the closed-shell component
on the other side when comparing constitutions. This identifies the
abstracting radical independently of component order. The species identity
uses canonical radical SMILES with declared stereo retained.
[QuantumPioneer kinetics schema](https://zenodo.org/records/20060606/files/SCHEMA.md?download=1)

| Attacking radical | Gains | Losses |
| --- | ---: | ---: |
| `[O]O` | 5,230 (70.04%) | 4,856 (74.60%) |
| `[O]OCO` | 2,167 (29.02%) | 1,626 (24.98%) |
| All other species | 70 (0.94%) | 27 (0.41%) |

| Quantity (kcal/mol) | Gains median [Q1, Q3] | Losses median [Q1, Q3] |
| --- | ---: | ---: |
| True barrier | 16.94 [14.25, 20.41] | 17.14 [14.15, 20.69] |
| True enthalpy | 8.68 [5.82, 12.47] | 8.70 [5.97, 12.64] |
| Barrier absolute error, D | 1.89 [0.88, 3.40] | 1.84 [0.84, 3.33] |
| Barrier absolute error, D+U | 1.80 [0.84, 3.23] | 1.87 [0.87, 3.41] |
| Enthalpy absolute error, D | 1.71 [0.94, 2.79] | 0.77 [0.33, 1.68] |
| Enthalpy absolute error, D+U | 0.74 [0.31, 1.57] | 1.67 [0.91, 2.81] |

The error table uses the absolute error of each reaction’s three-seed mean
prediction. The CSV also gives the alternative measure that averages the
absolute errors of the three seeds, along with all species and energy
ranges. [QuantumPioneer composition](quantumpioneer_composition.csv)
