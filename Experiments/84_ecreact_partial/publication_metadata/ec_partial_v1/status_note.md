# ECReact results available from Request 83

Snapshot at **2026-11-26 14:24 UTC**. Request 83 continues on the workstation.

**Experiment A is partial.** 54/150 matched fits have verified predictions. Level 1 has 4/50, Level 2 has 0/50, and Level 3 has 50/50. The Level 3 estimates are complete on the common five-repeat, five-fold outer schedule. Pooled context raised Level 3 MCC by **+0.0171** [+0.0108, +0.0239] and reduced error by **+0.0158** [+0.0100, +0.0223], using 10,000 paired scaffold-group draws.

The paired differences with Level 1 require the other levels to close. The completed training-rule results are included for all 75 fold-by-contrast checks. [Per-fit inventory](ec_level_fit_status.csv)
At the observed CPU rate, the remaining common-split fits and paired intervals are expected in roughly two to four hours from this snapshot. This estimate depends on early stopping and workstation load.

**Experiment B mapping is complete.** The pinned mapper accepted 185,343/185,734 under the original mapping policy. The stricter atom-conserving EnzymeMap label is defined for 4,017/185,734 (2.16%) of rows. Removing 603 exact or reverse EnzymeMap overlaps leaves 3,414. Edit labels agree with the unique available author mapping in 544/595 comparisons (91.4%). Two rows have multiple competing author edit labels.

**Experiment B fits are complete.** Both cohorts passed all 30 fits, and the final held-out results are in [the target-pair table](ecreact_target_pair.csv).
- all_mapped, EC2-minus-edit error reduction **+0.0441** [-0.0108, +0.1021].
- without_enzymemap, EC2-minus-edit error reduction **+0.0316** [-0.0296, +0.0914].
