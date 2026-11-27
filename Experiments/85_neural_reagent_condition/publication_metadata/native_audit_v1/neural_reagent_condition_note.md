# Native reagent and condition reference audit

T5Chem is usable as a matched native-split neural reagent reference. Seeds 42–44 score the same
14,215 frozen test rows as the fingerprints, with mean exact-set accuracy
16.98%, micro-F1 0.5062 and mean per-row Jaccard
0.4072. The CSV gives sample SD across three optimizer seeds on this one fixed
split. Parrot seeds 42 and 43 score all 65,946 fingerprint test rows, a strict subset
of the 68,075 author test rows. The 2,129 omitted author rows
are accounted for by the fingerprint benchmark's frozen eligibility rules
(2,127 input/condition overlaps and
2 component parse refusals). Canonicalizing molecules,
dropping empty slots, and ignoring their order within each role leaves both seeds at
2.50% exact set. The author's strict ordered five-slot matcher gives
2.49% for both on these same rows. Both saved fits return dichloromethane as the only
top-ranked condition on every reaction. The train-only canonical vocabulary covers every reference
slot on these test rows. This points to collapsed top-ranked predictions in the saved adaptation.
Seed 44 remains at its existing partial checkpoint. Exclude these fits as a published-Parrot
reference. The published Parrot-LM-E obtained 26.91% ordered top-1 on the author test set with an
additional fivefold SMILES augmentation stage [Wang et al., 2023, Table 1](https://yuquanli.github.io/pdf/paper_2023b.pdf).
This audit rescored the local unaugmented adaptation's saved candidates.

The study23 and study29 set metrics preserve the benchmark role-labelled target identities for
conditions. Within-role slot order and repeated components are discarded and an empty union has
Jaccard zero. The author score compares encoded identities in the original five slot positions,
including its empty category. The original helper was replayed from the pinned author source on
every distinct observed five-slot agreement pattern and weighted by its row count. The
canonical-ordered companion measures the effect of canonical SMILES while keeping slot positions.
