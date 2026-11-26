# ECReact class labels for the Request 75 figures

The map decodes all 247 released indices, with 8 at EC1, 63 at EC2 and
176 at EC3. The pinned upstream CLAIRE table provides both the integer
encodings and the annotated EC strings. The SynRXN curator copies its
integer encoding columns. Verification reproduced the curated row IDs,
reaction text, splitting, original positions and three integer labels
in order on all 185,734 released rows. [Pinned upstream table](https://github.com/phuocchung123/SynCat/blob/051e7bf9eca68067b6f8d47f7f29f3e855cc7c74/Data/raw/claire_full.csv.gz) · [Pinned SynRXN curation script](https://github.com/TieuLongPhan/SynRXN/blob/e7c03766bd6a405da468e281d922cf1beccdd56a/script/build_classification_dataset.py#L601) · [SynRXN v1.2.1 release](https://zenodo.org/records/22045145)

The CSV keeps the literal source values `-`, `3.A` and `3.A.1` in the
mapping. The dash represents a missing EC annotation in the upstream
table. The alphanumeric strings are source labels.

- [Complete index map](ecreact_label_map.csv) uses the level and index as the lookup key.
- [Case A](case_a_ec.csv) adds the true label, D prediction, D+U prediction
  and training partner label as EC strings to 36 existing ECReact rows.
  For the pentanamide case `A_ecreact_2_08`, index 20 is **EC 3.5** and
  partner index 6 is **EC 3.1**.
- [Case C](case_c_ec.csv) decodes the EC3 labels on both sides of 40 existing
  ECReact fingerprint pairs. The prediction fields are empty because the
  Case C archive contains a fingerprint comparison with no model predictions.

[Source and verification receipt](LABEL_AUDIT.json)
