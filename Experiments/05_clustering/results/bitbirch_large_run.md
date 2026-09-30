# Large-data BitBIRCH extension

Runs BitBIRCH only on `tpl_ub`, `tpl_b`, `ecreact_1`, `ecreact_2`, and `ecreact_3`.
It uses the existing threshold sweep (0.30–0.90), branching factor 50, and diameter
merge criterion. It is intentionally separate from the default clustering suite:
k-means and HDBSCAN are not being run on these large tables.

Eligible fingerprints are SynRFP, DRFP, and RDKit structural fingerprints; BitBIRCH
requires packed binary features, so it excludes dense RDKit-difference and RXNFP-MLM.
