# EHReact activity comparison

Each value is mean +/- SD across 15 assay-repeat scores: three eligible assays and five repeated stratified group-CV splits. MCC uses a 0.5 probability threshold; AP and ROC-AUC use probabilities. Assays with fewer than ten active or inactive reactions were excluded.

| Model | Fingerprint | MCC | AP | ROC-AUC |
|---|---|---:|---:|---:|
| knn | synrfp-wl-r2-b1024-parity-delta | 0.152 +/- 0.141 | 0.384 +/- 0.123 | 0.694 +/- 0.126 |
| knn | drfp | 0.156 +/- 0.061 | 0.403 +/- 0.139 | 0.710 +/- 0.085 |
| knn | rdkit_struct | 0.247 +/- 0.139 | 0.413 +/- 0.184 | 0.692 +/- 0.065 |
| knn | rdkit_diff | 0.250 +/- 0.196 | 0.402 +/- 0.234 | 0.687 +/- 0.090 |
| knn | rxnfp | 0.112 +/- 0.092 | 0.389 +/- 0.116 | 0.706 +/- 0.121 |
| knn | rxnfp_mlm | 0.184 +/- 0.201 | 0.412 +/- 0.108 | 0.699 +/- 0.134 |
| linear | synrfp-wl-r2-b1024-parity-delta | 0.457 +/- 0.097 | 0.557 +/- 0.178 | 0.795 +/- 0.117 |
| linear | drfp | 0.405 +/- 0.084 | 0.548 +/- 0.187 | 0.817 +/- 0.097 |
| linear | rdkit_struct | 0.388 +/- 0.115 | 0.555 +/- 0.178 | 0.800 +/- 0.103 |
| linear | rdkit_diff | 0.379 +/- 0.194 | 0.503 +/- 0.220 | 0.781 +/- 0.114 |
| linear | rxnfp | 0.387 +/- 0.176 | 0.516 +/- 0.207 | 0.784 +/- 0.140 |
| linear | rxnfp_mlm | 0.373 +/- 0.159 | 0.550 +/- 0.194 | 0.796 +/- 0.116 |
| mlp | synrfp-wl-r2-b1024-parity-delta | 0.105 +/- 0.149 | 0.407 +/- 0.226 | 0.680 +/- 0.082 |
| mlp | drfp | 0.119 +/- 0.145 | 0.366 +/- 0.235 | 0.609 +/- 0.088 |
| mlp | rdkit_struct | 0.095 +/- 0.128 | 0.379 +/- 0.227 | 0.640 +/- 0.035 |
| mlp | rdkit_diff | 0.124 +/- 0.132 | 0.332 +/- 0.138 | 0.529 +/- 0.141 |
| mlp | rxnfp | 0.119 +/- 0.148 | 0.417 +/- 0.135 | 0.714 +/- 0.128 |
| mlp | rxnfp_mlm | 0.171 +/- 0.144 | 0.431 +/- 0.180 | 0.718 +/- 0.110 |

Eligible assays: alcoholdehydrogenase E-2 (65 reactions, 28 active); csp2_csp3_coupling Class (694, 28); dehalogenase E-1 (46, 13).

Detailed per-assay values are in `per_assay.csv`; replicate-level values are in `../runs.csv`.
