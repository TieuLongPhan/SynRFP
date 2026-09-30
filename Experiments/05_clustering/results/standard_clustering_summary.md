# Standard clustering benchmark

Final refresh: 2026-09-25. All K-means cells contain ten external seeds and
ten internal restarts per seed. The complete comparison includes all five
label-free fingerprints. BitBIRCH remains limited to the three packed binary
fingerprints by design.

Primary metric: adjusted mutual information (AMI) against each dataset’s primary label. The benchmark contains Schneider, USPTO-50k, SynTemp R0, and two Mech31k label levels.

## K-means

`k` equals the number of primary labels. Values are mean ± sample SD over ten
initialization seeds.

| Dataset | SynRFP | DRFP | RDKit structural | RDKit difference | RXNFP-MLM |
|---|---:|---:|---:|---:|---:|
| schneider_ub | **0.575 ± 0.011** | 0.571 ± 0.005 | 0.187 ± 0.004 | 0.438 ± 0.021 | 0.538 ± 0.004 |
| schneider_b | **0.719 ± 0.008** | 0.649 ± 0.009 | 0.218 ± 0.007 | 0.518 ± 0.021 | 0.490 ± 0.003 |
| uspto50k_ub | **0.402 ± 0.011** | 0.314 ± 0.016 | 0.099 ± 0.000 | 0.297 ± 0.013 | 0.262 ± 0.000 |
| uspto50k_b | **0.435 ± 0.019** | 0.334 ± 0.013 | 0.106 ± 0.001 | 0.297 ± 0.006 | 0.320 ± 0.008 |
| syntemp_r0 | **0.702 ± 0.007** | 0.594 ± 0.007 | 0.146 ± 0.002 | 0.323 ± 0.006 | 0.586 ± 0.002 |
| mech_1 | **0.363 ± 0.018** | 0.263 ± 0.018 | 0.087 ± 0.000 | 0.261 ± 0.009 | 0.080 ± 0.000 |
| mech_2 | **0.613 ± 0.009** | 0.475 ± 0.013 | 0.129 ± 0.001 | 0.301 ± 0.009 | 0.158 ± 0.003 |

## BitBIRCH

Threshold is calibrated by AMI on `uspto50k_b` and then frozen: SynRFP `0.35`, DRFP `0.35`, RDKit structural `0.65`. RXNFP-MLM is dense and therefore cannot use BitBIRCH.

| Dataset | SynRFP | DRFP | RDKit structural |
|---|---:|---:|---:|
| schneider_ub | 0.446 | 0.155 | 0.205 |
| schneider_b | 0.432 | 0.185 | 0.212 |
| uspto50k_ub | 0.195 | 0.030 | 0.099 |
| uspto50k_b | 0.279 | 0.142 | 0.103 |
| syntemp_r0 | 0.296 | 0.238 | 0.126 |
| mech_1 | 0.159 | 0.023 | 0.107 |
| mech_2 | 0.246 | 0.037 | 0.122 |

## HDBSCAN sweep (oracle)

Each cell is the best AMI among `min_cluster_size` {5, 10, 25, 50}; it is useful for diagnosing geometry but is label-tuned per dataset and should not be used for a strict ranking.

| Dataset | SynRFP | DRFP | RDKit structural | RDKit difference | RXNFP-MLM |
|---|---:|---:|---:|---:|---:|
| schneider_ub | 0.311 (m=10) | **0.319 (m=10)** | 0.123 (m=5) | 0.201 (m=10) | 0.158 (m=5) |
| schneider_b | **0.521 (m=50)** | 0.395 (m=25) | 0.133 (m=5) | 0.160 (m=5) | 0.268 (m=25) |
| uspto50k_ub | **0.215 (m=25)** | 0.150 (m=5) | 0.027 (m=5) | 0.007 (m=25) | 0.056 (m=50) |
| uspto50k_b | **0.239 (m=25)** | 0.154 (m=10) | 0.028 (m=5) | 0.034 (m=5) | 0.164 (m=10) |
| syntemp_r0 | **0.390 (m=50)** | 0.292 (m=10) | 0.045 (m=5) | 0.075 (m=5) | 0.229 (m=25) |
| mech_1 | **0.215 (m=10)** | 0.141 (m=5) | 0.029 (m=5) | 0.026 (m=25) | 0.052 (m=5) |
| mech_2 | **0.364 (m=25)** | 0.242 (m=5) | 0.046 (m=5) | 0.027 (m=25) | 0.069 (m=10) |
