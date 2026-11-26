# Pooled-channel weighting and k-means

The published construction factorial used MiniBatchKMeans with K selected from the full row count. Request 78 specifies K equal to the reference class count. Both scans use the original five initialization seeds and MiniBatchKMeans settings on the same 12 full factorial panels. The D2048 reference and all weight settings were refitted with scikit-learn. Ten endpoints use sparse float32 matrices. The two TPL endpoints use dense updates with a packed-bit k-means++ initializer. Its squared distances and selected centres match native initialization on validation inputs. Both layouts preserve the exact weighted feature values. The factorial-K rows also provide the archived dense D2048 and w = 1 AMI for direct comparison. Of those 120 matched anchors, 91 agree at 1e-12 AMI. The maximum absolute difference is 0.0025 AMI. Sparse accumulation and the recorded runtime can affect cluster assignments, so the weighting contrasts compare fits using a common layout within each endpoint.

All settings use WL radius 2, parity and fingerprint seed 1. At w = 0 only the 1,024-bit signed block remains active. D2048 uses all 2,048 bits for signed changes. The distance decomposition samples 2,000 unordered pairs per type without replacement, conditional on the reference labels. It uses the same pairs for both blocks. The full-row cohort includes the original zero placeholders, and the CSV counts sampled pairs involving them.

U has the smaller between/within ratio on 11/12 endpoints and on 9/10 endpoints with a negative paper contrast. Both blocks can carry class structure. Weighting tests whether this distance pattern translates to higher AMI after clustering.

In the tables, a loss means the paper’s mean D+U minus D AMI over 18 matched constructions is negative. 'Zero helps' compares w = 0 with w = 1. 'Monotone' requires every step toward a smaller weight to preserve or improve mean AMI. The best weight is descriptive on the same cohort.

## Known K

On the 10 paper-loss endpoints, w = 0 exceeds w = 1 on 8 and the full scan is monotone on 1. For this WL r2 configuration, w = 1 is below D2048 on 9 endpoints. Among those, w = 0 improves on w = 1 on 9, with a monotone scan on 1.

| Endpoint | Paper loss | D2048 AMI | w = 1 AMI | w = 0 AMI | Best w | Zero helps | Monotone | D ratio | U ratio | U ratio lower |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | --- |
| ecreact_1 | Yes | 0.3294 | 0.3535 | 0.3303 | 1 | No | No | 1.147 | 1.172 | No |
| ecreact_2 | Yes | 0.5265 | 0.5322 | 0.5176 | 1 | No | No | 1.487 | 1.459 | Yes |
| ecreact_3 | Yes | 0.5872 | 0.5772 | 0.5860 | 0 | Yes | No | 2.138 | 2.092 | Yes |
| mech_1 | Yes | 0.3390 | 0.2872 | 0.3545 | 0.25 | Yes | No | 1.088 | 0.983 | Yes |
| mech_2 | Yes | 0.5970 | 0.5381 | 0.6043 | 0 | Yes | Yes | 1.180 | 1.003 | Yes |
| schneider_b | Yes | 0.6971 | 0.6554 | 0.7054 | 0 | Yes | No | 1.450 | 1.189 | Yes |
| schneider_ub | No | 0.5543 | 0.5633 | 0.5578 | 0.75 | No | No | 1.319 | 1.143 | Yes |
| syntemp_r0 | Yes | 0.6923 | 0.6277 | 0.6891 | 0.5 | Yes | No | 1.233 | 1.054 | Yes |
| tpl_b | Yes | 0.7423 | 0.6964 | 0.7412 | 0.5 | Yes | No | 2.000 | 1.282 | Yes |
| tpl_ub | No | 0.6340 | 0.6129 | 0.6282 | 0.75 | Yes | No | 1.616 | 1.233 | Yes |
| uspto50k_b | Yes | 0.3868 | 0.3028 | 0.4060 | 0.25 | Yes | No | 1.112 | 1.026 | Yes |
| uspto50k_ub | Yes | 0.3714 | 0.3017 | 0.3726 | 0.5 | Yes | No | 1.094 | 1.005 | Yes |

## Factorial K

On the 10 paper-loss endpoints, w = 0 exceeds w = 1 on 9 and the full scan is monotone on 2. For this WL r2 configuration, w = 1 is below D2048 on 10 endpoints. Among those, w = 0 improves on w = 1 on 9, with a monotone scan on 2.

| Endpoint | Paper loss | D2048 AMI | w = 1 AMI | w = 0 AMI | Best w | Zero helps | Monotone | D ratio | U ratio | U ratio lower |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | --- |
| ecreact_1 | Yes | 0.3201 | 0.3017 | 0.3204 | 0 | Yes | Yes | 1.147 | 1.172 | No |
| ecreact_2 | Yes | 0.5230 | 0.4965 | 0.5206 | 0 | Yes | No | 1.487 | 1.459 | Yes |
| ecreact_3 | Yes | 0.5860 | 0.5609 | 0.5842 | 0 | Yes | No | 2.138 | 2.092 | Yes |
| mech_1 | Yes | 0.4171 | 0.3876 | 0.4139 | 0.25 | Yes | No | 1.088 | 0.983 | Yes |
| mech_2 | Yes | 0.6126 | 0.5618 | 0.6149 | 0 | Yes | Yes | 1.180 | 1.003 | Yes |
| schneider_b | Yes | 0.7230 | 0.7161 | 0.7114 | 0.5 | No | No | 1.450 | 1.189 | Yes |
| schneider_ub | No | 0.6180 | 0.6183 | 0.6207 | 0.5 | Yes | No | 1.319 | 1.143 | Yes |
| syntemp_r0 | Yes | 0.6867 | 0.6747 | 0.6958 | 0 | Yes | No | 1.233 | 1.054 | Yes |
| tpl_b | Yes | 0.7369 | 0.7273 | 0.7341 | 0.25 | Yes | No | 2.000 | 1.282 | Yes |
| tpl_ub | No | 0.6072 | 0.6111 | 0.6105 | 0.75 | No | No | 1.616 | 1.233 | Yes |
| uspto50k_b | Yes | 0.4370 | 0.4224 | 0.4369 | 0.75 | Yes | No | 1.112 | 1.026 | Yes |
| uspto50k_ub | Yes | 0.4151 | 0.3920 | 0.4201 | 0 | Yes | No | 1.094 | 1.005 | Yes |

See the [weight scan](clustering_weight_scan.csv), [per-seed AMI](clustering_weight_cells.csv), and [distance ratios](clustering_distance_ratio.csv).
