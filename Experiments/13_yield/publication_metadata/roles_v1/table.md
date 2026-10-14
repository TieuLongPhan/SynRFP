| Configuration (2048 total bits) | BH random (10) | BH OOD (4) | Suzuki random (10) | NCS OOD (40) |
|---|---:|---:|---:|---:|
| Delta r2 | 0.8313 ± 0.0105 | 0.6361 ± 0.1269 | 0.8436 ± 0.0086 | 0.2943 ± 0.2212 |
| Delta r3 | 0.9507 ± 0.0030 | 0.6812 ± 0.1719 | 0.8457 ± 0.0085 | 0.3197 ± 0.2399 |
| Union r2 | 0.9481 ± 0.0051 | 0.6774 ± 0.1556 | 0.8461 ± 0.0076 | 0.3509 ± 0.2145 |
| Union r3 | 0.9508 ± 0.0035 | 0.6687 ± 0.1218 | <u>0.8483 ± 0.0132</u> | 0.3393 ± 0.2074 |
| Delta + union r2 | 0.9506 ± 0.0046 | 0.6799 ± 0.1303 | 0.8473 ± 0.0117 | 0.3489 ± 0.2084 |
| Delta + union r3 | <u>0.9511 ± 0.0042</u> | **0.6930 ± 0.1302** | **0.8514 ± 0.0079** | 0.3367 ± 0.2045 |
| Roles r2 | 0.9472 ± 0.0056 | 0.6848 ± 0.1292 | 0.8449 ± 0.0114 | **0.3691 ± 0.2277** |
| Roles r3 | **0.9522 ± 0.0028** | <u>0.6925 ± 0.1170</u> | 0.8458 ± 0.0082 | <u>0.3672 ± 0.2226</u> |
| DRFP r3 | 0.9478 ± 0.0040 | 0.6546 ± 0.1271 | 0.8404 ± 0.0096 | 0.2889 ± 0.2284 |

R² mean ± descriptive split-score standard deviation. Best mean is bold, second underlined; ranking uses unrounded means and does not imply statistical significance. All entries use the same CPU MLP, shared-valid rows, and original splits. Roles and delta + union use 1024 + 1024 bits; delta and union use 2048 bits. DRFP uses radius3, 2048 bits, rings=True. New variants are exploratory after prior test results were viewed. Split scores are correlated. NCS contains 22609 shared-valid rows from27497, including exclusion of all4312 DreherDoyle2018 rows. No configuration is promoted by this ranking.
