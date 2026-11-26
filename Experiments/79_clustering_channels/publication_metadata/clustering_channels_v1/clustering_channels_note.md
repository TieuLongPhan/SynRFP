# Channel allocation in BitBIRCH and k-means

The scan contains **300 endpoint scores** from **250 native BitBIRCH fits** across all 12 endpoints. ECReact hierarchy levels reuse the same cluster assignments because their ordered reaction panels are identical. All published cohort rows and refusal placeholders are retained. Each construction uses WL radius 2, parity with fingerprint seed 1, and a 2,048-bit budget.

BitBIRCH used Tanimoto with the archived SynRFP threshold 0.35, branching factor 50 and diameter merging. The threshold had been selected by AMI on balanced USPTO50K using the historical 1,024-bit delta fingerprint. It was fixed across all endpoints and all five constructions in this scan. The five input-order seeds are 20260924 through 20260928, matched across constructions at each endpoint. BitBIRCH used no reference class count. The two k-means protocols retain their respective Request 78 K rules.

The BitBIRCH table gives arithmetic means and sample standard deviations across input orders. Singleton share counts reactions in singleton clusters. Runtime times the native fit and assignment extraction. Group means below give each of the six endpoint views the same weight, including the three related ECReact views.

| Method | Pooled setting | Reaction-centre ΔAMI vs D | Context ΔAMI vs D | Reaction-centre recovery vs full D+U | Context recovery vs full D+U |
|---|---|---:|---:|---:|---:|
| BitBIRCH | `du1024_1024` | -0.0684 | +0.0031 | +0.0000 | +0.0000 |
| BitBIRCH | `du1536_512` | -0.0664 | +0.0046 | +0.0020 | +0.0015 |
| BitBIRCH | `du1792_256` | -0.0755 | -0.0074 | -0.0071 | -0.0105 |
| k-means, reference K | `u1` | -0.0533 | -0.0148 | +0.0000 | +0.0000 |
| k-means, reference K | `u0.75` | -0.0093 | -0.0103 | +0.0440 | +0.0045 |
| k-means, reference K | `u0.5` | +0.0012 | -0.0015 | +0.0545 | +0.0133 |
| k-means, reference K | `u0.25` | +0.0096 | -0.0070 | +0.0630 | +0.0079 |
| k-means, reference K | `u0` | +0.0072 | -0.0013 | +0.0606 | +0.0135 |
| k-means, factorial K | `u1` | -0.0216 | -0.0138 | +0.0000 | +0.0000 |
| k-means, factorial K | `u0.75` | -0.0102 | -0.0101 | +0.0114 | +0.0037 |
| k-means, factorial K | `u0.5` | +0.0003 | -0.0055 | +0.0219 | +0.0083 |
| k-means, factorial K | `u0.25` | +0.0002 | -0.0099 | +0.0218 | +0.0039 |
| k-means, factorial K | `u0` | +0.0026 | -0.0025 | +0.0243 | +0.0113 |

- **BitBIRCH**. Full D+U lowered AMI on 7/12 endpoints. The group changes versus D were -0.0684 for reaction-centre endpoints and +0.0031 for context endpoints. The context group gained more in 0 of 2 lower-U settings.
- **k-means, reference K**. Full D+U lowered AMI on 9/12 endpoints. The group changes versus D were -0.0533 for reaction-centre endpoints and -0.0148 for context endpoints. The context group gained more in 0 of 4 lower-U settings.
- **k-means, factorial K**. Full D+U lowered AMI on 10/12 endpoints. The group changes versus D were -0.0216 for reaction-centre endpoints and -0.0138 for context endpoints. The context group gained more in 0 of 4 lower-U settings.

For k-means, `u0` retains a 1,024-bit D block with U weighted to zero. The D reference uses 2,048 bits. Unequal BitBIRCH allocations change the fold modulus as well as the share of U bits. Their results therefore reflect both effects. AMI changes are descriptive on these fixed panels and at this threshold.

See [BitBIRCH summary](bitbirch_scan.csv), [individual orders](bitbirch_order_scores.csv) and [endpoint and group contrasts](group_summary.csv).
