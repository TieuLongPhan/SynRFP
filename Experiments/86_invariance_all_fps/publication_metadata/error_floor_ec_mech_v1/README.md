# Classification error floors on ECReact L1 and Mech L2

The denominator is the shared valid subset of each full study 65 endpoint. The encoder mask and labels are identical at every stage. The first stage groups by the exact archived reaction text. Exact D, U and D+U group by the full WL radius 2 token counts before sketching and replay Request 74's published results. The support stage retains D sign and U presence while dropping count magnitudes. The last stage groups by exact bytes of the released parity features with seed 1. The combined feature uses 1,024 bits per channel.

Within each class, the best constant prediction is the majority label. The error floor is the sum of the minority counts divided by the number of encoded rows. These are empirical error floors on this cohort. A sketch can merge classes of the same label without increasing this floor.

| Endpoint | Encoded / total | Source text | Exact D | D support | D folded | Exact D+U | D+U support | D+U folded |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ecreact_1 | 185,720 / 185,734 | 1.272884% | 1.462417% | 1.469416% | 1.469416% | 1.350958% | 1.380573% | 1.380573% |
| mech_2 | 31,667 / 31,667 | 0.000000% | 0.006316% | 0.006316% | 0.006316% | 0.000000% | 0.000000% | 0.000000% |

The detailed CSV gives incremental errors at each transition, the number of merged classes, and measurements for pooled-only and 1,024-bit channels.
