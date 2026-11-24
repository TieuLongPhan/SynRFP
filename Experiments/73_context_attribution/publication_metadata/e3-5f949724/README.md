# E3 exact-delta yield-gap results

The workstation run completed 27 fits across nine joint precursor holdouts and
2,368 recipe-matched yield gaps on the pinned Buchwald–Hartwig screen. The
manifest records the runner revision, input hash, environment and holdout sizes.

Gap MSE is in squared yield percentage points. Each class averages three pair
scores, and the overall result weights the three classes equally. The two
delta controls produced zero gaps within numerical tolerance.

| Exact-delta class | Zero-gap MSE | Pooled-context MSE | MSE reduction |
|---|---:|---:|---:|
| delta01 | 143.94 | 155.21 | -11.28 |
| delta02 | 406.65 | 155.97 | +250.68 |
| delta03 | 374.57 | 295.30 | +79.27 |
| Equal-class mean | 308.39 | 202.16 | +106.22 |

Pooled context reduced the equal-class mean gap MSE by 34.4%, with improvements
in two of three classes. These descriptive results concern one screen with
overlapping training sets. Each precursor form appears in two holdouts.

The release includes per-fit, per-pair and per-class scalar scores. The
completion receipt records hashes for the local splits and predictions.
