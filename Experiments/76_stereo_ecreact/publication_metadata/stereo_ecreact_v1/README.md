# ECReact stereo-aware signed counts

All profiles use WL radius 2 and exact occurrence counts on the study 65
cohort. Fourteen previously refused ECReact source rows remain outside the
analysis. The ordinary D, U and D+U keys match Request 74 on every encoded
row. Oracle accuracy is the majority-label accuracy over exact classes.

| Endpoint | Profile | D ceiling | U ceiling | D+U ceiling | D+U minus D (pp) | D empty | Purity of empty D |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ecreact_1 | ordinary | 98.5376% | 98.6458% | 98.6490% | 0.1115 | 0.165% | 94.444% |
| ecreact_1 | stereo_tetra | 98.6808% | 98.7115% | 98.7142% | 0.0334 | 0.015% | 62.963% |
| ecreact_1 | stereo_ez | 98.7190% | 98.7250% | 98.7271% | 0.0081 | 0.000% | — |
| ecreact_2 | ordinary | 97.5625% | 97.7606% | 97.7638% | 0.2014 | 0.165% | 87.255% |
| ecreact_2 | stereo_tetra | 97.7891% | 97.8608% | 97.8635% | 0.0743 | 0.015% | 51.852% |
| ecreact_2 | stereo_ez | 97.8495% | 97.8801% | 97.8823% | 0.0328 | 0.000% | — |
| ecreact_3 | ordinary | 96.0710% | 96.5060% | 96.5136% | 0.4426 | 0.165% | 43.464% |
| ecreact_3 | stereo_tetra | 96.6062% | 96.7106% | 96.7182% | 0.1120 | 0.015% | 51.852% |
| ecreact_3 | stereo_ez | 96.7020% | 96.7343% | 96.7413% | 0.0393 | 0.000% | — |

| Stereo profile | Nonempty D among 15 | D separates the saved collision partner |
| --- | ---: | ---: |
| stereo_tetra | 7/15 | 10/15 |
| stereo_ez | 15/15 | 15/15 |

The 15 pairs were selected in Request 75 from saved original-model
outcomes. The new checks compare representations and empirical label
ceilings. Each stereo digest is rooted in its connected molecule.
The additional atom or bond label can carry configuration elsewhere
in the rooted molecular certificate. See [token profiles](token_profiles.md).

See [endpoint_summary.csv](endpoint_summary.csv),
[empty_delta_summary.csv](empty_delta_summary.csv) and
[stereo_only_rescues.csv](stereo_only_rescues.csv).
