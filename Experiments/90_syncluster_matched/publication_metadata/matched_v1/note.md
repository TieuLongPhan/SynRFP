# Matched SynCluster comparison on USPTO50K UB

The cohort contains the 43,381 admitted SynCluster author records with all ten reference classes, in archived train-then-validation order. Author positions identify records because the archived ID column repeats (20,781 distinct IDs). All fingerprint arms score every admitted row. The five baselines use the released recipes; each reuses 43,040 exact-input rows and encodes the other 341 from the author-normalized inputs. Each applicable k-means comparison uses seeds 0–9 and the frozen MiniBatchKMeans settings. The optional template-count k-means comparison preserves all original signed counts in a sparse 2,231-column matrix.

| Method | Algorithm | Requested k or distance cutoff | Occupied clusters | AMI |
|---|---|---:|---:|---:|
| native raw | Butina | 0.600 | 378.0 | 0.4762 |
| native residual merge | Butina + residual merge | 0.600 | 77.0 | 0.4648 |
| SynRFP standard | MiniBatchKMeans | 147 | 147.0 | 0.4222 ± 0.0064 |
| SynRFP standard | MiniBatchKMeans | 77 | 77.0 | 0.4152 ± 0.0052 |
| SynRFP standard | MiniBatchKMeans | 378 | 377.0 | 0.4069 ± 0.0023 |
| SynRFP standard | Butina | 0.600 | 7712.0 | 0.3290 |
| SynRFP standard · nearest grid count | Butina | 0.900 | 258.0 | 0.3444 |
| SynRFP standard · nearest refined count | Butina | 0.885 | 375.0 | 0.3561 |
| DRFP | Butina | 0.600 | 17549.0 | 0.2002 |
| drfp | MiniBatchKMeans | 378 | 376.8 | 0.3549 ± 0.0081 |
| crxnfp | MiniBatchKMeans | 378 | 378.0 | 0.4158 ± 0.0015 |
| rdkit_diff | MiniBatchKMeans | 378 | 378.0 | 0.3185 ± 0.0028 |
| rdkit_struct | MiniBatchKMeans | 378 | 378.0 | 0.1173 ± 0.0018 |
| rxnfp_mlm | MiniBatchKMeans | 378 | 378.0 | 0.3128 ± 0.0023 |
| SynCluster template counts | MiniBatchKMeans | 147 | 147.0 | 0.4814 ± 0.0023 |
| SynCluster template counts | MiniBatchKMeans | 378 | 362.9 | 0.4783 ± 0.0011 |

At k = 378, SynRFP AMI was 0.4069, compared with 0.4222 under its original k = 147 rule and 0.4762 for archived native SynCluster. Matching the requested cluster count alone left a gap. With both representations evaluated by k-means at k = 378, SynCluster template counts scored 0.4783 with 362.9 occupied clusters on average. At k = 147, both representations occupied all 147 clusters and template counts scored 0.4814. These are comparisons on the author cohort with its original corrected templates. The upstream template correction included source test inputs without class labels, as recorded in the archived protocol.

Among the other benchmark fingerprints at k = 378, crxnfp led with mean AMI 0.4158, compared with 0.4069 for SynRFP.

Native Butina uses a distance cutoff of 0.60, equivalent to a similarity threshold of 0.40 at the stated precision. Both bit-vector arms ran the original RDKit path. The full SynRFP replay reproduced its cluster centers and assignments exactly. The nearest count on the 0.05 grid was 258 at 0.90. Count-only refinement on the intervening interval gave 375 at 0.885. The refinement used cluster count alone. The CSV reports the complete tested grids and all requested AMI, ARI, NMI, and occupancy statistics.

K-means initialization was accelerated for some integer-valued matrices using exact integer squared distances. The optimizer, update arithmetic and seeds are those of the archived protocol. Four full-cohort controls verified identical centers and assignments. Single deterministic runs have empty standard deviations.

The run used CPU only and took 21.8 minutes from the cohort preparation receipt. Recorded service wall times were 61.5 s for the final primary/template k-means lane, 500.0 s for the final baseline lane, and 137.7 s for the sweep. The two full native Butina replays took 1127.8 s for SynRFP and 1128.3 s for DRFP. Service timings overlap. The final k-means lane timings exclude the initial unmodified fits reused from before the acceleration.
