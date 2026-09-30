# Paper-ready SynRFP interpretation

SynRFP represents a reaction through the signed difference between product and
reactant graph-token counts. This construction suppresses unchanged spectators
and preserves reaction direction before hashing. The controlled perturbations
show both properties directly: adding the same spectator to both sides leaves
SynRFP unchanged, while reversing the reaction changes SynRFP. An unsigned
delta remains invariant under reversal, separating cancellation from direction
encoding.

The accompanying conceptual schematic is rendered at
`results/synrfp_mechanism_schematic.png`; it should be paired with the
quantitative perturbation results rather than presented as an independent
chemical validation.

The performance pattern is consistent with that design. In the complete
8,100-cell kNN factorial, signed delta has mean MCC 0.9262 versus 0.7475 for
the matched union representation, with a positive contrast on all ten
endpoints. The effect is largest for Mechanistic L1/L2 and USPTO50K UB and is
small for ECReact L1/L3, so the result should be described as chemistry-
dependent rather than universally optimal. Clustering gives the same direction
on the small benchmark, while large-data results include explicit exceptions:
DRFP leads on TPL UB and RDKit difference leads SynRFP on ECReact L1 ARI.

The explanatory evidence is supported by robustness controls. SynRFP remains
ahead in paired scaffold subsamples, remains ahead after shared-valid-row and
L2-normalization controls on Schneider B, and changes by about 5.7% of bits
under the tested hash-seed perturbations. The corrected reaction-centre audit
finds a mean occurrence-level centre fraction of 0.8749 across 31,673
mechanistic reactions. SynRFP also uses a substantially smaller estimated
feature payload than the dense RDKit-difference and RXNFP references. These
results explain why the representation can be compact and useful without
claiming that every hashed bit is uniquely interpretable.

An additional perturbation control sampled 2,000 reactions from each of
Schneider B, USPTO50K B, ECReact L1, and Mechanistic L2. SynRFP was identical
after randomized atom and component serialization on 100% of sampled rows
(shared-valid rates 99.65%--100%), while reversal identity was 0--0.35% and
mean reversal similarity stayed below 0.04. DRFP was reversal-invariant on all
rows, but its serialization invariance fell to 0--40.3% depending on the
dataset. This separates order invariance from reaction-direction sensitivity
and is reported as a controlled representation property, not as a claim about
chemical equivalence under reversal.

The classification ranking remains provisional until the queued XGBoost TPL
cells and factorial XGBoost/MLP matrices finish. Final ranking tables must be
regenerated after those matrices pass the completeness audit.

The current model-level stress cases are informative rather than random noise:
TPL B/UB and fine-resolution SynTemp are the recurring classification losses
against DRFP, while RDKit difference is the closest XGBoost comparator. The
MLP convergence audit gives SynRFP a median stopping epoch of 28, so its
current ranking does not show a general sign of under-training. These cases
should be reported as applicability limits and revisited only after the final
XGBoost and MLP matrices are complete.
