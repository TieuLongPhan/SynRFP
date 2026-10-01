# Validation suite: what is run and why

This folder adds four studies to the archived benchmark. They answer the
questions the archived results leave open, on the same data and splits, with
5-NN and MLP only. Execution steps are in [RUNBOOK.md](RUNBOOK.md); all
settings are in [study.yaml](study.yaml).

## Questions

| # | Question | Script | Decision it feeds |
| --- | --- | --- | --- |
| 1 | Do the competitive margins survive when every encoder receives the same canonical input? | `00_setup.py`, `01_input_audit.py`, existing `featurize.py` / `classify.py` / `mlp_gpu_queue.py` per policy | Gate A |
| 2 | Which construction choice produces the gain: cancellation, multiplicity, direction, or folding? | `02_attribution.py` | Gate B |
| 3 | Do representations separate on a cross-database retrieval task with automatic labels? | `03_retrieval_proxy.py`, `03b_retrieval_rxnfp.py` | Gate C |
| 4 | Is localization on the reaction centre independent of the allocation convention, and what does sketching cost in collisions? | `04_locality.py` | Interpretation claims |
| 5 | What do encoding and exact search actually cost? | `05_cost.py`, `05b_cost_rxnfp.py` | Cost claims |

## Why the first question exists

The archived DRFP baseline received raw reaction strings. DRFP fingerprints
of chemically identical but differently written strings can differ; the
archived mechanism tables in particular carry bracketed atoms left over from
atom-map removal. On a laptop pilot (5-NN, five grouped folds, Mech L2),
canonical input raised DRFP from 0.889 to 0.935 MCC while SynRFP stayed at
0.965, so the margin shrank from about 0.075 to about 0.030. One endpoint and
five folds do not settle the comparison; the full rerun does.

## Input policies

| Policy | String given to every encoder |
| --- | --- |
| `raw` | As in the source table (archived protocol) |
| `n1` | Atom maps removed; RDKit canonical isomeric SMILES per component; components sorted within each field. No tautomer, salt, charge, or protonation changes. **Primary.** |
| `n2` | `n1`, then components occurring identically on both sides are removed. A common spectator policy for every encoder; 5-NN sensitivity arm. |

No benchmark table uses the middle (agent) field, so "agent policy" reduces to
how unchanged components on both sides are treated, which `n2` isolates.
If RDKit cannot parse a component, the raw string is kept and counted.
All policies share rows, labels, scaffold groups, and split codes, so every
cross-policy difference is paired on identical splits.

## Attribution design

Six payloads from the same side counts (set or multiset difference, unsigned
or signed, and two pooled-content controls), two folding rules (parity, OR),
two bit lengths (1024 primary, 2048 sensitivity), three hash seeds (all
reported), two vocabularies (SynRFP WL tokens; DRFP shingles pushed through
the same payloads), an exact unhashed reference, and two learners. Each
contrast changes one thing. The script first asserts that its
multiset-signed/parity payload reproduces the library fingerprint bit for bit.

Predeclared contrasts: signed minus unsigned (within set, within multiset);
multiset minus set (within unsigned, within signed); change minus pooled;
parity minus OR; exact minus folded. A formal table reports, per payload, the
share of reactions whose key set is unchanged by reversal, by an added
spectator on both sides, and by doubling every component.

## Retrieval proxy design

Queries: ECReact reactions sourced from BRENDA, PathBank, and MetaNetX.
Reference: a frozen Rhea release, one candidate per master reaction, scored by
the better of its two written directions. Relevance: EC agreement to the third
digit. Same-reaction candidates are removed per query. The primary metric is
precision at ten, averaged within EC class and then across classes, with
failed queries scored zero. Uncertainty is a paired cluster bootstrap over EC
numbers. Queries are split by EC number into a development part (20 %) and a
final part scored once.

This is a proxy. EC agreement is an automatic label; it does not establish
that a retrieved reaction is a chemically useful analogue, and it says nothing
about enzyme activity. Its job is to show whether the methods separate before
expert adjudication is commissioned. The query records are typically
unbalanced single-product strings while Rhea is balanced, so the task also
tests tolerance to different record conventions.

## Gates (frozen in study.yaml)

| Gate | Passes when |
| --- | --- |
| A, common input | Under `n1`: SynRFP minus DRFP 5-NN family-weighted mean MCC ≥ 0.010, the leave-one-family-out minimum > 0, and the MLP family-weighted mean ≥ −0.005 |
| B, attribution | Each of signed − unsigned (multiset), multiset − set (signed), and change − pooled is positive on ≥ 3 of 4 endpoints with 5-NN and on ≥ 2 with MLP (WL vocabulary) |
| C, retrieval proxy | On the final part: SynRFP exceeds the best label-free baseline by ≥ 0.03 macro P@10 with the bootstrap interval above 0, or is within 0.02 of the best learned encoder |

A failed gate is a result. It changes which claims the manuscript makes; it
is not a reason to alter the protocol.

## Not in this suite

XGBoost; the 162-configuration factorial (already complete); expert-judged
retrieval (commissioned only if Gate C passes); learned encoders other than
RXNFP (add their feature files as `runs/retrieval_proxy/features/<name>.npz`
with arrays `Q`, `C`, `n_bits=0`, `q_ok`, `c_ok` and they are scored
automatically).
