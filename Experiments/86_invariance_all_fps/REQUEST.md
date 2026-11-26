# Request 86: SMILES-rewriting and reversal audit for all seven fingerprints

Written 26 November 2026 by the paper agent. The manuscript is in
`~/Documents/TACsy/paper/synrfp_paper`; do not edit it. Report the branch and commit.

## Why

The SI table "Fingerprint agreement after SMILES rewriting and reaction reversal" comes from
`Experiments/07_robustness/robustness.py` (`run_perturb`). It covers only four fingerprints
(SynRFP 1,024-bit WL parity delta, DRFP, RDKit structural, RDKit difference) on Schneider B,
USPTO50K B, ECReact L1 and Mech L2. The paper now wants:

- the endpoints Schneider UB, TPL B, ECReact L1 and Mech L2 (replace Schneider B and USPTO50K B);
- all seven benchmark arms: SynRFP standard preset, SynRFP task-selected classification preset,
  DRFP, cRXNFP, RDKit difference, RDKit structural, RXNFP-MLM, each with the exact settings of the
  benchmark release;
- one run for all rows, so the table is consistent.

The paper agent also found why SynRFP encodes 99.95 % / 99.65 % of the sampled Schneider B /
USPTO50K B reactions while RDKit encodes 100 %: SynRFP parses components with sanitization
(`Chem.MolFromSmiles`), and records with an impossible phosphorus valence in a by-product fail
(`O=[P+](c1ccccc1)(c1ccccc1)c1ccccc1` in Wittig reactions, `O=[P](Cl)(Cl)(Cl)Cl` in PCl5
chlorinations). The RDKit fingerprints use `rdChemReactions.ReactionFromSmarts(s, useSmiles=True)`,
which does not sanitize, and encode these records as written. Please report this per fingerprint.

## Protocol

Same as `run_perturb`, unchanged except for endpoints and fingerprints:

1. Sample 2,000 reactions per endpoint with `np.random.default_rng(0)` (all rows if fewer).
2. Rewriting: shuffle component order and write each component as a random non-canonical SMILES
   (`randomise`). Reversal: swap reactants and products.
3. Encode original, rewritten and reversed forms with each fingerprint, using the benchmark
   preprocessing for that fingerprint (state it, e.g. canonicalization before DRFP).
4. Report per endpoint and fingerprint: valid original, valid rewritten, valid reversed, shared-valid
   fractions; exact agreement after rewriting and after reversal on shared-valid pairs; mean
   similarity (Tanimoto for bit vectors, cosine for continuous embeddings such as RXNFP-MLM, cRXNFP
   and RDKit difference). For continuous vectors, also report agreement within 1e-6 absolute
   tolerance, since exact float equality can fail for numerical reasons.
5. Also report DRFP agreement after rewriting without canonicalization, as in the current table, so
   the effect of the preprocessing stays visible.

## Failure audit

For every fingerprint, list the records that fail to encode in any form, with the exception class
and message, grouped into causes (e.g. invalid valence, unparsable token, tokenizer limit). Give
counts per endpoint and cause, one example SMILES per cause, and say whether the failure comes from
sanitization, the tokenizer or the model (e.g. RXNFP token limit).

## Output

On a new branch `exp/86-...`:
`perturbation_all_fps.csv` (one row per endpoint and fingerprint, columns as in
`Experiments/07_robustness/results/perturb.csv` plus the similarity and tolerance columns),
`encoding_failures.csv` (endpoint, fingerprint, form, cause, count, example),
`invariance_note.md` (one paragraph: which fingerprints are invariant to rewriting, which separate a
reaction from its reverse, and why some fingerprints encode fewer records).

## Constraints

Same as requests 74 to 85: pinned commits and hashes, package versions, no weights, no per-row
fingerprints, state anything that cannot be reproduced. Do not change the benchmark results.
