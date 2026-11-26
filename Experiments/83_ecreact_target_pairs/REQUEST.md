# Request 83: target pairs in ECReact

Written 26 November 2026 by the paper agent. The manuscript is in
`~/Documents/TACsy/paper/synrfp_paper`; do not edit it. Export the files named here and report the
branch and commit.

## Why

EnzymeMap gives the paper its classification target pair: EC class against bond-edit class on the
same reactions, where pooled context helps EC more, as predicted. ECReact is a much larger enzymatic
set (185,720 reactions) and could test the same idea. It carries EC labels at three levels but no atom
mapping, so it has no bond-edit label. ECReact and EnzymeMap share BRENDA provenance, so overlap must
be reported.

## Experiment A (required): EC level 1 against EC level 3

Same reactions, two labels. Level 3 depends more on the substrate than level 1. Using the published
MLP classification protocol (grouped partitions, seeds) and 2,048-bit D and D+U (1,024 + 1,024),
report the D to D+U change in MCC and in error rate for EC level 1, 2 and 3, the paired difference of
the gains (level 3 minus level 1, level 2 minus level 1) with paired group-bootstrap 95% intervals,
and whether the training-only prediction rule of the EnzymeMap test (same thresholds, not relaxed)
predicts the ordering. Reuse published fits where they exist.

## Experiment B (required): EC class against bond-edit class

1. Atom-map ECReact with a pinned mapper (RXNMapper or the mapper used for EnzymeMap), and derive
   bond-edit labels with the same definition as the EnzymeMap test. Report the mapping success rate
   and, on reactions shared with EnzymeMap (match on canonical unmapped SMILES), the agreement of the
   derived bond-edit labels with EnzymeMap's.
2. Run the EnzymeMap target-pair test on ECReact (EC level 2 against bond-edit class, same protocol,
   same rule, same intervals). Report it twice: on all mapped reactions and after removing reactions
   that also occur in EnzymeMap.

## Output

`ec_level_pairs.csv`, `mapping_quality.csv`, `ecreact_target_pair.csv`, `overlap_with_enzymemap.csv`,
`ecreact_target_pairs_note.md` (one paragraph: does context help the substrate-dependent target more,
and does the answer hold without the EnzymeMap overlap).

## Constraints

Same as requests 74 to 82: pinned commits and hashes, package versions, no weights, no per-row
predictions, state anything that cannot be reproduced. Push on a new branch named `exp/83-...` and
report the commit.
