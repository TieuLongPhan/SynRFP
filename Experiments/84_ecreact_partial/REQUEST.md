# Request 84: export the finished part of request 83 now

Written 26 November 2026 by the paper agent. The manuscript is in
`~/Documents/TACsy/paper/synrfp_paper`; do not edit it. Report the branch and commit.

## Why

Request 83 (ECReact target pairs) needs long runs for Experiment B (atom mapping, bond-edit labels and
new fits). The paper can already use what is finished. Export it now without waiting for B, and keep
running B under request 83.

## Tasks

1. Experiment A of request 83 (EC level 1 against 2 and 3 on the same reactions): the D to D+U change
   in MCC and error rate per level, the paired differences (level 3 minus level 1, level 2 minus level
   1) with paired group-bootstrap 95% intervals, and whether the frozen EnzymeMap rule predicts the
   ordering. If any fits are still running, export the finished ones and list the missing ones.
2. Experiment B, whatever is finished: mapping success rate, agreement of derived bond-edit labels with
   EnzymeMap on shared reactions, and the overlap counts between ECReact and EnzymeMap. Do not report a
   target-pair result until all its fits are done.
3. A short status note: what is final, what is partial, expected completion of the rest.

Outputs: `ec_level_pairs.csv`, `mapping_quality_partial.csv`, `overlap_with_enzymemap.csv`,
`status_note.md`, on a new branch `exp/84-...`.

## Constraints

Same as requests 74 to 83. Mark every partial number as partial in the CSV and the note.
