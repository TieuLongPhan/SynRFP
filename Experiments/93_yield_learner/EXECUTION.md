# Request 93 workstation execution

Request 93 compares a 500-tree random forest with the existing yield MLP.
It also compares a balanced random forest and a binary MLP on the released
Buchwald–Hartwig success labels. The runs start after verified publication of
Request 91 and its service exit.

The preflight binds 69 splits and the saved training, validation and test row
identities for each of the seven released representations. It uses their
saved model inputs directly, including train-only standardization. The two
additional WL radius-three context variants reuse their archived vectors
where available. USPTO needs the D and D+U vectors generated with the released
literal graph profile. All 143,535 existing D+S vectors must replay before
these context matrices are admitted. No benchmark representation is
re-encoded for fitting.

Part C trains on the author `labels` column. The preflight observes 183
differences compared with reapplying the >10 rule to the numeric yield
column. Success metrics use the released labels and the condition-ranking
evaluation uses measured yields, as in Request 92. A 0.5 probability cutoff
is fixed for binary balanced accuracy. The archived Request 92 recipes,
ranking and eligibility rules are pinned.

The RF context contrast uses every matched split. The archived MLP contrast
is included only when all three selectors match. The full-row Buchwald–
Hartwig and USPTO reference contexts require a separate matched MLP fit
study to fill those legacy comparator cells. The seven-representation MLP
rank comparison already uses the exact same released splits on every panel.
XGBoost is outside project scope.

From this isolated worktree, `plan93.py` may run during BitBIRCH as a
read-only preflight against the saved supports. `queue93.sh` waits for the
BitBIRCH publication receipt before leasing CPU for the new run. It uses
the `synrfp` environment for graph encoding and the archived `synprop`
CPU environment for learners. The production queue limits concurrency
to 20 single-threaded jobs, including at most four simultaneous USPTO fits.
Four scalar tables, two classifier tables and the provenance and narrative
are pushed as a new commit after closure. Model checkpoints and raw
predictions remain in the workstation artifact directory.
