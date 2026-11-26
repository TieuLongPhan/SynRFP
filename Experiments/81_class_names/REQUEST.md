# Request 81: official class names for the Schneider and USPTO50K labels

Written 26 November 2026 by the paper agent. The manuscript is in
`~/Documents/TACsy/paper/synrfp_paper`; do not edit it. No fitting. Export the files named here and
report the branch and commit.

## Why

The SI figures print class names next to the reactions. Two sets are not yet verified against a
pinned source.

1. Schneider B and UB. The paper maps the integer labels 0 to 49 to NameRxn codes in the order of
   the SynProp copy of the dataset (`SynProp/Data/classification/schneider50k.csv`, columns `y` and
   `rxn_class`), so 10 is 1.7.6 and 25 is 2.6.3. The figures label these "methyl esterification" and
   "Fischer-Speier esterification". These two names were written from memory and must be checked.
2. USPTO50K. The paper maps labels 0 to 9 to superclasses 1 to 10 of Liu et al. (2017), Table 1. This
   is verified against the paper, but the label order in the SynRXN release should be confirmed.

## Tasks

1. Find a pinned source that gives the NameRxn reaction name for each of the 50 Schneider codes, for
   example the supporting information of Schneider et al. (2015, doi 10.1021/ci5006614), the rxnfp
   notebooks (`rxn4chemistry/rxnfp`, `nbs/`), or the SynCat or SynRXN curation scripts. Export the code,
   name and source for all 50 codes, and confirm or correct the names of 1.7.6 and 2.6.3.
2. Confirm that SynRXN `classification/schneider_b.csv.gz` and `schneider_u.csv.gz` integer labels
   follow the SynProp `y` to `rxn_class` order, by matching reaction SMILES on all rows.
3. Confirm that SynRXN `classification/uspto_50k_b.csv.gz` and `_u` labels 0 to 9 correspond to Liu et
   al. superclasses 1 to 10, by matching reactions to the original USPTO-50K release with its class
   column.
4. If a source cannot be pinned, say so and do not infer names from the reactions.

Outputs: `schneider_class_names.csv` (index, code, name, source), `uspto50k_label_check.md`,
`class_names_note.md`.

## Constraints

Same as requests 74 to 80. Push on a new branch named `exp/81-...` and report the commit.
