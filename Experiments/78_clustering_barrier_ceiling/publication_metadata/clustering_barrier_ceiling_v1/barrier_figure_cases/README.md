# Fig. S6 barrier case

Two para-substituted aryl chlorides undergo substitution with methoxide in
methanol at 323 K. The pair shares one held-out fold. The released activation
free energies are **34.09** and **31.48 kcal/mol**, derived from experimental
rate constants under the matching reported conditions. Both cases come from
the same experimental reference. [SNAr dataset paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9528810/) · [Original experimental reference](https://doi.org/10.1039/JR9630003492)

| SynRXN ID | Author's entry | Aryl chloride para substituent | ΔG‡ (kcal/mol) |
| --- | ---: | --- | ---: |
| snar_83 | 84 | Trifluoromethyl | 34.09 |
| snar_84 | 85 | Acetoxy | 31.48 |

Exact WL radius-2 signed counts agree and are nonempty. The archived
2,048-bit D fingerprints agree, and the signed halves of the D+U fingerprints
agree. Exact pooled counts and the pooled halves of the archived D+U
fingerprints distinguish the two reactions. The case illustrates a
representation that can distinguish the pair. The values shown are the
reference barriers. [Figure data with reaction SMILES](barrier_figure_cases.csv)

The selection required different reactant and product constitutions, one
substituted reactant component, a shared reactant component, and an
absolute target gap of at least one quarter of the full cohort sample SD.
A common grouped test fold ranked first, followed by total heavy atoms
and then the source positions. The frozen audited inputs replay the published
D and D+U parity vectors for every row of the three screened cohorts.

For both SN2 and E2/SN2 the only D class that adding U subdivides is the
empty-D class. It contains 636 rows and 316 distinct reactant constitutions.
SNAr has 45 subdivided nonempty D classes. It supplies the direct analogue
of the nonempty signed-count example in Fig. S6. [Screen counts](barrier_case_scan.csv)

The author CSV comes from the original supporting archive. The audit matches
all 503 SynRXN SNAr rows to a source reaction and exact activation free
energy, and the two selected entries have unique matches. [Author supporting archive](https://pmc-oa-opendata.s3.amazonaws.com/PMC9528810.1/SC-012-D0SC04896H-s002.zip) · [Input hashes and selection rule](AUDIT.json)

## Check on the existing yield panel

The archived bromide pair in recipe `delta03_pair02:241` measured 12.62%
and 83.15% yield. The standard D model predicted 31.21% for both. The pooled
model predicted 35.27% and 48.20%. Here, pooling distinguishes the two
assays and recovers 12.93 of the 70.53 percentage-point gap. The two chloride
recipes `delta01_pair03:075` and `delta01_pair03:092` both round to the
figure's 1% and 45% yields. Their pooled prediction gaps are 5.02 and 1.09
points, respectively. [Archived matched yield assays](https://github.com/TieuLongPhan/SynRFP/blob/9e2c2de9/Experiments/75_chemistry_cases/publication_metadata/chemistry_cases_v1/case_d.csv)
