# USPTO-50K label order

The zero-based table index plus one equals the superclass number in Liu et al. Table 1. The class names in the original Schneider 2016 data archive agree with Table 1. The SynRXN balanced and unbalanced cohorts each have 50,016 rows and exactly match their pinned SynCat inputs by reaction SMILES, label and split. Balanced and unbalanced variants share every row identifier and label in order. [Liu et al. Table 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC5658761/#tbl1) · [Original Schneider 2016 data](https://doi.org/10.1021/acs.jcim.6b00564.s002)

| Independent reference | Rows in release | Unique agreeing class | Unique conflicting class | Multiple candidate classes | No chemical match |
|---|---:|---:|---:|---:|---:|
| Schneider 2016 Data Set B | 50,000 | 49,987 | 0 | 2 | 27 |
| Liu 2017 published model input | 50,037 | 48,921 | 0 | 2 | 1,093 |
| Union of references | — | 50,006 | 0 | 2 | 8 |

Canonical matching used unmapped isomeric SMILES, sorted within each reaction side. The original 2016 tables include conditions and minor products. For that source the audit compared its major product and each documented precursor projection using both author mappings. For Liu et al. it decoded the published source and target sequences and read the superclass from the published `RX` tag. The audit grouped references by canonical key before inspecting labels. For the 10 rows without a uniquely corroborating class, the archived SynCat lineage verifies the SynRXN copy and the independent chemical check remains unresolved. [Liu et al. model inputs at the pinned revision](https://github.com/pandegroup/reaction_prediction_seq2seq/tree/9a6f040198ce990cbdeea812e7e99df7becd142c/processed_data)

The [crosswalk](uspto50k_label_crosswalk.csv) gives the ten labels and per-class match counts. [AUDIT.json](AUDIT.json) records the source revisions, file hashes, counts and software versions.
