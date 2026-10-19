# NCS evidence analysis

All seven task factorials are available.

Canonical input cells: 20,448. Ready tasks: classification, clustering, yield, barrier, feasibility, reagent, condition. No configuration is selected.

The four prespecified contrasts are delta_union minus delta, roles minus delta, roles minus delta_union, and union minus delta. Each pair matches tokenizer, radius, sketch, source endpoint, evaluation regime and native partition. Optimizer seeds sharing a condition fold or fixed native/expert test population are averaged before the contrast. Clustering initialization seeds remain explicitly labelled initialization comparisons, never CV folds.

Endpoint effects average partition contrasts; families average related endpoints; task summaries balance families. Grouped and native reagent/condition views remain separate. Barrier benefit negates the raw RMSE difference; other benefits equal raw score differences. Metrics are never averaged across tasks. Standard deviations are descriptive correlated-partition variation, with no p-values or confidence claims.

## NCS priorities after the current queue

1. Explain the balance between changed and retained context using the matched mode effects and the separate bounded token audit. At full token-count level U=|D|+2S, so delta_union and roles encode equivalent count information before support/sketch compression; a gain between their finite fingerprints needs a compression or budget explanation.
2. Preserve clustering as a central application, with known-resolution and label-free resolution experiments kept distinct. Legacy chemical coherence is useful evidence; representative selection has mixed outcomes and must not be sold as universal superiority. Any further utility study needs a new fixed protocol and the missing crxnfp comparator.
3. Jointly choose one practical preset after this diagnostic table; no automatic selection or test-max promotion occurs here. Treat historical-data selection as exploratory, and separate later confirmation from this screen.
4. Finish specialist references already queued, report barrier weaknesses and compute tradeoffs, then decide which additional references justify effort. This report does not train Parrot, T5Chem, adapted SynCat feasibility or a full SynCluster comparison.
5. Use native recorded fit times and cache sizes as descriptive cost evidence now. Run an isolated fixed-hardware encoding benchmark only after the busy research queues drain.

## Existing evidence and boundaries

Inventory: 785 files with SHA-256 values, read without modifying original reports, feature caches or model weights.

Legacy SynRFP utility/locality uses WL radius2 delta at1024bits; it does not establish new 2048-bit modes or all sketches. The 168-cell representative-selection study uses two 20,000-row samples, three seeds, four budgets, five fingerprints, random and greedy controls. It lacks crxnfp and includes rapidly saturating coarse-label coverage. Product-group diversity and mapped element/bond-change signatures are declared proxies, not exact reaction identity or experimental utility.

Task and source-family counts are aggregation conventions, not claims that all datasets are independent. EHReact is excluded. Existing negative results remain visible. NCS suitability is a research target, not an acceptance prediction.
