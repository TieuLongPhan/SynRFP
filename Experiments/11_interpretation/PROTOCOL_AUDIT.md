# Seed and split protocol audit

- SynRFP hash seed: **1** (fixed across runs).
- Track A: **25** grouped scaffold splits; seed `repeat*5 + fold`, with repeat-specific shuffled folds.
- Track B: **3** model seeds on one split; datasets without a published split use fallback seed **42**.
- Factorial: **162** configurations × 10 endpoints × 5 seeds = **8100 cells/model**.
- Saved records with undeclared seed metadata: **0**.

This audit separates hash/preprocessing determinism from split and
model-initialization replicates; it does not assess statistical independence.
