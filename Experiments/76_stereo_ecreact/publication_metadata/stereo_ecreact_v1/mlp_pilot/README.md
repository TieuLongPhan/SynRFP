# Five-partition ECReact MLP pilot

The pilot uses the stereo E/Z profile selected from the count-ceiling audit.
ECReact L2 and L3 each use study 65 partition seeds 1–5. These are five saved
partitions spanning parts of two repeats, so the sample SD is descriptive.

Two new SynRFP arms use stereo-aware D and D+U. Five archived fingerprint
baselines share those exact held-out rows. The new arms reuse the study 65
optimizer fitting and validation indices, initial weights, RNG state, batch
size, Adam update and accuracy-based stopping rule. The native training loop
runs on CPU with the current Torch build. The archived comparator scores came
from the original execution environment.

See [cells.csv](cells.csv) for per-partition MCC and [summary.csv](summary.csv)
for mean and sample SD. [original_arm_comparison.csv](original_arm_comparison.csv)
pairs each stereo arm with its corresponding ordinary SynRFP arm in the same
held-out partition. These differences describe the pilot on previously used
ECReact splits. They include any numerical effect from the execution backend.

The [audit](AUDIT.json) pins the source state, selectors, encoder and native
worker. Retained weights reproduced each new arm's saved test predictions.
