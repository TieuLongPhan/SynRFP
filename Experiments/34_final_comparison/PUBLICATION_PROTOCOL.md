# Publish the final main fingerprint comparison

The user explicitly authorized committing and pushing the completed seven-task
SynRFP comparison on 19 October 2026, without waiting for the separate specialist
references. The destination is the current `exp/reproducible-benchmark` branch on
`origin`. This publisher does not launch training or use a GPU.

Publication requires the canonical experiment34 final receipt, its identical
local/export copies, all 8,238 scientific cells, all seven completed task grids,
the selected 2,048-bit construction, native regression and clustering completion,
verified component result and output hashes, and the original frozen report/source
seals. It rechecks the keyed six-fingerprint panels rather than using file counts
or inactive processes as completion evidence. The existing exploratory-selection
qualification remains explicit. New experiment35 references never gate this push.

A fixed allowlist contains seven raw-cell tables, seven summaries, the final
report, completion and plan metadata, the frozen preset/selection metadata and
the small experiment34 scientific/report/publisher sources. Model weights, raw
inputs, feature caches, assignments, unrelated changes and experiment35 are
excluded. This is a result/metadata publication, not a standalone training archive;
the committed scope metadata states that boundary.

The publisher builds a separate Git index from HEAD and adds only the allowlist.
Every staged blob must match its SHA256 snapshot. It creates a commit with that
exact tree and uses an expected-old-HEAD compare-and-swap update while holding the
real Git index lock. Only publication-owned index entries are copied into the
latest real index; unrelated user stages and unstaged work are preserved. A
concurrent HEAD change aborts publication. The normal push never uses force,
rebase, reset, stash or deletion. A divergent remote is retained for inspection.

The persistent CPU-only watcher retries network failures after retaining the
single local commit. Its local `publication/PUBLISHED.json` records the verified
remote commit, exact seven task counts, scientific gate and published file hashes.
Weights stay local. A short completion line is appended to the local SS log.
Unexpected scientific/source changes fail closed with a retained diagnostic.

Prepare and review before enabling the service:

```bash
python -m unittest discover -s Experiments/34_final_comparison -p test_publish_main_results.py
python Experiments/34_final_comparison/publish_main_results.py --prepare
```

The reviewed service is capped at 0.1 CPU and 256 MiB RAM. Waiting never delays
or blocks training. Final source/hash verification reads metadata only.
