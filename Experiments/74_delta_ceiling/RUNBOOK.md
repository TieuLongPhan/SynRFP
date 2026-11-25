# Request 74 execution

[REQUEST.md](REQUEST.md) defines the scope. The released result is under
`publication_metadata/delta_ceiling_v1`.

## Inputs and encoding

Use the study 65 native classification plan in
`default_optimal_benchmark_v1/native/classification/plan.json` under the
workstation run archive. The plan pins the fourteen cleaned label tables and
their 25 grouped split-code arrays. It points to nine shared reaction sources.
Selection artifacts from the completed task-selected MLPs identify the actual
optimizer fitting rows. The ECReact source repeats some IDs across its original
partitions. The positional identity binds the reaction, encoded bits, and
label across those rows.

Use the workstation `synrfp` Python environment. The count encoder imports the
study 20 classification graph and tokenizer code, the study 43 corrected
Morgan module, and the study 51 recovery adapter for the three verified TPL
balanced dative inputs. It checks each profile against the matching frozen
validity mask and replays archived parity fingerprints on a systematic
sample and on the recovered rows. The actual classes use the complete
unsketched counts. Every encoded row must match a frozen row position.
The withheld refusal masks are identical for the four profiles. The matched
source records and any later-dropped label rows remain in their original
archives.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=
study=Experiments/74_delta_ceiling
base="$HOME/.local/share/synrfp-runs"
python "$study/encode_counts.py" --base "$base" \
  --native-repo ../SynRFP \
  --morgan-module ../SynRFP-morgan-ecfp-v004/synrfp/tokenizers/morgan.py \
  --output "$base/delta_ceiling_request74_v2" --jobs 24
python "$study/summarize.py" --base "$base" \
  --encoding "$base/delta_ceiling_request74_v2" \
  --ec-raw "$pinned_ec_raw" --context-mcc "$pinned_paper_context_csv" \
  --output "$study/publication_metadata/delta_ceiling_v1" --jobs 10
```

The context MCC input comes from
`data/benchmark_refresh/source_data/context_score_changes.csv` in paper commit
`ac83ccf`. Obtain it using `git show` from the separate paper clone. The original
ECReact release checksum and paper input checksum are in the audit. Both
commands read the previous benchmarks without modifying them. Count-key
intermediates stay in the run archive.

## Optional checks

The archived SynCat optimizer and held-out selectors are present. Study 55
already covers five fingerprint arms and 350 matched fits. The two SynRFP arms
would require 140 additional full-population MLP fits on those selectors.
Optional experiment B was deferred.

The alcohol stereo probe uses a SynKit environment with the configured
canonicalization API. It reads the two substrate pairs and their constitution-only
intended products from the paper's pinned condition-ranking output. Run in the
E1 environment with the same thread limits. The reported outcome is bit-vector
separation on these four substrates.

```sh
python "$study/stereo_probe.py" --paper-repo ../SynRFP-paper-attribution \
  --output "$base/delta_ceiling_request74_v2/stereo_feasibility.json"
```
