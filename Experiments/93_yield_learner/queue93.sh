#!/usr/bin/env bash
set -euo pipefail

root="$HOME/.local/share/synrfp-runs/request93_yield_learner_v1"
bitbirch="$HOME/.local/share/synrfp-runs/request91_bitbirch_factorial_v1"
gitdir="$(pwd)"
e=Experiments/93_yield_learner
out="$e/publication_metadata/yield_learner_v1"
py="$HOME/anaconda3/envs/synrfp/bin/python"
fitpy="$HOME/anaconda3/envs/synprop/bin/python"
remote=exp/93-yield-learner
unit=synrfp-request91-full
bitdir="$gitdir/../SynRFP-bitbirch-factorial"
receipt="Experiments/91_bitbirch_factorial/publication_metadata/bitbirch_factorial_v1/COMPLETE.json"

mkdir -p "$root"
[[ -s "$root/PLAN.json" ]]
[[ "$(git branch --show-current)" == "$remote" ]]
export CUDA_VISIBLE_DEVICES=''
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1

publish_remote() {
    local branch="$1"
    local i
    for i in 1 2 3 4; do
        if git push -u origin "$branch"; then return 0; fi
        sleep "$((15*i))"
    done
    return 1
}

while [[ ! -s "$bitbirch/PUBLISHED_COMMIT" ]] || systemctl --user is-active --quiet "$unit"; do
    printf 'waiting_for_bitbirch_publication\n' > "$root/QUEUE_STAGE"
    sleep 30
done
[[ "$(git rev-parse HEAD)" == "$(cat "$root/SOURCE_COMMIT")" ]]
git diff --quiet HEAD -- "$e"
sha256sum --status -c "$root/PLAN.sha256"
source_commit="$(cat "$bitbirch/PUBLISHED_COMMIT")"
[[ "$source_commit" =~ ^[0-9a-f]{40}$ ]]
git -C "$bitdir" show "$source_commit:$receipt" | "$py" -c 'import sys,json; d=json.load(sys.stdin); assert d["status"]=="complete" and d["prepare"]["expected_fits"]==49400'
[[ "$(git -C "$bitdir" rev-parse HEAD)" == "$source_commit" ]]
printf '%s\n' "$source_commit" > "$root/BITBIRCH_FINISH_VERIFIED"

printf 'preparing_uspto_context\n' > "$root/QUEUE_STAGE"
"$py" "$e/context93.py" --jobs 20
printf 'training\n' > "$root/QUEUE_STAGE"
"$fitpy" "$e/fit93.py" --jobs 20
printf 'reporting\n' > "$root/QUEUE_STAGE"
"$fitpy" "$e/report93.py"
git add "$out/cells.csv" "$out/summary.csv" "$out/ranking.csv" "$out/context.csv" \
        "$out/classifier_ranking.csv" "$out/classifier_metrics.csv" "$out/note.md" "$out/COMPLETE.json"
git diff --cached --check
git commit -m 'Publish Request 93 yield learner and success-classifier comparisons'
printf 'publishing\n' > "$root/QUEUE_STAGE"
publish_remote "$remote"
published="$(git ls-remote origin "refs/heads/$remote" | cut -f1)"
[[ "$published" == "$(git rev-parse HEAD)" ]]
printf '%s\n' "$published" > "$root/PUBLISHED_COMMIT"
printf 'complete\n' > "$root/QUEUE_STAGE"
