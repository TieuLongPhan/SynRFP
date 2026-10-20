# Five independent Path fits

Five-thread fits used nearly five CPUs without improving early Mech throughput.
This new operational version runs five independent one-CPU fits, each capped at
7GiB, on cores18,19,26,27,28. Aggregate research admission retains the stricter
85% RAM/CPU checks, within the user-authorized90% RAM ceiling. All200 archived
native main fits used at most5.469GiB peakRSS with the same2048-wide unscaled
float32 geometry. The7GiB cap has approximately28% headroom over that maximum.
An OOM is a failure, never a dropped row or permission to change a recipe.

The original scientific plan, sources, features, rows, all10 seeds, class counts,
algorithms, n_init and metrics stay sealed. Existing completed cells preserve
their recorded thread counts and artifacts. Root cooperatively drains the old
worker at a cell boundary and pauses only its governor before this continuation.
No old worker or model is killed, deleted or relabelled.

The120(dataset,seed) population pairs are indivisible: all_rows precedes
shared_valid_all_six on the same lane. Deterministic longest-processing-time
partitioning uses only the already-completed native main elapsed-time catalogue.
Its exact240-cell union,200 fresh fits and40 identical-population copies receive
a separately sealed operational plan and reviewed five-lane execution.

Each worker invokes the unchanged numerical producer through a scheduling-only
adapter, retains original STARTED/YIELDED/COMPLETE evidence, and adds an
OPERATIONAL receipt with the new source/partition/review hashes. A separate report
adapter verifies all five shards and their240-cell union before invoking the
unchanged original scoring/export mathematics. Six hours is a conditional target,
not a guaranteed completion time; unseen TPL costs, memory contention, audits or
resource waits can extend it. No GPU or baseline fitting is introduced.
