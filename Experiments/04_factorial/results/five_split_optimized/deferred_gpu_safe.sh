#!/usr/bin/env bash
set -u
cd /home/labhhc4/Documents/Workspace/Long/SynRFP
# Wait for the TPL benchmark tail before reclaiming factorial ownership. The
# CPU workers are intentionally stopped only after the benchmark GPUs are free;
# completed JSON cells remain authoritative and unfinished cells are replayed.
while true; do
  active=0
  for pid in 4161432 4161433; do
    if ps -p "$pid" -o args= 2>/dev/null | rg -q 'classify.py.*--(study benchmark|study factorial).*--models xgb'; then
      active=1
    fi
  done
  [ "$active" -eq 0 ] && break
  wait_ticks=$(( ${wait_ticks:-0} + 1 ))
  if [ $((wait_ticks % 120)) -eq 0 ]; then
    echo "$(date -Is) waiting for TPL benchmark workers; CPU factorial ownership remains active"
  fi
  sleep 30
done
echo "$(date -Is) TPL benchmark workers finished; reclaiming factorial XGBoost ownership"
for pid in 1057049 1057050 1057051 1057052; do
  if ps -p "$pid" -o args= 2>/dev/null | rg -q 'classify.py.*--study factorial.*--models xgb'; then
    kill -TERM "$pid" 2>/dev/null || true
  fi
done
sleep 5
# A terminated worker can leave a claim lock for its in-progress cell. Only
# lock files without a completed JSON result are removed; finished results are
# never overwritten by the replay.
find Experiments/03_classification/results/factorial -name '*.json.lock' -type f -delete
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
mkdir -p Experiments/04_factorial/results/five_split_optimized
(
  CUDA_VISIBLE_DEVICES=0 /home/labhhc4/anaconda3/envs/synrfp/bin/python -u Experiments/03_classification/classify.py \
    --config Experiments/03_classification/results/cpu_parallel_finish/xgb_gpu_tpl0_config.yaml \
    --study factorial --models xgb --seeds 0 1 2 3 4 \
    --datasets schneider_ub uspto50k_ub syntemp_r0 ecreact_1 mech_1 tpl_b \
    > Experiments/04_factorial/results/five_split_optimized/xgb_gpu0_all.log 2>&1
) & p0=$!
(
  CUDA_VISIBLE_DEVICES=1 /home/labhhc4/anaconda3/envs/synrfp/bin/python -u Experiments/03_classification/classify.py \
    --config Experiments/03_classification/results/cpu_parallel_finish/xgb_gpu_tpl1_config.yaml \
    --study factorial --models xgb --seeds 0 1 2 3 4 \
    --datasets schneider_b uspto50k_b ecreact_3 mech_2 \
    > Experiments/04_factorial/results/five_split_optimized/xgb_gpu1_all.log 2>&1
) & p1=$!
printf '%s %s\n' "$p0" "$p1" > Experiments/04_factorial/results/five_split_optimized/xgb_stage_pids.txt
wait "$p0" "$p1"
/home/labhhc4/anaconda3/envs/synrfp/bin/python -u Experiments/03_classification/mlp_gpu_queue.py \
  --config Experiments/config.yaml --gpu-python /home/labhhc4/anaconda3/envs/rxnfp/bin/python \
  --devices 0 1 --studies factorial \
  --control Experiments/04_factorial/results/five_split_optimized/mlp_gpu \
  > Experiments/04_factorial/results/five_split_optimized/mlp_queue.log 2>&1

# Finalization is deliberately after both GPU stages.  ANOVA withholds output
# on incomplete matrices, while the interpretation audit can safely refresh
# provisional summaries and will preserve that explicit status in its JSON.
{
  echo "$(date -Is) GPU stages finished; running final matrix checks"
  anova_status=0
  /home/labhhc4/anaconda3/envs/synrfp/bin/python Experiments/03_classification/../04_factorial/anova.py \
    --config Experiments/config.yaml || anova_status=$?
  echo "$(date -Is) ANOVA exit status: ${anova_status}"
  audit_status=0
  Experiments/11_interpretation/run_audit.sh || audit_status=$?
  echo "$(date -Is) interpretation audit exit status: ${audit_status}"
  if [ "$anova_status" -ne 0 ] || [ "$audit_status" -ne 0 ]; then
    echo "$(date -Is) FINALIZATION FAILED"
    exit 1
  fi
  echo "$(date -Is) FINALIZATION PASSED"
} > Experiments/04_factorial/results/five_split_optimized/finalization.log 2>&1
