#!/usr/bin/env bash
# Precompute frozen-branch fusion vectors for the gemma 30-encoder pool:
# context_all15 + question_all15 caches over train/validation + 4 eval splits.
# One GPU per side (2 total; qwen budget-check may hold a 3rd -> cap ok).
set -u
cd "${BEATRICE_ROOT:-.}"
PY=python
export PROBE_PACKED_ROOT=./data/gemma4/packed_features
export PROBE_FAST_COLLATE=1
export PROBE_CUDA_PREFETCH=1
S=configs/experiment/probe/gemma4_search
LOG=logs/gemma4/search
mkdir -p "$LOG" outputs/gemma4/cached_fusion_vecs
MIN_FREE_MB=30000
MAX_GPUS=2

if [ -n "${PC_GPUS:-}" ]; then
  read -ra FREE <<< "$PC_GPUS"
else
  mapfile -t FREE < <(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits \
    | awk -F', ' -v m=$MIN_FREE_MB '($2-$3)>=m {print $1}')
  FREE=("${FREE[@]:0:$MAX_GPUS}")
fi
[ "${#FREE[@]}" -lt 2 ] && { echo "GEMMA_PRECOMPUTE_FAILED need 2 gpus, have ${#FREE[@]} $(date)"; exit 1; }
echo "[pc] gpus: ${FREE[*]}"

run_side() {
  local g=$1 tag=$2 side=$3
  local out=outputs/gemma4/cached_fusion_vecs/${side}_all15
  local log=$LOG/precompute_${tag}_all15.log
  if [ -f "$out/validation.pt" ] && grep -q PRECOMPUTE_DONE "$log" 2>/dev/null; then
    echo "[pc] skip $side (done)"; return 0
  fi
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    CUDA_VISIBLE_DEVICES=$g "$PY" scripts/precompute_frozen_fusion_vecs.py \
      --config "$S/pc_${tag}_all15.yaml" \
      --out "$out" --side "$side" --splits train,validation \
      --eval-config test="$S/eval_${tag}_test.yaml" \
      --eval-config popqa="$S/eval_${tag}_popqa.yaml" \
      --eval-config nq="$S/eval_${tag}_nq.yaml" \
      --eval-config triviaqa="$S/eval_${tag}_triviaqa.yaml" \
      > "$log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ] || ! grep -q PRECOMPUTE_DONE "$log"; then
    echo "[pc] FAIL $side rc=$rc (see $log)"; return 1
  fi
  echo "[pc] done $side ($(date))"
}

run_side "${FREE[0]}" ctx context &
p1=$!
run_side "${FREE[1]}" q question &
p2=$!
f=0; wait $p1 || f=1; wait $p2 || f=1

if [ $f -eq 0 ]; then
  "$PY" scripts/build_joint_cache.py \
    --ctx outputs/gemma4/cached_fusion_vecs/context_all15 \
    --q outputs/gemma4/cached_fusion_vecs/question_all15 \
    --out outputs/gemma4/cached_fusion_vecs/joint_30br \
    --splits train,validation,test,popqa,nq,triviaqa > "$LOG/build_joint.log" 2>&1 \
    && echo "GEMMA_PRECOMPUTE_DONE joint_30br $(date)" \
    || { echo "GEMMA_PRECOMPUTE_FAILED joint (see $LOG/build_joint.log) $(date)"; exit 1; }
else
  echo "GEMMA_PRECOMPUTE_FAILED sides $(date)"; exit 1
fi
