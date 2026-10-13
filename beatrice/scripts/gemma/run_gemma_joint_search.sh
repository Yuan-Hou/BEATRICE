#!/usr/bin/env bash
# Joint 30-branch combo search on the gemma cache, mirroring qwen round-12:
#   1. families + greedy (search_fusion_combos)
#   2. exhaustive k=2 (C(30,2)=435), sharded x2
#   3. exhaustive k=3 (C(30,3)=4060), sharded x2
# All on cached vecs (GPU-light). 2 GPUs via SEARCH_GPUS="a b" (default 3 5).
set -u
cd "${BEATRICE_ROOT:-.}"
PY=python
CACHE=outputs/gemma4/cached_fusion_vecs/joint_30br
REP=outputs/gemma4/reports
LOG=logs/gemma4/search
mkdir -p "$REP" "$LOG"
read -ra G <<< "${SEARCH_GPUS:-3 5}"
ES=test,popqa,nq,triviaqa
# placeholder champion for shard bookkeeping; real champion read off merged rows
CH2=hid_prob,q_quad
CH3=hid_prob,q_quad,q_hid_prob_attn

if ! grep -q SEARCH_DONE "$LOG/joint_families_greedy.log" 2>/dev/null; then
  CUDA_VISIBLE_DEVICES=${G[0]} "$PY" scripts/search_fusion_combos.py \
    --cache "$CACHE" --side context --seeds 3 --eval-splits "$ES" \
    --out "$REP/joint_families_greedy" > "$LOG/joint_families_greedy.log" 2>&1 \
    || { echo "GEMMA_SEARCH_FAILED families rc=$? $(date)"; exit 1; }
  echo "SEARCH_DONE" >> "$LOG/joint_families_greedy.log"
fi
echo "[search] families+greedy done $(date)"

for k in 2 3; do
  ch=$CH2; [ "$k" -eq 3 ] && ch=$CH3
  pids=()
  for s in 0 1; do
    CUDA_VISIBLE_DEVICES=${G[$s]} "$PY" scripts/exhaustive_sharded.py \
      --cache "$CACHE" --side context --k "$k" --seeds 3 \
      --shard "$s" --nshards 2 --eval-splits "$ES" \
      --champion "$ch" --out "$REP/exhaustive_k${k}_joint" \
      > "$LOG/exhaustive_k${k}_shard${s}.log" 2>&1 &
    pids+=($!)
  done
  f=0; for p in "${pids[@]}"; do wait "$p" || f=1; done
  [ $f -ne 0 ] && { echo "GEMMA_SEARCH_FAILED k=$k shards $(date)"; exit 1; }
  "$PY" scripts/exhaustive_sharded.py \
    --cache "$CACHE" --side context --k "$k" --seeds 3 --nshards 2 \
    --eval-splits "$ES" --champion "$ch" --out "$REP/exhaustive_k${k}_joint" --merge \
    > "$LOG/exhaustive_k${k}_merge.log" 2>&1 \
    || { echo "GEMMA_SEARCH_FAILED k=$k merge $(date)"; exit 1; }
  echo "[search] exhaustive k=$k done $(date)"
done
echo "GEMMA_SEARCH_DONE $(date)"
