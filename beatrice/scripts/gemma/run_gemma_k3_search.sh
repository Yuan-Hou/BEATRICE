#!/usr/bin/env bash
# Champion search per the round-12 protocol: champion = best k=3 combo,
# certified global by exhaustive enumeration. Two settings:
#   joint  (with-ctx label, 30-branch pool, C(30,3)=4060)  - sharded x2
#   q-only (closed-book label, 15-branch pool, C(15,3)=455) - single shard
# Plus q-only families+greedy for the report tables.
set -u
cd "${BEATRICE_ROOT:-.}"
PY=python
JCACHE=outputs/gemma4/cached_fusion_vecs/joint_30br
QCACHE=outputs/gemma4/cached_fusion_vecs/question_all15
REP=outputs/gemma4/reports
LOG=logs/gemma4/search
mkdir -p "$REP" "$LOG"
read -ra G <<< "${SEARCH_GPUS:-3 5}"
ES=test,popqa,nq,triviaqa
# placeholder champions for shard bookkeeping; real champion read off merged rows
JCH=attention,hid_prob,hidden   # greedy k3 result
QCH=hid_prob_attn,hidden,quad

# q-only families + greedy (cheap, on gpu G[1] first; k3 exhaustive follows)
if ! grep -q SEARCH_DONE "$LOG/qonly_families_greedy.log" 2>/dev/null; then
  CUDA_VISIBLE_DEVICES=${G[1]} "$PY" scripts/search_fusion_combos.py \
    --cache "$QCACHE" --side question --seeds 3 --eval-splits "$ES" \
    --out "$REP/qonly_families_greedy" > "$LOG/qonly_families_greedy.log" 2>&1 \
    || { echo "GEMMA_K3_FAILED qonly-families rc=$? $(date)"; exit 1; }
  echo "SEARCH_DONE" >> "$LOG/qonly_families_greedy.log"
fi
echo "[k3] qonly families+greedy done $(date)"

# joint k3 shards (both GPUs) + qonly k3 (after qonly greedy, on G[1] alongside shard1)
pids=()
for s in 0 1; do
  CUDA_VISIBLE_DEVICES=${G[$s]} "$PY" scripts/exhaustive_sharded.py \
    --cache "$JCACHE" --side context --k 3 --seeds 3 \
    --shard "$s" --nshards 2 --eval-splits "$ES" \
    --champion "$JCH" --out "$REP/exhaustive_k3_joint" \
    > "$LOG/exhaustive_k3_joint_shard${s}.log" 2>&1 &
  pids+=($!)
done
CUDA_VISIBLE_DEVICES=${G[0]} "$PY" scripts/exhaustive_sharded.py \
  --cache "$QCACHE" --side question --k 3 --seeds 3 \
  --shard 0 --nshards 1 --eval-splits "$ES" \
  --champion "$QCH" --out "$REP/exhaustive_k3_qonly" \
  > "$LOG/exhaustive_k3_qonly.log" 2>&1 &
pids+=($!)
f=0; for p in "${pids[@]}"; do wait "$p" || f=1; done
[ $f -ne 0 ] && { echo "GEMMA_K3_FAILED shards $(date)"; exit 1; }

"$PY" scripts/exhaustive_sharded.py --cache "$JCACHE" --side context --k 3 --seeds 3 \
  --nshards 2 --eval-splits "$ES" --champion "$JCH" --out "$REP/exhaustive_k3_joint" --merge \
  > "$LOG/exhaustive_k3_joint_merge.log" 2>&1 || { echo "GEMMA_K3_FAILED joint merge $(date)"; exit 1; }
echo "[k3] joint merge done $(date)"
"$PY" scripts/exhaustive_sharded.py --cache "$QCACHE" --side question --k 3 --seeds 3 \
  --nshards 1 --eval-splits "$ES" --champion "$QCH" --out "$REP/exhaustive_k3_qonly" --merge \
  > "$LOG/exhaustive_k3_qonly_merge.log" 2>&1 || { echo "GEMMA_K3_FAILED qonly merge $(date)"; exit 1; }
echo "GEMMA_K3_DONE $(date)"
