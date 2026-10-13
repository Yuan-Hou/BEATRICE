#!/usr/bin/env bash
# gemma-4-E4B hidden-layer sweep: extract per-layer answer hiddens for both
# settings across free GPUs, then train per-layer probes (saplma mean/last +
# llmsknow) and emit AUROC curves. Resumable: shards with .done markers skip.
set -u
cd "${BEATRICE_ROOT:-.}"
PY=python
LOG=logs/gemma4
mkdir -p "$LOG"
MIN_FREE_MB=25000
MAX_GPUS=3  # user cap 2026-07-17: use at most 3 GPUs
TRAIN_SHARDS=8

# task list: "setting group shard" — heaviest first for greedy balancing
TASKS=()
for setting in withctx noctx; do
  for ((i=0; i<TRAIN_SHARDS; i++)); do TASKS+=("$setting train $i/$TRAIN_SHARDS"); done
done
for setting in withctx noctx; do
  for grp in test_id validation popqa_ood; do TASKS+=("$setting $grp 0/1"); done
done

mapfile -t FREE < <(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits \
  | awk -F', ' -v m=$MIN_FREE_MB '($2-$3)>=m {print $1}')
FREE=("${FREE[@]:0:$MAX_GPUS}")
if [ "${#FREE[@]}" -eq 0 ]; then echo "GEMMA_LAYER_SWEEP_FAILED no free gpu $(date)"; exit 1; fi
echo "[sweep] gpus: ${FREE[*]}  tasks: ${#TASKS[@]}"

# round-robin task queues per gpu
declare -A QUEUE
for i in "${!TASKS[@]}"; do
  g=${FREE[$((i % ${#FREE[@]}))]}
  QUEUE[$g]+="${TASKS[$i]};"
done

pids=()
for g in "${FREE[@]}"; do
  [ -z "${QUEUE[$g]:-}" ] && continue
  (
    IFS=';' read -ra items <<< "${QUEUE[$g]}"
    for it in "${items[@]}"; do
      [ -z "$it" ] && continue
      read -r setting grp shard <<< "$it"
      tag="${setting}_${grp}_$(echo "$shard" | tr / -)"
      echo "[sweep] gpu$g -> $tag ($(date))"
      CUDA_VISIBLE_DEVICES=$g "$PY" scripts/gemma/extract_gemma_layer_features.py \
        --setting "$setting" --group "$grp" --shard "$shard" \
        >> "$LOG/lsweep_${tag}.log" 2>&1
      echo "[sweep] gpu$g <- $tag rc=$? ($(date))"
    done
  ) &
  pids+=($!)
done
wait "${pids[@]}"

miss=0
for it in "${TASKS[@]}"; do
  read -r setting grp shard <<< "$it"
  stem="${grp}__shard$(echo "$shard" | tr / of)"
  # shard i/n -> shard{i}of{n}
  stem="${grp}__shard${shard%/*}of${shard#*/}"
  if [ ! -f "data/gemma4/layer_features/$setting/$stem.done" ]; then
    echo "[sweep] MISSING $setting/$stem"
    miss=1
  fi
done
[ "$miss" -ne 0 ] && { echo "GEMMA_LAYER_SWEEP_FAILED extraction $(date)"; exit 1; }

echo "[sweep] curves ($(date))"
CUDA_VISIBLE_DEVICES=${FREE[0]} "$PY" scripts/layer_sweep_curves.py \
  --base data/gemma4/layer_features --out outputs/reports/gemma_layer_sweep_curves \
  >> "$LOG/lsweep_curves.log" 2>&1
if grep -q "LAYER_SWEEP_DONE" "$LOG/lsweep_curves.log"; then
  echo "GEMMA_LAYER_SWEEP_DONE $(date)"
else
  echo "GEMMA_LAYER_SWEEP_FAILED curves $(date)"
fi
