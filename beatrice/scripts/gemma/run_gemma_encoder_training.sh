#!/usr/bin/env bash
# Train the full gemma 30-encoder set (15 ctx + 15 qonly) on free GPUs.
# Resumable: units with .done markers skip. Success gate = rc==0 AND checkpoint
# AND val-preds files exist AND the log carries the trainer's best_auc marker
# (completion-check lesson: log marker, not just artifacts).
#
# Usage: bash scripts/gemma/run_gemma_encoder_training.sh
set -u
cd "${BEATRICE_ROOT:-.}"
PY=python
export PROBE_PACKED_ROOT=./data/gemma4/packed_features
export PROBE_FAST_COLLATE=1   # batch-direct packed reads (0.65s/batch qwen-era path needs BOTH flags)
export PROBE_CUDA_PREFETCH=1  # pinned dual-slot buffers + side-stream H2D overlap
LOG=logs/gemma4/encoders
MARK=outputs/gemma4/.enc_markers
mkdir -p "$LOG" "$MARK"
MIN_FREE_MB=30000
MAX_GPUS=3  # hard cap on concurrent GPUs

"$PY" scripts/gemma/make_gemma_encoder_configs.py || { echo "GEMMA_ENCODERS_FAILED configgen $(date)"; exit 1; }
MANIFEST=configs/experiment/probe/gemma4_encoders/manifest.json
mapfile -t UNITS < <("$PY" -c "
import json
for u in json.load(open('$MANIFEST'))['units']:
    print('|'.join([u['unit'], u['config'], u['checkpoint'], u['val_preds']]))
")
echo "[enc] units=${#UNITS[@]} packed_root=$PROBE_PACKED_ROOT"

mapfile -t FREE < <(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits \
  | awk -F', ' -v m=$MIN_FREE_MB '($2-$3)>=m {print $1}')
FREE=("${FREE[@]:0:$MAX_GPUS}")
[ "${#FREE[@]}" -eq 0 ] && { echo "GEMMA_ENCODERS_FAILED no free gpu $(date)"; exit 1; }
echo "[enc] gpus: ${FREE[*]}"

run_unit() {
  local g=$1 unit=$2 cfg=$3 ckpt=$4 preds=$5
  local log="$LOG/${unit}.log"
  [ -f "$MARK/${unit}.done" ] && { echo "[enc] skip $unit (.done)"; return 0; }
  echo "[enc] gpu$g train $unit ($(date))"
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    CUDA_VISIBLE_DEVICES=$g "$PY" scripts/train_probe.py --config "$cfg" >> "$log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ] || [ ! -f "$ckpt" ] || [ ! -f "$preds" ] || ! grep -q "best_auc" "$log"; then
    echo "[enc] FAIL $unit rc=$rc (see $log)"; return 1
  fi
  touch "$MARK/${unit}.done"
  echo "[enc] done $unit ($(date)) $(grep -o 'best_auc=[0-9.]*' "$log" | tail -1)"
}

# filter .done before lane assignment so resumed runs spread stragglers
PENDING=()
for it in "${UNITS[@]}"; do
  IFS='|' read -r u _rest <<< "$it"
  if [ -f "$MARK/${u}.done" ]; then echo "[enc] skip $u (.done)"; else PENDING+=("$it"); fi
done
declare -A QUEUE
for i in "${!PENDING[@]}"; do
  g=${FREE[$((i % ${#FREE[@]}))]}
  QUEUE[$g]+="${PENDING[$i]};;"
done

pids=(); fails=0
for g in "${FREE[@]}"; do
  [ -z "${QUEUE[$g]:-}" ] && continue
  (
    lane_fail=0
    IFS=';;' read -ra items <<< "${QUEUE[$g]}"
    for it in "${items[@]}"; do
      [ -z "$it" ] && continue
      IFS='|' read -r unit cfg ckpt preds <<< "$it"
      run_unit "$g" "$unit" "$cfg" "$ckpt" "$preds" || lane_fail=1
    done
    exit $lane_fail
  ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || fails=1; done

n=$(ls "$MARK" 2>/dev/null | grep -c '\.done$')
if [ "$fails" -eq 0 ] && [ "$n" -eq "${#UNITS[@]}" ]; then
  echo "GEMMA_ENCODERS_DONE units=$n $(date)"
else
  echo "GEMMA_ENCODERS_FAILED done=$n/${#UNITS[@]} $(date)"
  exit 1
fi
