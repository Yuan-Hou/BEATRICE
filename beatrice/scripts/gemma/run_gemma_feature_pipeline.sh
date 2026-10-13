#!/usr/bin/env bash
# Rolling Gemma training-feature pipeline: per unit (dataset shard) do
#   extract -> pack -> bit-exact verify -> delete tensors+sidecars -> .done
# so transient disk stays at ~one-shard footprint. Resumable: .done units skip,
# extraction itself resumes via meta. Units come from the manifest written by
# scripts/gemma/make_gemma_feature_configs.py.
#
# Usage:
#   bash scripts/gemma/run_gemma_feature_pipeline.sh <LAYER>          # full run
#   bash scripts/gemma/run_gemma_feature_pipeline.sh <LAYER> smoke    # 12-row dry run
set -u
cd "${BEATRICE_ROOT:-.}"
PY=python
LAYER=${1:?usage: run_gemma_feature_pipeline.sh LAYER [smoke]}
MODE=${2:-full}
SFX=""; [ "$MODE" = "smoke" ] && SFX="_smoke"
LOG=logs/gemma4
MARK=data/gemma4/features${SFX}/.markers
mkdir -p "$LOG" "$MARK"
MIN_FREE_MB=25000
MAX_GPUS=3  # user cap 2026-07-17: use at most 3 GPUs

MAKEFLAGS=(--layer "$LAYER"); [ "$MODE" = "smoke" ] && MAKEFLAGS+=(--smoke)
"$PY" scripts/gemma/make_gemma_feature_configs.py "${MAKEFLAGS[@]}" || {
  echo "GEMMA_FEATURE_PIPELINE_FAILED configgen $(date)"; exit 1; }
MANIFEST=configs/experiment/features/gemma4/manifest${SFX}.json
PACKED_ROOT=$("$PY" -c "import json;print(json.load(open('$MANIFEST'))['packed_root'])")

mapfile -t UNITS < <("$PY" -c "
import json
for u in json.load(open('$MANIFEST'))['units']:
    print('|'.join([u['unit'], u['family'], u['config'], u['meta'], u['tensor_dir'], u['sidecar_dir']]))
")
echo "[featpipe] layer=$LAYER mode=$MODE units=${#UNITS[@]} packed_root=$PACKED_ROOT"

mapfile -t FREE < <(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits \
  | awk -F', ' -v m=$MIN_FREE_MB '($2-$3)>=m {print $1}')
FREE=("${FREE[@]:0:$MAX_GPUS}")
if [ "${#FREE[@]}" -eq 0 ]; then echo "GEMMA_FEATURE_PIPELINE_FAILED no free gpu $(date)"; exit 1; fi
[ "$MODE" = "smoke" ] && FREE=("${FREE[0]}")
echo "[featpipe] gpus: ${FREE[*]}"

run_unit() {  # $1 gpu, $2.. unit fields
  local g=$1 unit=$2 family=$3 cfg=$4 meta=$5 tdir=$6 scdir=$7
  local log="$LOG/feat_${unit}.log"
  if [ -f "$MARK/${unit}.done" ]; then echo "[featpipe] skip $unit (.done)"; return 0; fi

  echo "[featpipe] gpu$g extract $unit ($(date))"
  # cap intra-op threads: 6 concurrent extractors with default (=ncore) threads
  # thrash the box (observed loadavg ~450/201, GPUs idle, 3-5x slowdown)
  OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24 \
    CUDA_VISIBLE_DEVICES=$g "$PY" scripts/extract_features.py --config "$cfg" >> "$log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ] || ! tail -20 "$log" | grep -q "Total failed:"; then
    echo "[featpipe] FAIL extract $unit rc=$rc (crash; see $log)"; return 1
  fi

  # success-only meta: gemma occasionally greedy-generates an empty answer; only
  # that failure reason is tolerated, and only below the rate gate.
  # empty-greedy rates measured per 40-row samples: musique-ctx 5%, musique-q 12%,
  # nq-ctx 25% (harmful retrieval), tqa-q 5%. Reason whitelist is the real bug gate;
  # the rate is a secondary sanity check, so keep it loose.
  local rate=0.30
  "$PY" scripts/gemma/filter_success_meta.py --meta "$meta" --max-fail-rate "$rate" >> "$log" 2>&1
  if ! tail -5 "$log" | grep -q "FILTER_META_OK"; then
    echo "[featpipe] FAIL filter $unit (unexpected failure reasons/rate; see $log)"; return 1
  fi
  local meta_ok="${meta%.jsonl}_ok.jsonl"

  echo "[featpipe] pack $unit ($(date))"
  "$PY" scripts/pack_sidecar_features.py --meta "${family}:${meta_ok}" \
      --out-root "$PACKED_ROOT" --workers 8 >> "$log" 2>&1
  if ! tail -5 "$log" | grep -q "PACK_ALL_DONE"; then
    echo "[featpipe] FAIL pack $unit (see $log)"; return 1
  fi

  echo "[featpipe] verify $unit ($(date))"
  "$PY" scripts/verify_packed_features.py --meta "${family}:${meta_ok}" \
      --packed-root "$PACKED_ROOT" >> "$log" 2>&1
  if ! tail -5 "$log" | grep -q "VERIFY_PACKED_OK"; then
    echo "[featpipe] FAIL verify $unit -- keeping raw artifacts (see $log)"; return 1
  fi

  rm -rf "$tdir" "$scdir"
  touch "$MARK/${unit}.done"
  echo "[featpipe] done $unit ($(date))"
}

# greedy lane assignment: manifest is heaviest-first; round-robin over free gpus.
# Filter out .done units FIRST so resumed runs spread the remainder across lanes
# (otherwise skips keep their slots and stragglers pile onto one gpu).
PENDING=()
for it in "${UNITS[@]}"; do
  IFS='|' read -r u _rest <<< "$it"
  if [ -f "$MARK/${u}.done" ]; then echo "[featpipe] skip $u (.done)"; else PENDING+=("$it"); fi
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
      IFS='|' read -r unit family cfg meta tdir scdir <<< "$it"
      run_unit "$g" "$unit" "$family" "$cfg" "$meta" "$tdir" "$scdir" || lane_fail=1
    done
    exit $lane_fail
  ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || fails=1; done

n_done=$(ls "$MARK" 2>/dev/null | grep -c '\.done$')
if [ "$fails" -eq 0 ] && [ "$n_done" -eq "${#UNITS[@]}" ]; then
  echo "GEMMA_FEATURE_PIPELINE_DONE units=$n_done $(date)"
else
  echo "GEMMA_FEATURE_PIPELINE_FAILED done=$n_done/${#UNITS[@]} $(date)"
  exit 1
fi
