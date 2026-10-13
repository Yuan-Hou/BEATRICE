#!/usr/bin/env bash
# Complete the gemma baseline battery to qwen parity — v2.
# The v1 layer features (sweep era) were hidden-only; the faithful battery needs
# the full schema (hid_logdet / attn_logdet / perplexity / entropies), so this
# driver re-extracts EVERYTHING (archived old dir: layer_features_v1_hiddenonly),
# spread across two GPUs, then runs faithful+llmsknow per setting, then eigenscore.
# B-class scorers already done (outputs/gemma4/predictions/baselines/bclass).
# Resumable via per-shard .done markers.
set -u
cd "${BEATRICE_ROOT:-.}"
PY=python
LOG=logs/gemma4/baselines
mkdir -p "$LOG"
read -ra G <<< "${BL_GPUS:?set BL_GPUS=\"a b\"}"
LF=data/gemma4/layer_features
CFG_Q=configs/experiment/probe/baselines/gemma4/baselines_question_noctx_gemma4.yaml
CFG_C=configs/experiment/probe/baselines/gemma4/baselines_context_withctx_gemma4.yaml
EVAL_GROUPS="validation test_id popqa_ood nq triviaqa"  # NB: "GROUPS" is a readonly bash special var — assignments silently no-op and $GROUPS expands to a numeric gid
export SE_CACHE_BASE=data/gemma4/se_cache
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

# job list: setting group shard  (train sharded x4; rewritten x4)
JOBS=()
for st in noctx withctx; do
  for s in 0 1 2 3; do JOBS+=("$st train $s/4"); done
  for g in validation test_id popqa_ood nq triviaqa; do JOBS+=("$st $g 0/1"); done
done
for s in 0 1 2 3; do JOBS+=("withctx train_rewritten $s/4"); done

ex_lane() {  # ex_lane <gpu> <job indices...>
  local gpu=$1; shift
  for i in "$@"; do
    read -r st g sh <<< "${JOBS[$i]}"
    local mark="$LF/$st/${g}__shard$(echo "$sh" | sed 's#/#of#').done"
    [ -f "$mark" ] && continue
    CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/gemma/extract_gemma_layer_features.py \
      --setting "$st" --group "$g" --shard "$sh" >> "$LOG/extract_${st}_${g}.log" 2>&1 \
      || { echo "[extract] FAIL $st/$g/$sh (gpu$gpu)"; return 1; }
    echo "[extract] done $st/$g/$sh (gpu$gpu, $(date))"
  done
}

idx0=(); idx1=()
for i in "${!JOBS[@]}"; do
  if [ $((i % 2)) -eq 0 ]; then idx0+=("$i"); else idx1+=("$i"); fi
done
ex_lane "${G[0]}" "${idx0[@]}" & P0=$!
ex_lane "${G[1]}" "${idx1[@]}" & P1=$!
f=0; wait $P0 || f=1; wait $P1 || f=1
[ $f -ne 0 ] && { echo "GEMMA_BASELINES_FAILED extraction $(date)"; exit 1; }
echo "[extract] ALL_DONE $(date)"

# ---------- scoring (gpu G[0]) + eigenscore (gpu G[1]) in parallel ----------
(
  for st in noctx withctx; do
    CFG=$CFG_Q; [ "$st" = withctx ] && CFG=$CFG_C
    CUDA_VISIBLE_DEVICES=${G[0]} "$PY" scripts/run_faithful_baselines.py --config "$CFG" \
      --setting $st --device cuda --layer-base "$LF" --groups $EVAL_GROUPS \
      --out-dir outputs/gemma4/predictions/baselines/faithful/$st \
      > "$LOG/faithful_$st.log" 2>&1 || { echo "[score] FAIL faithful $st"; exit 1; }
    CUDA_VISIBLE_DEVICES=${G[0]} "$PY" scripts/run_llmsknow.py --setting $st \
      --layer-base "$LF" --groups $EVAL_GROUPS \
      --out-dir outputs/gemma4/predictions/baselines/faithful/$st \
      > "$LOG/llmsknow_$st.log" 2>&1 || { echo "[score] FAIL llmsknow $st"; exit 1; }
    echo "[score] faithful+llmsknow $st done ($(date))"
  done
) & S1=$!
(
  for st in noctx withctx; do
    CUDA_VISIBLE_DEVICES=${G[1]} "$PY" scripts/extract_eigenscore.py \
      --setting $st --model-name google/gemma-4-E4B-it \
      --samples-base data/gemma4/samples --out-dir data/gemma4/eigenscore \
      --groups $EVAL_GROUPS >> "$LOG/eigenscore_$st.log" 2>&1 \
      || { echo "[eig] FAIL eigenscore $st"; exit 1; }
    echo "[eig] eigenscore $st done ($(date))"
  done
) & S2=$!
f=0; wait $S1 || f=1; wait $S2 || f=1
if [ $f -eq 0 ]; then echo "GEMMA_BASELINES_DONE $(date)"; else echo "GEMMA_BASELINES_FAILED scoring $(date)"; exit 1; fi
