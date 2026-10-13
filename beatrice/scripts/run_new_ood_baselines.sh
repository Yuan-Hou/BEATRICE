#!/bin/bash
# Full NQ / TriviaQA baseline pipeline.
# Extends the internal-state baselines to the two extra OOD sets (nq=1182,
# triviaqa=1277, judge-consensus purified), three families: (1) hidden-probe
# faithful (layer feature extraction -> rescoring), (2) sampling consistency
# (vLLM sampling -> SE/SelfCheck/BayesSE/EigenScore), (3) Tier-C RAG
# (lookback/redeep/lumina). Calibration stays validation-only; the new sets are
# report-only. Rewrites of the old groups are deterministic recomputation on
# the same data (for metric consistency).
#
# Usage (detach from the parent process tree):
#   setsid nohup bash scripts/run_new_ood_baselines.sh > logs/new_ood_baselines/driver.out 2>&1 &
# Idempotent: each step drops $LOG/<name>.ok on success; reruns skip done steps.
set -u
cd "${BEATRICE_ROOT:-.}"
PY=python
export PYTORCH_NVML_BASED_CUDA_CHECK=0
LOG=logs/new_ood_baselines
mkdir -p "$LOG"
CFG_Q=configs/experiment/probe/baselines/baselines_question_noctx_qwen9b.yaml
CFG_C=configs/experiment/probe/baselines/baselines_context_withctx_qwen9b.yaml
ALLG="validation test_id popqa_ood nq triviaqa"
GEN_URL=http://localhost:2333/v1   # generation backend (vLLM OpenAI-compatible)

step(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG/driver.log"; }

run(){ # run <name> <gpu|-> <cmd...>; GPU tasks retry once on failure
  local name=$1 gpu=$2; shift 2
  if [ -f "$LOG/$name.ok" ]; then step "SKIP $name"; return 0; fi
  local tries=1; [ "$gpu" != "-" ] && tries=2
  local rc=1
  for ((t=1; t<=tries; t++)); do
    step "START $name (gpu=$gpu, try=$t)"
    if [ "$gpu" = "-" ]; then
      "$@" > "$LOG/$name.log" 2>&1; rc=$?
    else
      CUDA_VISIBLE_DEVICES=$gpu "$@" > "$LOG/$name.log" 2>&1; rc=$?
    fi
    if [ $rc -eq 0 ]; then touch "$LOG/$name.ok"; step "DONE $name"; return 0; fi
    step "FAIL $name rc=$rc (try=$t)"; sleep 20
  done
  return $rc
}

need(){ # stage gate: all .ok markers must exist or the pipeline stops
  for n in "$@"; do
    if [ ! -f "$LOG/$n.ok" ]; then step "GATE-BLOCKED: missing $n.ok — aborting"; exit 1; fi
  done
}

step "================ pipeline start ================"

# ---------------- Stage A: layer feature extraction + vLLM sampling ----------------
run lf_noctx_nq       1 $PY scripts/extract_layer_features.py --config $CFG_Q --setting noctx   --group nq       --with-attention 1 &
run lf_noctx_tqa      2 $PY scripts/extract_layer_features.py --config $CFG_Q --setting noctx   --group triviaqa --with-attention 1 &
run lf_withctx_nq     5 $PY scripts/extract_layer_features.py --config $CFG_C --setting withctx --group nq       --with-attention 1 &
run lf_withctx_tqa    6 $PY scripts/extract_layer_features.py --config $CFG_C --setting withctx --group triviaqa --with-attention 1 &
(
  run sample_noctx   - $PY scripts/sample_generations.py --config $CFG_Q --setting noctx   --splits eval --eval-names nq triviaqa --base-url $GEN_URL --workers 32 &&
  run sample_withctx - $PY scripts/sample_generations.py --config $CFG_C --setting withctx --splits eval --eval-names nq triviaqa --base-url $GEN_URL --workers 32
) &
wait
need lf_noctx_nq lf_noctx_tqa lf_withctx_nq lf_withctx_tqa sample_noctx sample_withctx
step "======== Stage A done ========"

# ---------------- Stage B: Tier-C extraction + sampling-based scoring ----------------
( run lumina_nq    1 $PY scripts/run_lumina.py      --config $CFG_C --group nq &&
  run lumina_tqa   1 $PY scripts/run_lumina.py      --config $CFG_C --group triviaqa ) &
( run redeep_nq    2 $PY scripts/extract_redeep.py  --config $CFG_C --group nq       --shard 0/1 &&
  run redeep_tqa   2 $PY scripts/extract_redeep.py  --config $CFG_C --group triviaqa --shard 0/1 &&
  run eig_withctx  2 $PY scripts/extract_eigenscore.py --setting withctx --groups nq triviaqa ) &
( run lookback_nq  5 $PY scripts/extract_lookback.py --config $CFG_C --group nq &&
  run lookback_tqa 5 $PY scripts/extract_lookback.py --config $CFG_C --group triviaqa &&
  run secache_noctx   5 $PY scripts/cache_semantic_entropy.py --setting noctx   --groups nq triviaqa &&
  run secache_withctx 5 $PY scripts/cache_semantic_entropy.py --setting withctx --groups nq triviaqa &&
  run selfcheck_noctx   5 $PY scripts/run_selfcheckgpt.py --setting noctx   --groups nq triviaqa &&
  run selfcheck_withctx 5 $PY scripts/run_selfcheckgpt.py --setting withctx --groups nq triviaqa &&
  run bayes_noctx   5 $PY scripts/run_bayesian_se.py --setting noctx   --groups nq triviaqa &&
  run bayes_withctx 5 $PY scripts/run_bayesian_se.py --setting withctx --groups nq triviaqa ) &
( run eig_noctx    6 $PY scripts/extract_eigenscore.py --setting noctx --groups nq triviaqa ) &
wait
need lumina_nq lumina_tqa redeep_nq redeep_tqa eig_withctx eig_noctx \
     lookback_nq lookback_tqa secache_noctx secache_withctx \
     selfcheck_noctx selfcheck_withctx bayes_noctx bayes_withctx
step "======== Stage B done ========"

# ---------------- Stage C: faithful rescoring (incl. nq/triviaqa) + Tier-C scoring ----------------
( run faithful_noctx   1 $PY scripts/run_faithful_baselines.py --config $CFG_Q --setting noctx   --device cuda --groups $ALLG &&
  run llmsknow_noctx   1 $PY scripts/run_llmsknow.py --setting noctx   --groups $ALLG ) &
( run faithful_withctx 2 $PY scripts/run_faithful_baselines.py --config $CFG_C --setting withctx --device cuda --groups $ALLG &&
  run llmsknow_withctx 2 $PY scripts/run_llmsknow.py --setting withctx --groups $ALLG ) &
( run tierc  - $PY scripts/run_tierc_baselines.py --groups $ALLG &&
  run bclass - $PY scripts/run_bclass_scorers.py  --groups $ALLG ) &
wait
need faithful_noctx faithful_withctx llmsknow_noctx llmsknow_withctx tierc bclass
step "======== Stage C done ========"

step "================ pipeline ALL DONE ================"
# Per-method scores land under outputs/; see docs/PIPELINE.md stage 7.
