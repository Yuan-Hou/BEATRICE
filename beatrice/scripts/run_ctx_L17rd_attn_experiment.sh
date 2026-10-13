#!/usr/bin/env bash
# With-context 4-branch simplex experiment. Waits for the two feature extractions to
# finish (GPU 4: rewritten 34k -> single meta; GPU 6: original 34k -> 4 metas, popqa
# written last), then trains the four single-signal branches (attn=512), the full
# fusion + 4 leave-one-out fusions, and runs validation/test/popqa evals.
set -Euo pipefail
cd "$(dirname "$0")/.."
PY=python
C=configs/experiment/probe/ctx_L17rd_attn
RW_META=data/features_judge_consensus/qwen3_5_9b_rewritten_context_L17_rd/rewritten_context_cqa_features_meta.jsonl
ORIG_LAST=data/features_judge_consensus/qwen3_5_9b_L17_rd/popqa_features_meta.jsonl
mkdir -p logs/ctx_L17rd_attn

echo "[0/3] waiting for feature extractions (rewritten meta + original popqa meta) ..."
while [ ! -s "$RW_META" ] || [ ! -s "$ORIG_LAST" ]; do sleep 300; done
sleep 60  # let the last writer flush
for m in "$RW_META" data/features_judge_consensus/qwen3_5_9b_L17_rd/{musique,hotpotqa_supporting_only,twowiki_supporting_only,popqa}_features_meta.jsonl; do
  echo "  $(wc -l < "$m") $m"
done

echo "[1/3] train 4 single-signal branches (2 per GPU: 4/4/6/6)"
for b in only_hidden only_probability only_residual only_attention; do
  $PY scripts/train_probe.py --config $C/train_${b}_ctxL17rd_attn.yaml > logs/ctx_L17rd_attn/branch_${b}.log 2>&1 &
done
wait
for b in only_hidden only_probability only_residual only_attention; do
  echo "[$b] $(tr '\r' '\n' < logs/ctx_L17rd_attn/branch_${b}.log | grep -iE 'best AUC' | tail -1)"
  test -f outputs/ctx_L17rd_attn_probes/${b}/context_checkpoint.pth || { echo "FATAL: missing checkpoint for $b"; exit 1; }
done

echo "[2/3] train 5 fusions in parallel (full + 4 leave-one-out)"
$PY scripts/train_probe.py --config $C/train_fusion_ctxL17rd_attn.yaml > logs/ctx_L17rd_attn/fusion_full.log 2>&1 &
for d in hidden probability residual attention; do
  $PY scripts/train_probe.py --config $C/train_fusion_ctxL17rd_attn_no_${d}.yaml > logs/ctx_L17rd_attn/fusion_no_${d}.log 2>&1 &
done
wait
for v in full no_hidden no_probability no_residual no_attention; do
  echo "[fusion_$v] $(tr '\r' '\n' < logs/ctx_L17rd_attn/fusion_${v}.log | grep -iE 'best AUC' | tail -1)"
done

echo "[3/3] eval all fusion variants on validation/test/popqa"
for var in ctxL17rd_attn ctxL17rd_attn_no_hidden ctxL17rd_attn_no_probability ctxL17rd_attn_no_residual ctxL17rd_attn_no_attention; do
  for grp in validation test popqa; do
    $PY scripts/evaluate_probe.py --config $C/eval_${grp}_${var}.yaml > logs/ctx_L17rd_attn/eval_${grp}_${var}.log 2>&1 \
      && echo "[eval ${grp} ${var}] $(tr '\r' '\n' < logs/ctx_L17rd_attn/eval_${grp}_${var}.log | grep -iE 'auc' | tail -1)" \
      || echo "[eval ${grp} ${var}] FAILED"
  done
done
echo "DONE_CTX_L17RD_ATTN_EXPERIMENT"
