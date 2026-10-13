#!/usr/bin/env bash
# Pairwise-modality encoder experiment (question-only, GPU 3): train the 6 pair
# encoders in parallel, then the two fusion variants (pairs6, singles4+pairs6), then
# val/test/popqa evals. Singles are reused frozen from outputs/L17rd_attn_probes.
set -Euo pipefail
cd "$(dirname "$0")/.."
PY=python
C=configs/experiment/probe/L17rd_attn_pairs
mkdir -p logs/L17rd_attn_pairs

echo "[1/3] train 6 pairwise branches in parallel"
for b in hid_prob hid_resid hid_attn prob_resid prob_attn resid_attn; do
  $PY scripts/train_probe.py --config $C/train_pair_${b}.yaml > logs/L17rd_attn_pairs/branch_${b}.log 2>&1 &
done
wait
for b in hid_prob hid_resid hid_attn prob_resid prob_attn resid_attn; do
  echo "[$b] $(tr '\r' '\n' < logs/L17rd_attn_pairs/branch_${b}.log | grep -iE 'best AUC' | tail -1)"
  test -f outputs/L17rd_attn_pairs_probes/${b}/question_checkpoint.pth || { echo "FATAL: missing checkpoint for $b"; exit 1; }
done

echo "[2/3] train 2 fusion variants in parallel"
for v in pairs6 singles4_pairs6; do
  $PY scripts/train_probe.py --config $C/train_fusion_${v}.yaml > logs/L17rd_attn_pairs/fusion_${v}.log 2>&1 &
done
wait
for v in pairs6 singles4_pairs6; do
  echo "[fusion_$v] $(tr '\r' '\n' < logs/L17rd_attn_pairs/fusion_${v}.log | grep -iE 'best AUC' | tail -1)"
done

echo "[3/3] eval"
for v in pairs6 singles4_pairs6; do
  for grp in validation test popqa; do
    $PY scripts/evaluate_probe.py --config $C/eval_${grp}_${v}.yaml > logs/L17rd_attn_pairs/eval_${grp}_${v}.log 2>&1 \
      && echo "[eval ${grp} ${v}] $(tr '\r' '\n' < logs/L17rd_attn_pairs/eval_${grp}_${v}.log | grep -iE 'auc' | tail -1)" \
      || echo "[eval ${grp} ${v}] FAILED"
  done
done
echo "DONE_L17RD_ATTN_PAIRS_EXPERIMENT"
