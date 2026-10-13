#!/usr/bin/env bash
# 4-branch simplex experiment: retrain all four single-signal branches at
# attention_channels=256 (use_attention_mass=true; hidden@L17 / probability /
# residual+direction / attention entropy+mass), then the full 4-branch fusion plus
# four leave-one-out fusions, then val/test/popqa evals for each fusion variant.
# Everything pinned to GPU 3 via config gpu_id (trainer sets CUDA_VISIBLE_DEVICES).
set -Euo pipefail
cd "$(dirname "$0")/.."
PY=python
C=configs/experiment/probe/L17rd_attn
mkdir -p logs/L17rd_attn

echo "[1/3] train 4 single-signal branches in parallel (attn=256)"
for b in only_hidden only_probability only_residual only_attention; do
  $PY scripts/train_probe.py --config $C/train_${b}_L17rd_attn.yaml > logs/L17rd_attn/branch_${b}.log 2>&1 &
done
wait
for b in only_hidden only_probability only_residual only_attention; do
  echo "[$b] $(tr '\r' '\n' < logs/L17rd_attn/branch_${b}.log | grep -iE 'best AUC' | tail -1)"
  test -f outputs/L17rd_attn_probes/${b}/question_checkpoint.pth || { echo "FATAL: missing checkpoint for $b"; exit 1; }
done

echo "[2/3] train 5 fusions in parallel (full + 4 leave-one-out)"
$PY scripts/train_probe.py --config $C/train_fusion_L17rd_attn.yaml > logs/L17rd_attn/fusion_full.log 2>&1 &
for d in hidden probability residual attention; do
  $PY scripts/train_probe.py --config $C/train_fusion_L17rd_attn_no_${d}.yaml > logs/L17rd_attn/fusion_no_${d}.log 2>&1 &
done
wait
for v in full no_hidden no_probability no_residual no_attention; do
  echo "[fusion_$v] $(tr '\r' '\n' < logs/L17rd_attn/fusion_${v}.log | grep -iE 'best AUC' | tail -1)"
done

echo "[3/3] eval all fusion variants on validation/test/popqa"
for var in L17rd_attn L17rd_attn_no_hidden L17rd_attn_no_probability L17rd_attn_no_residual L17rd_attn_no_attention; do
  for grp in validation test popqa; do
    $PY scripts/evaluate_probe.py --config $C/eval_${grp}_${var}.yaml > logs/L17rd_attn/eval_${grp}_${var}.log 2>&1 \
      && echo "[eval ${grp} ${var}] $(tr '\r' '\n' < logs/L17rd_attn/eval_${grp}_${var}.log | grep -iE 'auc' | tail -1)" \
      || echo "[eval ${grp} ${var}] FAILED"
  done
done
echo "DONE_L17RD_ATTN_EXPERIMENT"
