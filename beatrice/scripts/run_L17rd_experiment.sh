#!/usr/bin/env bash
# Residual-direction experiment: retrain only_residual branch on the directional
# (norm+lens) features, then a 3-branch fusion (hidden@L17 + prob + residual-dir) plus
# leave-one-out fusions (drop residual, drop hidden). Configs carry gpu_id (the trainer
# pins CUDA_VISIBLE_DEVICES from it), so the 3 fusions run in parallel on gpu 2/3/5.
set -Euo pipefail
cd "$(dirname "$0")/.."
PY=python
C=configs/experiment/probe/L17rd
mkdir -p logs/L17rd

echo "[1/3] train only_residual(+direction) branch"
$PY scripts/train_probe.py --config $C/train_only_residual_L17rd.yaml > logs/L17rd/branch_residual.log 2>&1
tail -2 logs/L17rd/branch_residual.log | tr '\r' '\n' | grep -iE "best AUC|Done" || true

echo "[2/3] train 3 fusions in parallel (full / no_residual / no_hidden)"
$PY scripts/train_probe.py --config $C/train_fusion_L17rd.yaml             > logs/L17rd/full.log 2>&1 &
$PY scripts/train_probe.py --config $C/train_fusion_L17rd_no_residual.yaml > logs/L17rd/no_residual.log 2>&1 &
$PY scripts/train_probe.py --config $C/train_fusion_L17rd_no_hidden.yaml   > logs/L17rd/no_hidden.log 2>&1 &
wait
for v in full no_residual no_hidden; do echo "[$v] $(grep -iE 'best AUC' logs/L17rd/$v.log | tr '\r' '\n' | grep 'best AUC' | tail -1)"; done

echo "[3/3] eval"
for var in L17rd L17rd_no_residual L17rd_no_hidden; do
  for grp in validation test popqa; do
    $PY scripts/evaluate_probe.py --config $C/eval_${grp}_${var}.yaml 2>&1 | tail -1
  done
done
echo "DONE_L17RD_EXPERIMENT"
