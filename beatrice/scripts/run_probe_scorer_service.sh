#!/usr/bin/env bash
# Supervisor for the probe scorer HTTP service (docs/probe-scorer-service.md).
# - picks the freest GPU via the CUDA driver API (works despite the broken host NVML),
#   excluding gpu0 (reserved for the user's vLLM);
# - starts the server and restarts it whenever it dies (NVML-init lottery, OOM, etc.);
# - one timestamped log per run under logs/probe_scorer/.
# Usage:  bash scripts/run_probe_scorer_service.sh            # port 8008
#         PORT=8010 bash scripts/run_probe_scorer_service.sh
set -u
cd "$(dirname "$0")/.."
PY=python
CFG=configs/experiment/services/serve_probe_scorer_qwen.yaml
PORT="${PORT:-8008}"
MIN_FREE_GB="${MIN_FREE_GB:-22}"
mkdir -p logs/probe_scorer
export PYTORCH_NVML_BASED_CUDA_CHECK=0

pick_gpu() {
  $PY - "$MIN_FREE_GB" <<'EOF' 2>/dev/null
import os, sys
os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "0"
import torch
need = float(sys.argv[1]) * 1e9
best, bfree = None, 0.0
for i in range(torch.cuda.device_count()):
    if i == 0:  # reserved for the user's vLLM
        continue
    try:
        free, _ = torch.cuda.mem_get_info(i)
    except Exception:
        continue
    if free > bfree:
        best, bfree = i, free
print(best if best is not None and bfree >= need else "")
EOF
}

restarts=0
while true; do
  GPU=$(pick_gpu | tail -1)
  if [ -z "$GPU" ]; then
    echo "[svc $(date +%H:%M:%S)] no GPU with >=${MIN_FREE_GB}GB free; retrying in 120s"
    sleep 120
    continue
  fi
  LOG=logs/probe_scorer/server_$(date +%Y%m%d_%H%M%S)_gpu${GPU}.log
  echo "[svc $(date +%H:%M:%S)] starting on gpu${GPU} port ${PORT} -> ${LOG}"
  $PY scripts/serve_probe_scorer.py --config "$CFG" --gpu "$GPU" --port "$PORT" > "$LOG" 2>&1
  code=$?
  restarts=$((restarts + 1))
  echo "[svc $(date +%H:%M:%S)] server exited code=${code} (restart #${restarts}); tail:"
  tail -3 "$LOG"
  if [ "$restarts" -ge 200 ]; then
    echo "[svc] too many restarts; giving up"
    exit 1
  fi
  sleep 8
done
