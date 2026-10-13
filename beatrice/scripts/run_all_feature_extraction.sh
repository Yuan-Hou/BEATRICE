#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_all_feature_extraction.sh

Environment:
  PYTHON_BIN            Python executable. Default: python
  FEATURE_RUN_ROOT      Output root for logs/generated configs. Default: outputs/feature_extraction_runs
  FEATURE_CONFIGS       Optional newline/space/comma separated config allowlist. Default: all configs/experiment/features/**/*.yaml
  INCLUDE_REWRITTEN     Include rewritten-context feature config. Default: 1
  SKIP_MISSING_INPUTS   Skip configs whose input_result_path does not exist. Default: 1
  DRY_RUN               Print jobs without running. Default: 0
  SWEEP_PARALLEL        Use GPU parallel scheduling. Default: 1
  MAX_PARALLEL_GPUS     Max distinct GPUs this script may occupy. Default: 3
  GPU_MAX_USED_PCT      GPU eligible below this memory usage percent. Default: 10
  GPU_ALLOW_MULTIPLE_PER_GPU
                        Allow stacking jobs on the same GPU while it remains below GPU_MAX_USED_PCT. Default: 1
  GPU_IDS               Optional comma-separated GPU allowlist, e.g. 0,1,2
  GPU_POLL_SECONDS      Poll interval while waiting for GPUs. Default: 30
  GPU_START_DELAY_SECONDS
                        Wait after launching each job before checking GPUs again. Default: 60

The script copies each feature config into FEATURE_RUN_ROOT/configs, forces:
  gpu_id: <acquired gpu>
  max_examples: null
  resume: true
  retry_failed: true
then runs:
  python scripts/extract_features.py --config <generated_config>

Logs and status are saved under FEATURE_RUN_ROOT/<timestamp>/.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
GPU_MAX_USED_PCT="${GPU_MAX_USED_PCT:-10}"
GPU_ALLOW_MULTIPLE_PER_GPU="${GPU_ALLOW_MULTIPLE_PER_GPU:-1}"
MAX_PARALLEL_GPUS="${MAX_PARALLEL_GPUS:-3}"
GPU_START_DELAY_SECONDS="${GPU_START_DELAY_SECONDS:-60}"
source "${SCRIPT_DIR}/gpu_sweep_lib.sh"
trap 'gpu_sweep_kill_all; exit 130' INT TERM

PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN="${DRY_RUN:-0}"
INCLUDE_REWRITTEN="${INCLUDE_REWRITTEN:-1}"
SKIP_MISSING_INPUTS="${SKIP_MISSING_INPUTS:-1}"
FEATURE_RUN_ROOT="${FEATURE_RUN_ROOT:-outputs/feature_extraction_runs}"
RUN_ID="feature_full_$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${FEATURE_RUN_ROOT}/${RUN_ID}"
CONFIG_DIR="${RUN_DIR}/configs"
LOG_DIR="${RUN_DIR}/logs"
SUMMARY_TSV="${RUN_DIR}/summary.tsv"
SUMMARY_JSONL="${RUN_DIR}/summary.jsonl"
mkdir -p "${CONFIG_DIR}" "${LOG_DIR}"

load_configs() {
  if [[ -n "${FEATURE_CONFIGS:-}" ]]; then
    printf '%s\n' "${FEATURE_CONFIGS}" | tr ', ' '\n' | sed '/^$/d'
  else
    find configs/experiment/features -type f -name '*.yaml' | sort
  fi
}

mapfile -t CONFIGS < <(load_configs)
if [[ "${INCLUDE_REWRITTEN}" != "1" ]]; then
  filtered=()
  for cfg in "${CONFIGS[@]}"; do
    [[ "${cfg}" == *rewritten_context* ]] && continue
    filtered+=("${cfg}")
  done
  CONFIGS=("${filtered[@]}")
fi

if [[ "${SKIP_MISSING_INPUTS}" == "1" ]]; then
  existing=()
  skipped_file="${RUN_DIR}/skipped_missing_inputs.txt"
  : > "${skipped_file}"
  for cfg in "${CONFIGS[@]}"; do
    input_path="$(${PYTHON_BIN} - "${cfg}" <<'PY_READ_CFG'
import sys, yaml
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f) or {}
print(cfg.get('input_result_path') or '')
PY_READ_CFG
)"
    if [[ -n "${input_path}" && ! -f "${input_path}" ]]; then
      echo -e "${cfg}	${input_path}" >> "${skipped_file}"
      echo "[features] skip missing input config=${cfg} input=${input_path}" >&2
      continue
    fi
    existing+=("${cfg}")
  done
  CONFIGS=("${existing[@]}")
fi

if [[ "${#CONFIGS[@]}" -eq 0 ]]; then
  echo "[features] no configs selected" >&2
  exit 1
fi

{
  printf 'job_id\tstatus\tgpu\tconfig\tgenerated_config\tlog\tmeta_path\ttensor_dir\ttoken_ids_dir\tsidecar_dir\tsuccess\tfailed\n'
} > "${SUMMARY_TSV}"
: > "${SUMMARY_JSONL}"

printf '%s\n' "${CONFIGS[@]}" > "${RUN_DIR}/selected_configs.txt"
echo "[features] run_dir=${RUN_DIR}"
echo "[features] configs=${#CONFIGS[@]}"
echo "[features] summary=${SUMMARY_TSV}"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '[features] DRY_RUN selected configs:\n'
  printf '  %s\n' "${CONFIGS[@]}"
  exit 0
fi

gpu_sweep_init

make_job_config() {
  local source_config="$1"
  local generated_config="$2"
  local gpu="$3"
  "${PYTHON_BIN}" - "${source_config}" "${generated_config}" "${gpu}" <<'PY'
import sys
from pathlib import Path
import yaml

src, dst, gpu = sys.argv[1:]
with open(src, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f) or {}

cfg['gpu_id'] = str(gpu) if gpu else cfg.get('gpu_id')
cfg['max_examples'] = None
cfg['resume'] = True
cfg['retry_failed'] = True
cfg['save_every_batches'] = 2000
feature = cfg.setdefault('feature', {})
feature.setdefault('answer_tokens', 32)
feature.setdefault('prob_top_k', 20)
feature.setdefault('attn_implementation', 'eager')
feature.setdefault('torch_dtype', 'bfloat16')
output = cfg.setdefault('output', {})
if output.get('tensor_dir') and not output.get('sidecar_dir'):
    output['sidecar_dir'] = str(output['tensor_dir']) + '_sidecars'

Path(dst).parent.mkdir(parents=True, exist_ok=True)
with open(dst, 'w', encoding='utf-8') as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY
}

summarize_job() {
  local job_id="$1"
  local status="$2"
  local gpu="$3"
  local source_config="$4"
  local generated_config="$5"
  local log_path="$6"
  "${PYTHON_BIN}" - "${SUMMARY_TSV}" "${SUMMARY_JSONL}" "${job_id}" "${status}" "${gpu}" "${source_config}" "${generated_config}" "${log_path}" <<'PY'
import json
import re
import sys
from pathlib import Path
import yaml

summary_tsv, summary_jsonl, job_id, status, gpu, source_config, generated_config, log_path = sys.argv[1:]
status = int(status)
with open(generated_config, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f) or {}
out = cfg.get('output', {}) or {}
log_text = Path(log_path).read_text(encoding='utf-8', errors='ignore') if Path(log_path).exists() else ''
success = failed = ''
m = re.search(r'Done\. Total success:\s*(\d+)/(\d+)', log_text)
if m:
    success = m.group(1)
m = re.search(r'Total failed:\s*(\d+)/(\d+)', log_text)
if m:
    failed = m.group(1)
record = {
    'job_id': job_id,
    'status': 'ok' if status == 0 else f'failed:{status}',
    'gpu': gpu,
    'config': source_config,
    'generated_config': generated_config,
    'log': log_path,
    'meta_path': out.get('meta_path', ''),
    'tensor_dir': out.get('tensor_dir', ''),
    'token_ids_dir': out.get('token_ids_dir', ''),
    'sidecar_dir': out.get('sidecar_dir', ''),
    'success': success,
    'failed': failed,
}
with open(summary_jsonl, 'a', encoding='utf-8') as f:
    f.write(json.dumps(record, ensure_ascii=False) + '\n')
with open(summary_tsv, 'a', encoding='utf-8') as f:
    f.write('\t'.join(str(record[k]) for k in [
        'job_id', 'status', 'gpu', 'config', 'generated_config', 'log',
        'meta_path', 'tensor_dir', 'token_ids_dir', 'sidecar_dir', 'success', 'failed'
    ]) + '\n')
PY
}

run_one() {
  local gpu="$1"
  local source_config="$2"
  local idx="$3"
  local base job_id generated_config log_path status
  base="$(basename "${source_config}" .yaml)"
  job_id="$(printf '%02d_%s' "${idx}" "${base}" | tr -cs '[:alnum:]_.-' '_')"
  generated_config="${CONFIG_DIR}/${job_id}.yaml"
  log_path="${LOG_DIR}/${job_id}.log"
  make_job_config "${source_config}" "${generated_config}" "${gpu}"
  echo "[features] start job=${job_id} gpu=${gpu:-sequential} config=${source_config}"
  set +e
  "${PYTHON_BIN}" scripts/extract_features.py --config "${generated_config}" > "${log_path}" 2>&1
  status=$?
  set -e
  summarize_job "${job_id}" "${status}" "${gpu}" "${source_config}" "${generated_config}" "${log_path}"
  if [[ "${status}" -ne 0 ]]; then
    echo "[features] failed job=${job_id} gpu=${gpu:-sequential} status=${status} log=${log_path}" >&2
  else
    echo "[features] done job=${job_id} gpu=${gpu:-sequential} log=${log_path}"
  fi
  return "${status}"
}

idx=0
for cfg in "${CONFIGS[@]}"; do
  idx=$((idx + 1))
  gpu="$(gpu_sweep_acquire_gpu)"
  if [[ "${SWEEP_PARALLEL:-1}" == "1" ]]; then
    (
      run_one "${gpu}" "${cfg}" "${idx}"
    ) &
    pid=$!
    gpu_sweep_mark_busy "${gpu}" "${pid}" "$(basename "${cfg}" .yaml)"
    sleep "${GPU_START_DELAY_SECONDS:-20}"
  else
    run_one "" "${cfg}" "${idx}" || true
  fi

done

gpu_sweep_wait_all

echo "[features] all jobs finished"
echo "[features] summary=${SUMMARY_TSV}"
echo "[features] logs=${LOG_DIR}"
