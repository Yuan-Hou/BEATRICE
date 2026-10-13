#!/usr/bin/env bash

GPU_MAX_USED_PCT="${GPU_MAX_USED_PCT:-10}"
MAX_PARALLEL_GPUS="${MAX_PARALLEL_GPUS:-3}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-30}"
GPU_START_DELAY_SECONDS="${GPU_START_DELAY_SECONDS:-20}"
SWEEP_PARALLEL="${SWEEP_PARALLEL:-1}"
# When enabled, a scheduler may place more than one job on the same GPU if
# nvidia-smi still reports memory usage below GPU_MAX_USED_PCT. The number of
# distinct GPUs used remains capped by MAX_PARALLEL_GPUS.
GPU_ALLOW_MULTIPLE_PER_GPU="${GPU_ALLOW_MULTIPLE_PER_GPU:-0}"

SWEEP_GPU_POOL=()
SWEEP_BUSY_GPUS=()
SWEEP_BUSY_PIDS=()
SWEEP_BUSY_RUNS=()

_gpu_sweep_query_available() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[gpu-sweep] nvidia-smi not found; set SWEEP_PARALLEL=0 or run on a GPU host" >&2
    return 1
  fi

  local allowed="${GPU_IDS:-}"
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
    | awk -F, -v max_pct="${GPU_MAX_USED_PCT}" -v allowed="${allowed}" '
      BEGIN {
        n = split(allowed, arr, ",")
        for (i = 1; i <= n; i++) {
          gsub(/^[ \t]+|[ \t]+$/, "", arr[i])
          if (arr[i] != "") allow[arr[i]] = 1
        }
      }
      {
        idx = $1; used = $2; total = $3
        gsub(/^[ \t]+|[ \t]+$/, "", idx)
        gsub(/^[ \t]+|[ \t]+$/, "", used)
        gsub(/^[ \t]+|[ \t]+$/, "", total)
        if (allowed != "" && !(idx in allow)) next
        if (total > 0) {
          pct = used * 100.0 / total
          if (pct < max_pct) printf "%.6f\t%d\t%s\n", pct, used, idx
        }
      }' \
    | sort -n -k1,1 -k2,2 \
    | awk -F'\t' '{print $3}'
}

gpu_sweep_init() {
  SWEEP_GPU_POOL=()

  if [[ "${SWEEP_PARALLEL}" != "1" ]]; then
    echo "[gpu-sweep] SWEEP_PARALLEL=${SWEEP_PARALLEL}; using sequential mode"
    return 0
  fi

  while [[ "${#SWEEP_GPU_POOL[@]}" -eq 0 ]]; do
    mapfile -t SWEEP_GPU_POOL < <(_gpu_sweep_query_available)
    if [[ "${#SWEEP_GPU_POOL[@]}" -eq 0 ]]; then
      echo "[gpu-sweep] no GPU currently below ${GPU_MAX_USED_PCT}% memory usage; waiting ${GPU_POLL_SECONDS}s" >&2
      sleep "${GPU_POLL_SECONDS}"
    fi
  done

  if [[ "${#SWEEP_GPU_POOL[@]}" -gt "${MAX_PARALLEL_GPUS}" ]]; then
    SWEEP_GPU_POOL=("${SWEEP_GPU_POOL[@]:0:${MAX_PARALLEL_GPUS}}")
  fi

  echo "[gpu-sweep] candidate_gpus=${SWEEP_GPU_POOL[*]} max_distinct_gpus=${MAX_PARALLEL_GPUS} max_used_pct=${GPU_MAX_USED_PCT} allow_multiple_per_gpu=${GPU_ALLOW_MULTIPLE_PER_GPU}" >&2
}

_gpu_sweep_active_gpu_count() {
  local seen=" "
  local count=0
  local gpu
  for gpu in "${SWEEP_BUSY_GPUS[@]}"; do
    if [[ "${seen}" != *" ${gpu} "* ]]; then
      seen+="${gpu} "
      count=$((count + 1))
    fi
  done
  echo "${count}"
}

_gpu_sweep_is_active_gpu() {
  local gpu="$1"
  local g
  for g in "${SWEEP_BUSY_GPUS[@]}"; do
    [[ "${g}" == "${gpu}" ]] && return 0
  done
  return 1
}

_gpu_sweep_refresh_pool() {
  local candidates=()
  local gpu active_count
  mapfile -t candidates < <(_gpu_sweep_query_available)
  for gpu in "${candidates[@]}"; do
    local exists=0
    local old
    for old in "${SWEEP_GPU_POOL[@]}"; do
      if [[ "${old}" == "${gpu}" ]]; then
        exists=1
        break
      fi
    done
    [[ "${exists}" == "1" ]] && continue
    active_count="$(_gpu_sweep_active_gpu_count)"
    if [[ "${active_count}" -lt "${MAX_PARALLEL_GPUS}" ]]; then
      SWEEP_GPU_POOL+=("${gpu}")
    fi
  done
}

_gpu_sweep_is_busy() {
  local gpu="$1"
  local g
  for g in "${SWEEP_BUSY_GPUS[@]}"; do
    [[ "${g}" == "${gpu}" ]] && return 0
  done
  return 1
}

_gpu_sweep_external_available() {
  local gpu="$1"
  local line used total
  line="$(nvidia-smi -i "${gpu}" --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -n 1)"
  used="${line%%,*}"
  total="${line##*,}"
  used="${used//[[:space:]]/}"
  total="${total//[[:space:]]/}"
  [[ -n "${used}" && -n "${total}" && "${total}" -gt 0 ]] || return 1
  awk -v used="${used}" -v total="${total}" -v max_pct="${GPU_MAX_USED_PCT}" 'BEGIN { exit !((used * 100.0 / total) < max_pct) }'
}

gpu_sweep_prune_finished() {
  local new_gpus=()
  local new_pids=()
  local new_runs=()
  local running_pids pid i is_running rp
  running_pids=" $(jobs -pr) "
  for i in "${!SWEEP_BUSY_PIDS[@]}"; do
    pid="${SWEEP_BUSY_PIDS[$i]}"
    is_running=0
    for rp in ${running_pids}; do
      if [[ "${rp}" == "${pid}" ]]; then
        is_running=1
        break
      fi
    done
    if [[ "${is_running}" == "1" ]]; then
      stat="$(ps -p "${pid}" -o stat= 2>/dev/null | tr -d ' ')"
      if [[ "${stat}" == *Z* || -z "${stat}" ]]; then
        wait "${pid}" 2>/dev/null || true
        echo "[gpu-sweep] finished run=${SWEEP_BUSY_RUNS[$i]} gpu=${SWEEP_BUSY_GPUS[$i]} pid=${pid}" >&2
      else
        new_gpus+=("${SWEEP_BUSY_GPUS[$i]}")
        new_pids+=("${pid}")
        new_runs+=("${SWEEP_BUSY_RUNS[$i]}")
      fi
    else
      wait "${pid}" 2>/dev/null || true
      echo "[gpu-sweep] finished run=${SWEEP_BUSY_RUNS[$i]} gpu=${SWEEP_BUSY_GPUS[$i]} pid=${pid}" >&2
    fi
  done
  SWEEP_BUSY_GPUS=("${new_gpus[@]}")
  SWEEP_BUSY_PIDS=("${new_pids[@]}")
  SWEEP_BUSY_RUNS=("${new_runs[@]}")
}

gpu_sweep_acquire_gpu() {
  if [[ "${SWEEP_PARALLEL}" != "1" ]]; then
    echo ""
    return 0
  fi

  local gpu active_count busy_count
  while true; do
    gpu_sweep_prune_finished
    _gpu_sweep_refresh_pool || true
    for gpu in "${SWEEP_GPU_POOL[@]}"; do
      if ! _gpu_sweep_external_available "${gpu}"; then
        continue
      fi
      if [[ "${GPU_ALLOW_MULTIPLE_PER_GPU}" == "1" ]]; then
        if _gpu_sweep_is_active_gpu "${gpu}"; then
          echo "${gpu}"
          return 0
        fi
        active_count="$(_gpu_sweep_active_gpu_count)"
        if [[ "${active_count}" -lt "${MAX_PARALLEL_GPUS}" ]]; then
          echo "${gpu}"
          return 0
        fi
      else
        if ! _gpu_sweep_is_busy "${gpu}"; then
          active_count="$(_gpu_sweep_active_gpu_count)"
          if [[ "${active_count}" -lt "${MAX_PARALLEL_GPUS}" ]]; then
            echo "${gpu}"
            return 0
          fi
        fi
      fi
    done
    active_count="$(_gpu_sweep_active_gpu_count)"
    busy_count="${#SWEEP_BUSY_PIDS[@]}"
    echo "[gpu-sweep] waiting for GPU memory < ${GPU_MAX_USED_PCT}%; active_gpus=${active_count}/${MAX_PARALLEL_GPUS}; active_jobs=${busy_count}; busy=${SWEEP_BUSY_RUNS[*]:-none}" >&2
    sleep "${GPU_POLL_SECONDS}"
  done
}

gpu_sweep_mark_busy() {
  local gpu="$1"
  local pid="$2"
  local run_id="$3"
  [[ -n "${gpu}" ]] || return 0
  SWEEP_BUSY_GPUS+=("${gpu}")
  SWEEP_BUSY_PIDS+=("${pid}")
  SWEEP_BUSY_RUNS+=("${run_id}")
}

gpu_sweep_wait_all() {
  local pid
  for pid in "${SWEEP_BUSY_PIDS[@]}"; do
    wait "${pid}" || true
  done
  SWEEP_BUSY_GPUS=()
  SWEEP_BUSY_PIDS=()
  SWEEP_BUSY_RUNS=()
}

gpu_sweep_kill_all() {
  local pid
  for pid in "${SWEEP_BUSY_PIDS[@]}"; do
    kill "${pid}" >/dev/null 2>&1 || true
  done
}
