#!/usr/bin/env bash
# Gemma-4-E4B label construction: generation + 3-prompt judge voting over all
# datasets used by the probe pipeline (incl. rewritten-context augmentation).
# Resumable: run_eval.py skips completed example ids; this driver makes up to
# MAX_ROUNDS passes and only emits GEMMA_LABELGEN_DONE when every output file
# has as many rows as its source.
set -u
cd "${BEATRICE_ROOT:-.}"
PY=python
CFG_DIR=configs/experiment/generation/gemma4
LOG_DIR=logs/gemma4
mkdir -p "$LOG_DIR"
MAX_ROUNDS=3

# canary-first ordering: small sets surface systemic issues before the big burns
NAMES=(popqa nq triviaqa hotpotqa_supporting_only twowiki_supporting_only musique rewritten_context)

count_lines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }

complete() {
  local name=$1
  local src="data/gemma4/sources/${name}.jsonl"
  local out="data/generations/${name}_gemma4e4b_results.jsonl"
  local need have
  need=$(count_lines "$src")
  have=$(count_lines "$out")
  [ "$need" -gt 0 ] && [ "$have" -ge "$need" ]
}

for round in $(seq 1 $MAX_ROUNDS); do
  pending=0
  for name in "${NAMES[@]}"; do
    if complete "$name"; then
      echo "[driver] round $round: $name already complete"
      continue
    fi
    pending=1
    echo "[driver] round $round: running $name ($(date))"
    "$PY" scripts/run_eval.py --config "$CFG_DIR/${name}_gemma4e4b_eval.yaml" \
      >> "$LOG_DIR/labelgen_${name}.log" 2>&1
    rc=$?
    echo "[driver] round $round: $name exited rc=$rc, rows=$(count_lines data/generations/${name}_gemma4e4b_results.jsonl)/$(count_lines data/gemma4/sources/${name}.jsonl)"
  done
  [ "$pending" -eq 0 ] && break
done

fail=0
for name in "${NAMES[@]}"; do
  if ! complete "$name"; then
    echo "[driver] INCOMPLETE: $name rows=$(count_lines data/generations/${name}_gemma4e4b_results.jsonl)/$(count_lines data/gemma4/sources/${name}.jsonl)"
    fail=1
  fi
done

if [ "$fail" -eq 0 ]; then
  echo "GEMMA_LABELGEN_DONE $(date)"
else
  echo "GEMMA_LABELGEN_FAILED $(date)"
fi
