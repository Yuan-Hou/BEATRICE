#!/usr/bin/env bash
# Supplement pass: refill the Gemma unanimous-label set to exact Qwen sizes.
# Assumes candidates already built (build_supplement_candidates.py).
# Steps: base gen+judge on candidates -> 27B context rewrite for base-unanimous
# ones -> gemma withctx gen+judge on rewrites -> finalize scope + verify.
set -u
cd "${BEATRICE_ROOT:-.}"
PY=python
CFG=configs/experiment/generation/gemma4
LOG=logs/gemma4
mkdir -p "$LOG"

NAMES=(popqa nq triviaqa hotpotqa_supporting_only twowiki_supporting_only musique)

count_lines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }

run_until_complete() {  # $1=config path  $2=src jsonl  $3=out jsonl  $4=log tag
  for round in 1 2 3; do
    local need have
    need=$(count_lines "$2"); have=$(count_lines "$3")
    [ "$have" -ge "$need" ] && return 0
    echo "[supp] round $round: $4 ($have/$need)"
    "$PY" scripts/run_eval.py --config "$1" >> "$LOG/supp_$4.log" 2>&1
  done
  have=$(count_lines "$3"); need=$(count_lines "$2")
  [ "$have" -ge "$need" ]
}

fail=0
for name in "${NAMES[@]}"; do
  run_until_complete "$CFG/supp_${name}_gemma4e4b_eval.yaml" \
    "data/gemma4/sources_supplement/${name}.jsonl" \
    "data/generations/supp_${name}_gemma4e4b_results.jsonl" "$name" || { echo "[supp] INCOMPLETE $name"; fail=1; }
done
[ "$fail" -ne 0 ] && { echo "GEMMA_SUPPLEMENT_FAILED base-gen $(date)"; exit 1; }

echo "[supp] rewrite pass"
"$PY" scripts/gemma/make_supplement_rewrites.py >> "$LOG/supp_rewrite.log" 2>&1 || { echo "GEMMA_SUPPLEMENT_FAILED rewrite $(date)"; exit 1; }

run_until_complete "$CFG/supp_rewritten_context_gemma4e4b_eval.yaml" \
  "data/gemma4/sources_supplement/rewritten_supplement.jsonl" \
  "data/generations/supp_rewritten_context_gemma4e4b_results.jsonl" "rewritten" || { echo "GEMMA_SUPPLEMENT_FAILED rewritten-gen $(date)"; exit 1; }

echo "[supp] finalize"
"$PY" scripts/gemma/finalize_gemma_scope.py >> "$LOG/supp_finalize.log" 2>&1
rc=$?
tail -20 "$LOG/supp_finalize.log"
if [ "$rc" -eq 0 ]; then
  echo "GEMMA_SUPPLEMENT_DONE $(date)"
else
  echo "GEMMA_SUPPLEMENT_FAILED finalize $(date)"
fi
