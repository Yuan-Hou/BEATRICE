"""Build judge-consensus-purified feature metas for the new OOD sets (nq, triviaqa).

No subsampling (user decision 2026-07-05): keep EVERY example that passes
  1. unanimous 3-variant judge vote on BOTH runs (mirrors the original
     judge-consensus purification's min_agreement=1.0 + parse_ok), and
  2. successful feature extraction.

Writes purified metas next to the legacy ones:
  data/features_judge_consensus/qwen3_5_9b/{ds}_features_meta.jsonl           (context)
  data/features_judge_consensus/qwen3_5_9b_question/{ds}_question_features_meta.jsonl

Usage: python scripts/build_new_ood_meta.py
"""
from __future__ import annotations

import json
from pathlib import Path

DATASETS = ("nq", "triviaqa")


def unanimous(run: dict) -> bool:
    """All 3 judge variants parsed AND agree. NOTE: `parsed` holds the verdict itself
    (True=correct, False=incorrect, None=unparseable), not a parse-success flag."""
    jv = run.get("judge_vote") or {}
    variants = jv.get("variants") or []
    if not variants or run.get("judge_parse_failed"):
        return False
    verdicts = [v.get("parsed") for v in variants]
    if any(x is None for x in verdicts):
        return False
    return len(set(bool(x) for x in verdicts)) == 1


def consensus_ids(ds: str) -> set[str]:
    keep = set()
    total = 0
    for line in open(f"data/generations/{ds}_qwen3_5_9b_results.jsonl"):
        r = json.loads(line)
        total += 1
        runs = r.get("runs") or {}
        wo, wi = runs.get("without_context") or {}, runs.get("with_context") or {}
        if unanimous(wo) and unanimous(wi):
            keep.add(r["example"]["id"])
    print(f"[{ds}] consensus: {len(keep)}/{total} pass unanimous 3-variant vote on both runs")
    return keep


def purify(src: str, dst: str, ids: set[str]) -> int:
    kept = []
    for line in open(src):
        row = json.loads(line)
        ex_id = (row.get("example") or {}).get("id") or row.get("example_id")
        status = (row.get("feature_extraction") or {}).get("status")
        if ex_id in ids and status == "success":
            kept.append(line.rstrip("\n"))
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_text("\n".join(kept) + "\n")
    print(f"  {dst}: {len(kept)}")
    return len(kept)


def main():
    for ds in DATASETS:
        ids = consensus_ids(ds)
        purify(f"data/features/qwen3_5_9b/{ds}_features_meta.jsonl",
               f"data/features_judge_consensus/qwen3_5_9b/{ds}_features_meta.jsonl", ids)
        purify(f"data/features/qwen3_5_9b_question/{ds}_question_features_meta.jsonl",
               f"data/features_judge_consensus/qwen3_5_9b_question/{ds}_question_features_meta.jsonl", ids)
    print("NEW_OOD_META_DONE")


if __name__ == "__main__":
    main()
