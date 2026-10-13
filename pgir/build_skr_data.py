"""Build SKR-knn training data from probe-training judge labels (M5, spec §5).

SKR judges "does retrieval help this question?" via kNN over labeled training
questions. Labels here come from the probe project's judge-consensus runs:
  ir_better: with_context correct AND without_context incorrect
  ir_worse : without_context correct AND with_context incorrect
(ties carry no signal and are dropped, as in the SKR setup)

Caveat recorded for the paper: with_context labels use ORACLE (supporting/
rewritten) contexts, not retrieved ones — declared in RESULTS notes.

Only canonical TRAIN scope ids are used (dev/test untouched).
Run: cd <package_root> && python -m pgir.build_skr_data
"""
from __future__ import annotations

import json
from pathlib import Path

LING = Path("beatrice")
META_DIR = LING / "data/features_judge_consensus/qwen3_5_9b_question"
METAS = ["hotpotqa_supporting_only_question_features_meta.jsonl",
         "twowiki_supporting_only_question_features_meta.jsonl",
         "musique_question_features_meta.jsonl"]
OUT = Path(__file__).resolve().parent / "assets" / "skr_training_data.json"


def main():
    scope = json.loads((LING / "configs/sampling/fixed_scope_ids.json").read_text())
    train_ids = set(scope["ids"]["train"])
    rows, counts = [], {"ir_better": 0, "ir_worse": 0, "tie": 0, "skipped": 0}
    for m in METAS:
        with open(META_DIR / m) as f:
            for line in f:
                r = json.loads(line)
                ex = r.get("example") or {}
                eid = str(ex.get("id"))
                if eid not in train_ids:
                    continue
                runs = r.get("runs") or {}
                wo = (runs.get("without_context") or {}).get("is_correct")
                wi = (runs.get("with_context") or {}).get("is_correct")
                if wo is None or wi is None:
                    counts["skipped"] += 1
                    continue
                if wi and not wo:
                    rows.append({"question": ex["question"], "judgement": "ir_better"})
                    counts["ir_better"] += 1
                elif wo and not wi:
                    rows.append({"question": ex["question"], "judgement": "ir_worse"})
                    counts["ir_worse"] += 1
                else:
                    counts["tie"] += 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, ensure_ascii=False))
    print(json.dumps(counts), "->", OUT, f"({len(rows)} rows)")


if __name__ == "__main__":
    main()
