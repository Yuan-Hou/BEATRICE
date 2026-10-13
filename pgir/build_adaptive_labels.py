"""Build Adaptive-RAG query-complexity labels (A/B/C) for classifier training.

Official scheme (NAACL'24): silver-label queries by which system answers them —
A: no-retrieval answers correctly; B: single-step RAG answers correctly;
C: neither (needs iterative). Our sources:
  - 30k canonical TRAIN questions: judge labels from the probe project
    (A: without_context correct; B: with_context(ORACLE) correct & not A; C: rest)
    — oracle-context approximates single-step; declared in RESULTS notes.
  - 1500 dev-trace questions with REAL retrieval (r0/r1 EM): same rule, real
    single-step signal; these override/augment.
Output: assets/adaptive_labels_{train,val}.jsonl  ({"question", "label"}).

Run: cd <package_root> && python -m pgir.build_adaptive_labels
"""
from __future__ import annotations

import glob
import json
import random
from pathlib import Path

LING = Path("beatrice")
META_DIR = LING / "data/features_judge_consensus/qwen3_5_9b_question"
METAS = ["hotpotqa_supporting_only_question_features_meta.jsonl",
         "twowiki_supporting_only_question_features_meta.jsonl",
         "musique_question_features_meta.jsonl"]
OUT_DIR = Path(__file__).resolve().parent / "assets"


def main():
    scope = json.loads((LING / "configs/sampling/fixed_scope_ids.json").read_text())
    train_ids = set(scope["ids"]["train"])
    rows = []
    counts = {"A": 0, "B": 0, "C": 0}
    for m in METAS:
        with open(META_DIR / m) as f:
            for line in f:
                r = json.loads(line)
                ex = r.get("example") or {}
                if str(ex.get("id")) not in train_ids:
                    continue
                runs = r.get("runs") or {}
                wo = (runs.get("without_context") or {}).get("is_correct")
                wi = (runs.get("with_context") or {}).get("is_correct")
                if wo is None or wi is None:
                    continue
                label = "A" if wo else ("B" if wi else "C")
                rows.append({"question": ex["question"], "label": label})
                counts[label] += 1

    # real-retrieval dev labels (small but unbiased single-step signal)
    from pgir.tuning.trace import em, load_golds
    golds = load_golds(Path(__file__).resolve().parent / "datasets")
    dev_rows = []
    for ds in ("hotpotqa", "twowiki", "musique"):
        tf = sorted(glob.glob(str(Path("pgir/runs/flashrag") /
                    f"trace_{ds}_dev_full/**/pgir_trace_{ds}_dev.jsonl"), recursive=True))[-1]
        for line in open(tf):
            t = json.loads(line)
            g = golds[t["example_id"]]
            r0 = next((a for a in t["pool"] if a["round_index"] == 0 and a["status"] == "success"), None)
            r1 = next((a for a in t["pool"] if a["round_index"] == 1 and a["status"] == "success"), None)
            if not r0:
                continue
            if em(r0["answer"], g):
                label = "A"
            elif r1 is not None and em(r1["answer"], g):
                label = "B"
            else:
                label = "C"
            dev_rows.append({"question": t["question"], "label": label})

    random.Random(17).shuffle(rows)
    val = rows[:1000]                      # held-out silver val for the classifier
    train = rows[1000:] + dev_rows * 3     # upweight real-retrieval labels
    random.Random(17).shuffle(train)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, data in (("train", train), ("val", val)):
        with open(OUT_DIR / f"adaptive_labels_{name}.jsonl", "w") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print("silver counts:", counts, "| dev-real:", Counter(r["label"] for r in dev_rows),
          "| train:", len(train), "val:", len(val))


if __name__ == "__main__":
    main()
