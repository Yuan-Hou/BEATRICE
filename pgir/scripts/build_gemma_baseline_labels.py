"""Gemma retrain stage 1: build SKR-devreal data + Adaptive A/B/C labels.

Mirrors the Qwen recipe with Gemma sources:
  - dev-real signal: stage-0 norag/vanilla dev runs (intermediate_data.json),
    EM via the project normalizer — replaces the Qwen PGIR dev traces.
  - silver 30k: data/gemma4/labels/{ds}_labels.jsonl (runs.{without,with}_context
    .is_correct, oracle contexts — same caveat as Qwen, declared) joined with
    sources_final questions, gated by gemma4_scope train ids.
Outputs:
  assets/skr_training_data_devreal_gemma.json      ({question, judgement} list)
  assets/adaptive_labels_gemma_{train,val}.jsonl   ({question, label})

Run: cd <package_root> && python -m pgir.scripts.build_gemma_baseline_labels
"""
from __future__ import annotations

import glob
import json
import random
from collections import Counter
from pathlib import Path

from pgir.tuning.trace import em

PGIR = Path(__file__).resolve().parent.parent
LING = Path("beatrice")
RUNS = PGIR / "runs" / "flashrag_gemma"
SCOPE = LING / "configs/sampling/gemma4_scope_ids.json"
DS3 = ("hotpotqa", "twowiki", "musique")
LABEL_FILES = {"hotpotqa": "hotpotqa_supporting_only_labels.jsonl",
               "twowiki": "twowiki_supporting_only_labels.jsonl",
               "musique": "musique_labels.jsonl"}
SOURCE_FILES = {"hotpotqa": "hotpotqa_supporting_only.jsonl",
                "twowiki": "twowiki_supporting_only.jsonl",
                "musique": "musique.jsonl"}


def load_dev_outcomes():
    """{question: (norag_em, vanilla_em)} from the stage-0 dev runs."""
    out = {}
    for ds in DS3:
        per_q = {}
        for m in ("norag", "vanilla"):
            f = sorted(glob.glob(str(RUNS / f"gemma4_{m}_{ds}_dev/*/intermediate_data.json")))[-1]
            for it in json.load(open(f)):
                e = em(it["output"]["pred"] or "", it["golden_answers"])
                per_q.setdefault(it["question"], {})[m] = bool(e)
        for q, r in per_q.items():
            if "norag" in r and "vanilla" in r:
                out[q] = (r["norag"], r["vanilla"])
    return out


def main():
    dev = load_dev_outcomes()

    # ---- SKR devreal ----
    skr_rows, sc = [], Counter()
    for q, (e0, e1) in dev.items():
        if e1 and not e0:
            skr_rows.append({"question": q, "judgement": "ir_better"}); sc["ir_better"] += 1
        elif e0 and not e1:
            skr_rows.append({"question": q, "judgement": "ir_worse"}); sc["ir_worse"] += 1
        else:
            sc["tie"] += 1
    (PGIR / "assets" / "skr_training_data_devreal_gemma.json").write_text(
        json.dumps(skr_rows, ensure_ascii=False))
    print("SKR devreal:", dict(sc), f"-> {len(skr_rows)} rows")

    # ---- Adaptive A/B/C ----
    train_ids = set(json.loads(SCOPE.read_text())["ids"]["train"])
    rows, counts = [], Counter()
    for ds in DS3:
        qs = {json.loads(l)["id"]: json.loads(l)["question"]
              for l in open(LING / "data/gemma4/sources_final" / SOURCE_FILES[ds])}
        for line in open(LING / "data/gemma4/labels" / LABEL_FILES[ds]):
            r = json.loads(line)
            if r["id"] not in train_ids or r["id"] not in qs:
                continue
            wo = r["runs"]["without_context"]["is_correct"]
            wi = r["runs"]["with_context"]["is_correct"]
            if wo is None or wi is None:
                continue
            label = "A" if wo else ("B" if wi else "C")
            rows.append({"question": qs[r["id"]], "label": label})
            counts[label] += 1
    dev_rows = []
    for q, (e0, e1) in dev.items():
        dev_rows.append({"question": q,
                         "label": "A" if e0 else ("B" if e1 else "C")})
    random.Random(17).shuffle(rows)
    val = rows[:1000]
    train = rows[1000:] + dev_rows * 3     # upweight real-retrieval labels (qwen recipe)
    random.Random(17).shuffle(train)
    for name, data in (("train", train), ("val", val)):
        with open(PGIR / "assets" / f"adaptive_labels_gemma_{name}.jsonl", "w") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("adaptive silver:", dict(counts), "| dev-real:",
          dict(Counter(r["label"] for r in dev_rows)),
          "| train:", len(train), "val:", len(val))
    print("STAGE1 LABELS DONE")


if __name__ == "__main__":
    main()
