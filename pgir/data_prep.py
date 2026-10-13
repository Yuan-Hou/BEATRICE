"""Build FlashRAG-format datasets from the project's canonical split.

The main-table eval runs on the probe-training canonical split (user decision
2026-07-04), NOT FlashRAG's native splits: the original datasets were carved
into probe train/val, so native dev/test would contaminate. Sources:

  - scope ids:   configs/sampling/fixed_scope_ids.json  (authoritative)
  - examples:    outputs/predictions/frozen_fusion_context_{validation,test,popqa}_
                 ctxL17rd_attn_preds.jsonl  (source_meta.example embeds
                 question + golden answers; 2 augmentation variants per id -> dedup)

Output: pgir/datasets/<dataset>/{dev,test}.jsonl in FlashRAG
format {"id", "question", "golden_answers"} — dev = canonical validation
(tuning + calibration only), test = canonical test (popqa: OOD test only).

Run: cd <package_root> && python -m pgir.data_prep
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

LING = Path("beatrice")
SCOPE = LING / "configs/sampling/fixed_scope_ids.json"
PREDS = {
    "validation": LING / "outputs/predictions/frozen_fusion_context_validation_ctxL17rd_attn_preds.jsonl",
    "test": LING / "outputs/predictions/frozen_fusion_context_test_ctxL17rd_attn_preds.jsonl",
    "popqa": LING / "outputs/predictions/frozen_fusion_context_popqa_ctxL17rd_attn_preds.jsonl",
}
OUT = Path(__file__).resolve().parent / "datasets"

# canonical dataset name -> FlashRAG dataset dir name (keep ours, they're not
# FlashRAG's native splits and must not be confused with them)
DATASETS = ("hotpotqa", "twowiki", "musique", "popqa")


def _dataset_of(example_id: str) -> str:
    return example_id.split("-", 1)[0]


def build() -> dict:
    scope = json.loads(SCOPE.read_text())
    groups = {"dev": set(scope["ids"]["validation"]), "test": set(scope["ids"]["test"])}

    rows: dict[tuple[str, str], dict] = {}
    for src, path in PREDS.items():
        grp = "dev" if src == "validation" else "test"
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                eid = r["example_id"]
                if eid not in groups[grp] or (grp, eid) in rows:
                    continue
                ex = (r.get("source_meta") or {}).get("example") or {}
                answers = ex.get("answers") or []
                if not ex.get("question") or not answers:
                    raise ValueError(f"{eid}: missing question/answers in {path}")
                rows[(grp, eid)] = {
                    "id": eid,
                    "question": str(ex["question"]).strip(),
                    "golden_answers": [str(a) for a in answers],
                }

    counts = defaultdict(int)
    buckets: dict[tuple[str, str], list] = defaultdict(list)
    for (grp, eid), row in rows.items():
        ds = _dataset_of(eid)
        buckets[(ds, grp)].append(row)
        counts[f"{ds}/{grp}"] += 1

    for (ds, grp), items in sorted(buckets.items()):
        items.sort(key=lambda r: r["id"])
        out = OUT / ds / f"{grp}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for row in items:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    expected = {"hotpotqa/dev": 500, "twowiki/dev": 500, "musique/dev": 500,
                "hotpotqa/test": 500, "twowiki/test": 500, "musique/test": 500,
                "popqa/test": 1000}
    assert dict(counts) == expected, f"split counts off: {dict(counts)} != {expected}"
    return dict(counts)


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
    print(f"datasets written under {OUT}")
