"""Rebuild generation source datasets for the Gemma migration.

The authoritative example sets (exact eids used by the Qwen probe pipeline)
live inside data/features_judge_consensus/*_features_meta.jsonl; several
data/processed source files are damaged (0-byte hotpotqa_supporting, 168-row
popqa). This script extracts the embedded `example` objects into clean jsonl
sources under data/gemma4/sources/ for the Gemma label-construction runs.

The rewritten-context meta already embeds the rewritten context as
example.contexts (same eid as the base example), so it is exported the same
way and evaluated with_context-only.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
META_DIR = ROOT / "data" / "features_judge_consensus"
OUT_DIR = ROOT / "data" / "gemma4" / "sources"

SOURCES = {
    "hotpotqa_supporting_only": META_DIR / "qwen3_5_9b" / "hotpotqa_supporting_only_features_meta.jsonl",
    "musique": META_DIR / "qwen3_5_9b" / "musique_features_meta.jsonl",
    "twowiki_supporting_only": META_DIR / "qwen3_5_9b" / "twowiki_supporting_only_features_meta.jsonl",
    "popqa": META_DIR / "qwen3_5_9b" / "popqa_features_meta.jsonl",
    "nq": META_DIR / "qwen3_5_9b" / "nq_features_meta.jsonl",
    "triviaqa": META_DIR / "qwen3_5_9b" / "triviaqa_features_meta.jsonl",
    "rewritten_context": META_DIR / "qwen3_5_9b_rewritten_context" / "rewritten_context_cqa_features_meta.jsonl",
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, meta_path in SOURCES.items():
        out_path = OUT_DIR / f"{name}.jsonl"
        seen: set[str] = set()
        examples = []
        with meta_path.open() as f:
            for line in f:
                rec = json.loads(line)
                ex = rec.get("example")
                if not ex or not ex.get("id"):
                    continue
                if ex["id"] in seen:
                    continue
                seen.add(ex["id"])
                if not ex.get("question") or not ex.get("answers"):
                    continue
                if name == "rewritten_context":
                    ctxs = ex.get("contexts") or []
                    if not ctxs or not (ctxs[0].get("text") or "").strip():
                        continue
                examples.append(ex)
        with out_path.open("w") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        prefixes = Counter("-".join(ex["id"].split("-")[:2]) for ex in examples)
        print(f"{name}: {len(examples)} examples -> {out_path}")
        for k, v in sorted(prefixes.items()):
            print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
