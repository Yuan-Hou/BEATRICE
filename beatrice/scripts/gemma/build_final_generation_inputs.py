"""Merge main + supplement generation results and filter to the final Gemma scope.

Feature extraction consumes generation-result jsonl (record.example + runs).
The final scope (configs/sampling/gemma4_scope_ids.json) drops some original
ids and adds supplement ids, so extraction inputs must be the union filtered
to exactly the selected ids, in scope order.

Outputs: data/gemma4/generations_final/{name}_results.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "data" / "generations"
OUT = ROOT / "data" / "gemma4" / "generations_final"

IN_DOMAIN = ("hotpotqa_supporting_only", "musique", "twowiki_supporting_only", "popqa")


def load_results(*paths: Path) -> dict[str, dict]:
    out = {}
    for p in paths:
        if not p.exists():
            continue
        for line in p.open():
            r = json.loads(line)
            ex = r.get("example") or {}
            if ex.get("id"):
                out[ex["id"]] = r
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scope = json.load((ROOT / "configs/sampling/gemma4_scope_ids.json").open())
    all_ids = [i for g in ("train", "validation", "test") for i in scope["ids"][g]]

    def prefix_of(i: str) -> str:
        p = i.split("-")
        if p[0] == "musique":
            return "musique"
        if p[0] == "popqa":
            return "popqa"
        return {"hotpotqa": "hotpotqa_supporting_only", "twowiki": "twowiki_supporting_only"}[p[0]]

    for name in IN_DOMAIN:
        sel = [i for i in all_ids if prefix_of(i) == name]
        pool = load_results(GEN / f"{name}_gemma4e4b_results.jsonl",
                            GEN / f"supp_{name}_gemma4e4b_results.jsonl")
        missing = [i for i in sel if i not in pool]
        assert not missing, f"{name}: {len(missing)} scope ids missing from generations"
        with (OUT / f"{name}_results.jsonl").open("w") as f:
            for i in sel:
                f.write(json.dumps(pool[i], ensure_ascii=False) + "\n")
        print(f"{name}: {len(sel)}")

    for name in ("nq", "triviaqa"):
        sel = scope["extra_ids"][name]
        pool = load_results(GEN / f"{name}_gemma4e4b_results.jsonl",
                            GEN / f"supp_{name}_gemma4e4b_results.jsonl")
        missing = [i for i in sel if i not in pool]
        assert not missing, f"{name}: {len(missing)} missing"
        with (OUT / f"{name}_results.jsonl").open("w") as f:
            for i in sel:
                f.write(json.dumps(pool[i], ensure_ascii=False) + "\n")
        print(f"{name}: {len(sel)}")

    pool = load_results(GEN / "rewritten_context_gemma4e4b_results.jsonl",
                        GEN / "supp_rewritten_context_gemma4e4b_results.jsonl")
    missing = [i for i in all_ids if i not in pool]
    assert not missing, f"rewritten: {len(missing)} missing"
    with (OUT / "rewritten_context_results.jsonl").open("w") as f:
        for i in all_ids:
            f.write(json.dumps(pool[i], ensure_ascii=False) + "\n")
    print(f"rewritten_context: {len(all_ids)}")


if __name__ == "__main__":
    main()
