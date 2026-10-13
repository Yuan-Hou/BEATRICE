"""Extend nq/triviaqa supplement candidate pools.

Round-1 pools passed unanimity at only ~50%/~40% (judge split votes on
borderline answers vs terse gold lists — inherent to NQ/TriviaQA), leaving
shortfalls of 13 (nq) and 12 (triviaqa). Appends fresh deterministic
candidates sized for those pass rates and updates the manifest in place.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ragprobe.datasets.nq import NQBuilder  # noqa: E402
from ragprobe.datasets.triviaqa import TriviaQABuilder  # noqa: E402

SUPP = ROOT / "data" / "gemma4" / "sources_supplement"
SEED = 20260714

ADD = {"nq": 90, "triviaqa": 120}


def existing_ids(name: str) -> set[str]:
    ids = {json.loads(l)["id"] for l in (SUPP / f"{name}.jsonl").open()}
    ids |= {json.loads(l)["id"] for l in (ROOT / f"data/gemma4/sources/{name}.jsonl").open()}
    return ids


def main() -> None:
    manifest_path = SUPP / "manifest.json"
    manifest = json.load(manifest_path.open())
    rng = random.Random(SEED)

    # nq: leftover local spares first, then fresh idx>=1300 (live wiki fetch)
    used = existing_ids("nq")
    local = [json.loads(l) for l in (ROOT / "data/processed/nq_validation.jsonl").open()]
    spares = [r for r in local if r["id"] not in used and r.get("answers")]
    rng.shuffle(spares)
    new_nq = list(spares)
    fresh_needed = ADD["nq"] - len(new_nq)
    if fresh_needed > 0:
        recs = NQBuilder(max_examples=1300 + int(fresh_needed * 1.3), num_workers=16).build()
        fresh = [r for r in recs
                 if int(r["id"].rsplit("-", 1)[1]) >= 1300 and r["id"] not in used]
        new_nq += fresh[:fresh_needed]

    # triviaqa: extend the seeded stream well past the previous cut
    used = existing_ids("triviaqa")
    local_ids = {json.loads(l)["id"] for l in (ROOT / "data/processed/triviaqa_validation.jsonl").open()}
    recs = TriviaQABuilder(max_examples=1800).build()
    new_tqa = [r for r in recs
               if r["id"] not in local_ids and r["id"] not in used][: ADD["triviaqa"]]

    for name, rows in (("nq", new_nq), ("triviaqa", new_tqa)):
        with (SUPP / f"{name}.jsonl").open("a") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        manifest["candidates"][f"{name}/validation"].extend(r["id"] for r in rows)
        print(f"{name}: +{len(rows)} candidates (pool now {len(manifest['candidates'][f'{name}/validation'])})")

    manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
