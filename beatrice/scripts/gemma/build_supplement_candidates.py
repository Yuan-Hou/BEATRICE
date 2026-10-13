"""Sample supplement candidates so the Gemma unanimous-consensus label set can
be refilled to exactly the Qwen-era sizes (per dataset x split cell).

Failed ids cannot be rescued (generation, judging and rewriting are all
temperature-0 deterministic), so deficits are filled with NEW samples drawn
from the same pools with the same builders:
  - hotpotqa/musique/twowiki: HF pools via the registry builders (deterministic
    ids), excluding every id in the original scope.
  - popqa: unused HF row indices, contexts fetched live from Wikipedia
    (PopQABuilder methodology).
  - nq: local spares first (rows of data/processed/nq_validation.jsonl that are
    not in the used 1182), then fresh builder rows idx>=1300 as backup.
  - triviaqa: local spares (23) + fresh stream extension beyond the first 1300.

Outputs:
  data/gemma4/sources_supplement/<name>.jsonl        candidate examples
  data/gemma4/sources_supplement/manifest.json       ordered candidate ids per
      (dataset, hf_split) — selection order for the finalize step
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ragprobe.datasets.hotpotqa import HotpotQABuilder  # noqa: E402
from ragprobe.datasets.musique import MusiqueBuilder  # noqa: E402
from ragprobe.datasets.twowiki import TwoWikiMultiHopQABuilder  # noqa: E402
from ragprobe.datasets.nq import NQBuilder  # noqa: E402
from ragprobe.datasets.triviaqa import TriviaQABuilder  # noqa: E402
from ragprobe.datasets.popqa import PopQABuilder  # noqa: E402

SEED = 20260713
BUFFER = 1.5
OUT_DIR = ROOT / "data" / "gemma4" / "sources_supplement"
LABELS_DIR = ROOT / "data" / "gemma4" / "labels"


def load_label_ids(name: str) -> set[str]:
    p = LABELS_DIR / f"{name}_labels.jsonl"
    return {json.loads(l)["id"] for l in p.open()}


def compute_deficits():
    scope = json.load((ROOT / "configs/sampling/fixed_scope_ids.json").open())["ids"]
    base = set()
    for n in ("hotpotqa_supporting_only", "musique", "twowiki_supporting_only", "popqa"):
        base |= load_label_ids(n)
    rew = load_label_ids("rewritten_context")
    keep = base & rew  # triple unanimous (closed-book+withctx unanimous AND rewritten unanimous)
    deficits = {}
    for grp, ids in scope.items():
        for i in ids:
            parts = i.split("-")
            ds = parts[0]
            split = parts[1] if ds in ("musique", "popqa", "nq", "triviaqa") else parts[2]
            key = (ds, split, grp)
            deficits.setdefault(key, [0, 0])
            deficits[key][0] += 1
            if i not in keep:
                deficits[key][1] += 1
    return deficits, scope


def n_buf(deficit: int) -> int:
    return max(int(math.ceil(deficit * BUFFER)), deficit + 15)


def pick(records, used_ids, want, rng):
    pool = [r for r in records if r["id"] not in used_ids and r.get("answers")]
    idx = list(range(len(pool)))
    rng.shuffle(idx)
    return [pool[i] for i in idx[:want]]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("*.jsonl"):
        old.unlink()  # idempotent rebuild: emit() appends within one run
    deficits, scope = compute_deficits()
    used = {i for ids in scope.values() for i in ids}
    print("cell deficits:")
    for (ds, split, grp), (tot, d) in sorted(deficits.items()):
        if d:
            print(f"  {ds}/{split} [{grp}] {d}/{tot}")

    manifest = {"seed": SEED, "buffer": BUFFER, "candidates": {}}

    def want_for(ds, split):
        # candidates from one hf split may serve several scope groups
        need = sum(d for (a, b, _), (_, d) in deficits.items() if a == ds and b == split)
        return need, n_buf(need) if need else 0

    def emit(name, records, key):
        path = OUT_DIR / f"{name}.jsonl"
        mode = "a" if path.exists() else "w"
        with path.open(mode) as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        manifest["candidates"][key] = [r["id"] for r in records]
        print(f"{key}: {len(records)} candidates -> {path}")

    rng = random.Random(SEED)

    # ---- hotpotqa ----
    need_t, buf_t = want_for("hotpotqa", "train")
    need_v, buf_v = want_for("hotpotqa", "validation")
    if buf_t or buf_v:
        recs = HotpotQABuilder(context_mode="supporting_only").build()
        by_split = {"train": [], "validation": []}
        for r in recs:
            by_split.setdefault(r["split"], []).append(r)
        if buf_t:
            emit("hotpotqa_supporting_only", pick(by_split["train"], used, buf_t, rng), "hotpotqa/train")
        if buf_v:
            emit("hotpotqa_supporting_only", pick(by_split["validation"], used, buf_v, rng), "hotpotqa/validation")

    # ---- musique ----
    need_t, buf_t = want_for("musique", "train")
    need_v, buf_v = want_for("musique", "validation")
    if buf_t or buf_v:
        recs = MusiqueBuilder(context_mode="supporting_only").build()
        by_split = {"train": [], "validation": []}
        for r in recs:
            by_split.setdefault(r["split"], []).append(r)
        if buf_t:
            emit("musique", pick(by_split["train"], used, buf_t, rng), "musique/train")
        if buf_v:
            emit("musique", pick(by_split["validation"], used, buf_v, rng), "musique/validation")

    # ---- twowiki ----
    need_t, buf_t = want_for("twowiki", "train")
    need_v, buf_v = want_for("twowiki", "validation")
    if buf_t or buf_v:
        recs = TwoWikiMultiHopQABuilder(context_mode="supporting_only",
                                        include_splits=("train", "validation")).build()
        by_split = {"train": [], "validation": []}
        for r in recs:
            by_split.setdefault(r["split"], []).append(r)
        if buf_t:
            emit("twowiki_supporting_only", pick(by_split["train"], used, buf_t, rng), "twowiki/train")
        if buf_v:
            emit("twowiki_supporting_only", pick(by_split["validation"], used, buf_v, rng), "twowiki/validation")

    # ---- popqa (live wikipedia contexts for unused HF row indices) ----
    need_p, buf_p = want_for("popqa", "test")
    if buf_p:
        from datasets import load_dataset
        n_rows = len(load_dataset("akariasai/PopQA", split="test"))
        cand_idx = [i for i in range(n_rows) if f"popqa-test-{i:06d}" not in used]
        rng.shuffle(cand_idx)
        cand_idx = cand_idx[: buf_p * 2]  # extra headroom for wiki fetch failures
        scope_tmp = OUT_DIR / "_popqa_candidate_scope.json"
        scope_tmp.write_text(json.dumps({"ids": [f"popqa-test-{i:06d}" for i in cand_idx]}))
        recs = PopQABuilder(scope_ids_path=str(scope_tmp)).build()
        by_id = {r["id"]: r for r in recs}
        ordered = [by_id[f"popqa-test-{i:06d}"] for i in cand_idx if f"popqa-test-{i:06d}" in by_id]
        emit("popqa", ordered[:buf_p], "popqa/test")

    # ---- nq (local spares first, fresh idx>=1300 backup) ----
    used_nq = load_label_ids("nq")
    target_nq = 1182
    need_nq = target_nq - len(used_nq)
    if need_nq > 0:
        local = [json.loads(l) for l in (ROOT / "data/processed/nq_validation.jsonl").open()]
        spares = [r for r in local if r["id"] not in used_nq and r.get("answers")]
        rng.shuffle(spares)
        cands = spares
        if len(spares) < n_buf(need_nq):
            fresh_n = n_buf(need_nq) - len(spares)
            recs = NQBuilder(max_examples=1300 + fresh_n * 2, num_workers=16).build()
            fresh = [r for r in recs if int(r["id"].rsplit("-", 1)[1]) >= 1300][: fresh_n]
            cands = spares + fresh
        emit("nq", cands[: n_buf(need_nq)], "nq/validation")

    # ---- triviaqa (local spares + fresh stream extension) ----
    used_tqa = load_label_ids("triviaqa")
    target_tqa = 1277
    need_tqa = target_tqa - len(used_tqa)
    if need_tqa > 0:
        local = [json.loads(l) for l in (ROOT / "data/processed/triviaqa_validation.jsonl").open()]
        local_ids = {r["id"] for r in local}
        spares = [r for r in local if r["id"] not in used_tqa and r.get("answers")]
        rng.shuffle(spares)
        cands = list(spares)
        if len(cands) < n_buf(need_tqa):
            fresh_n = n_buf(need_tqa) - len(cands)
            recs = TriviaQABuilder(max_examples=1300 + fresh_n * 3).build()
            fresh = [r for r in recs if r["id"] not in local_ids and r["id"] not in used_tqa][: fresh_n]
            cands += fresh
        emit("triviaqa", cands[: n_buf(need_tqa)], "triviaqa/validation")

    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"manifest -> {OUT_DIR}/manifest.json")


if __name__ == "__main__":
    main()
