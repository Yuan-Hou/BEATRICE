"""Build the joint 30-branch fusion-vec cache: ctx branches + q_-prefixed
closed-book branches, aligned by example_id.

The ctx cache's train split carries base + rewritten-augmentation variants of the
same eid; each variant reuses the eid's single question-side vector (the question
is unchanged by context rewriting). Eval splits align 1:1. Every ctx row must
find its eid in the question cache — hard error otherwise.

Usage:
  python scripts/build_joint_cache.py \
      --ctx outputs/aligned/cached_fusion_vecs/context_all15 \
      --q outputs/aligned/cached_fusion_vecs/question_all15 \
      --out outputs/aligned/cached_fusion_vecs/joint_30br \
      --splits train,validation,test,popqa,nq,triviaqa
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", required=True)
    ap.add_argument("--q", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="train,validation,test,popqa,nq,triviaqa")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for split in [s for s in args.splits.split(",") if s]:
        c = torch.load(Path(args.ctx) / f"{split}.pt", map_location="cpu", weights_only=False)
        q = torch.load(Path(args.q) / f"{split}.pt", map_location="cpu", weights_only=False)
        q_row = {eid: i for i, eid in enumerate(q["example_ids"])}
        idx = []
        missing = 0
        for eid in c["example_ids"]:
            j = q_row.get(eid)
            if j is None:
                missing += 1
            else:
                idx.append(j)
        if missing:
            raise RuntimeError(f"{split}: {missing}/{len(c['example_ids'])} ctx eids missing from question cache")
        gather = torch.as_tensor(idx, dtype=torch.long)
        merged = dict(c)
        merged["vecs"] = dict(c["vecs"])
        merged["logits"] = dict(c["logits"])
        ctx_dim = next(iter(c["vecs"].values())).shape[1]
        for name in q["vecs"]:
            v = q["vecs"][name][gather]
            if v.shape[1] < ctx_dim:
                # old-protocol convention (docs/fusion-ablations.md): q-only vecs
                # zero-padded to the ctx encoder dim so one fusion head fits all
                v = torch.nn.functional.pad(v, (0, ctx_dim - v.shape[1]))
            merged["vecs"][f"q_{name}"] = v
            merged["logits"][f"q_{name}"] = q["logits"][name][gather]
        merged["modality_names"] = list(c["modality_names"]) + [f"q_{n}" for n in q["modality_names"]]
        torch.save(merged, out / f"{split}.pt")
        dup = len(c["example_ids"]) - len(set(c["example_ids"]))
        print(f"{split}: ctx N={len(c['example_ids'])} (dup eids {dup}) + q {len(q['vecs'])} branches -> {out/f'{split}.pt'}", flush=True)
    print("JOINT_CACHE_DONE", flush=True)


if __name__ == "__main__":
    main()
