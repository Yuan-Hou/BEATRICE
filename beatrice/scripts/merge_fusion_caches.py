"""Merge two frozen-fusion vec caches (same splits, same example order) into one.

Used to extend the ctx 10-branch cache (singles+pairs) with the 5 triple/quad branches
without re-running the expensive 10-branch precompute. Verifies example_ids match
exactly per split before merging vecs/logits dicts.

Usage:
  python scripts/merge_fusion_caches.py --base <dir> --extra <dir> --out <dir> \
      [--splits train,validation,test,popqa]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--extra", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="train,validation,test,popqa")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for split in [s for s in args.splits.split(",") if s]:
        a = torch.load(Path(args.base) / f"{split}.pt", map_location="cpu", weights_only=False)
        b = torch.load(Path(args.extra) / f"{split}.pt", map_location="cpu", weights_only=False)
        if list(a["example_ids"]) != list(b["example_ids"]):
            raise RuntimeError(f"{split}: example_ids mismatch (base N={len(a['example_ids'])}, extra N={len(b['example_ids'])})")
        overlap = set(a["vecs"]) & set(b["vecs"])
        if overlap:
            raise RuntimeError(f"{split}: branch overlap {overlap}")
        merged = dict(a)
        merged["vecs"] = {**a["vecs"], **b["vecs"]}
        merged["logits"] = {**a["logits"], **b["logits"]}
        merged["modality_names"] = list(a["modality_names"]) + list(b["modality_names"])
        torch.save(merged, out / f"{split}.pt")
        print(f"{split}: {len(a['vecs'])}+{len(b['vecs'])} branches, N={len(a['example_ids'])} -> {out/f'{split}.pt'}")
    print("MERGE_DONE")


if __name__ == "__main__":
    main()
