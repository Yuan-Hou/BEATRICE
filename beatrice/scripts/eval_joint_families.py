#!/usr/bin/env python
"""Evaluate joint-pool (ctx + closed-book) family subsets for the ablation master table.

Families (all under the with_context_correct label, joint_30br cache):
  all30       - every branch from both conditionings ("use everything")
  q15_all     - all 15 closed-book branches (conditioning-LOO: no ctx)
  singles8    - 4 ctx + 4 q single-channel encoders (pure late fusion, both conditionings)
  quad2       - quad + q_quad (pure early fusion per conditioning, late across)
  hand_dual5  - ctx champion 2 + qonly champion 3 (the hand-designed dual head set)

Usage: python scripts/eval_joint_families.py --out outputs/reports/joint_families
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from search_fusion_combos import eval_subset, load_split  # noqa: E402

FAMILIES = {
    "q15_all": lambda av: [b for b in av if b.startswith("q_")],
    "singles8": lambda av: ["hidden", "probability", "residual", "attention",
                            "q_hidden", "q_probability", "q_residual", "q_attention"],
    "quad2": lambda av: ["quad", "q_quad"],
    "hand_dual5": lambda av: ["hid_prob_attn", "hid_prob_resid",
                              "q_hid_resid", "q_hid_prob_attn", "q_hid_attn"],
    "all30": lambda av: list(av),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="outputs/cached_fusion_vecs/joint_30br")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--eval-splits", default="test,popqa,nq,triviaqa")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_splits = [x for x in args.eval_splits.split(",") if x]
    cache = Path(args.cache)
    splits = {s: load_split(cache, s, device) for s in ["train", "validation"] + eval_splits}
    available = sorted(splits["train"]["vecs"].keys())
    seeds = list(range(args.seeds))

    results = {}
    for name, fn in FAMILIES.items():
        subset = [b for b in fn(available) if b in available]
        r = eval_subset(subset, splits, "context", device, seeds, eval_splits=eval_splits)
        r.update({"branches": subset, "k": len(subset)})
        results[name] = r
        extra = " ".join(f"{k}={r[k]:.4f}" for k in eval_splits)
        print(f"  {name:12s} k={len(subset):2d} val={r['val']:.4f} {extra}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(f"{out}.json", "w"), indent=1)
    print(f"JOINT_FAMILIES_DONE -> {out}.json", flush=True)


if __name__ == "__main__":
    main()
