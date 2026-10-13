#!/usr/bin/env python
"""Exhaustive k-subset search over cached frozen-branch vectors.

Trains the standard frozen-fusion head on EVERY C(15,k) branch subset (same
protocol as search_fusion_combos: 3 seeds, val-selected early stop), then reports
where the greedy-selected champion ranks in the full subset distribution — i.e.
"greedy ~= global optimum at a fraction of the cost", plus the random-selection
baseline (the distribution itself).

All subsets share the same frozen encoder checkpoints, so encoder-training noise
is a common constant across rows; head seeds cover the trainable part.

Usage:
  python scripts/exhaustive_k_search.py --cache outputs/cached_fusion_vecs/ctx_15br \
      --side context --k 2 --seeds 3 --eval-splits test,popqa,nq,triviaqa \
      --champion hid_prob_attn,hid_prob_resid --out outputs/reports/exhaustive_k2_ctx
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from search_fusion_combos import eval_subset, load_split  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--side", choices=("context", "question"), required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--eval-splits", default="test,popqa")
    ap.add_argument("--champion", default="", help="comma-separated greedy champion subset to rank")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_splits = [x for x in args.eval_splits.split(",") if x]
    cache = Path(args.cache)
    splits = {s: load_split(cache, s, device) for s in ["train", "validation"] + eval_splits}
    available = sorted(splits["train"]["vecs"].keys())
    seeds = list(range(args.seeds))
    champion = frozenset(x for x in args.champion.split(",") if x)

    combos = list(itertools.combinations(available, args.k))
    print(f"[exhaustive] side={args.side} k={args.k} branches={len(available)} "
          f"subsets={len(combos)} seeds={args.seeds}", flush=True)

    rows = []
    t0 = time.time()
    for i, subset in enumerate(combos, 1):
        r = eval_subset(list(subset), splits, args.side, device, seeds, eval_splits=eval_splits)
        rows.append({"branches": list(subset), **r})
        if i % 10 == 0 or i == len(combos):
            el = time.time() - t0
            print(f"  {i}/{len(combos)} done ({el:.0f}s, ~{el/i:.1f}s/subset)", flush=True)

    rows.sort(key=lambda r: -r["val"])
    vals = np.array([r["val"] for r in rows])
    champ_rank, champ_row = None, None
    for rank, r in enumerate(rows, 1):
        if frozenset(r["branches"]) == champion:
            champ_rank, champ_row = rank, r
            break

    summary = {
        "side": args.side, "k": args.k, "n_subsets": len(rows), "seeds": args.seeds,
        "champion": sorted(champion), "champion_rank": champ_rank,
        "champion_val": champ_row["val"] if champ_row else None,
        "best_subset": rows[0]["branches"], "best_val": rows[0]["val"],
        "gap_champion_to_best": (rows[0]["val"] - champ_row["val"]) if champ_row else None,
        "random_selection_val": {"mean": float(vals.mean()), "std": float(vals.std()),
                                 "median": float(np.median(vals)),
                                 "p10": float(np.quantile(vals, .1)), "p90": float(np.quantile(vals, .9))},
        "n_within_5e4_of_best": int((vals >= rows[0]["val"] - 5e-4).sum()),
        "n_within_2e3_of_best": int((vals >= rows[0]["val"] - 2e-3).sum()),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "rows": rows}, open(f"{out}.json", "w"), indent=1)

    hdr = " | ".join(eval_splits)
    lines = [f"# Exhaustive k={args.k} subset search ({args.side}, seeds={args.seeds}, "
             f"{len(rows)} subsets)", "",
             f"- champion (greedy): `{sorted(champion)}` -> val={summary['champion_val']}, "
             f"**rank {champ_rank}/{len(rows)}**, gap to best = {summary['gap_champion_to_best']}",
             f"- random-selection val: mean={vals.mean():.4f} +- {vals.std():.4f} "
             f"(median {np.median(vals):.4f}, p90 {np.quantile(vals,.9):.4f})",
             f"- subsets within 5e-4 / 2e-3 of best: {summary['n_within_5e4_of_best']} / "
             f"{summary['n_within_2e3_of_best']}", "",
             f"| rank | subset | val | {hdr} |", "|---|---|---|" + "---|" * len(eval_splits)]
    for rank, r in enumerate(rows[:10], 1):
        star = " **<- greedy**" if frozenset(r["branches"]) == champion else ""
        cells = " | ".join(f"{r[s]:.4f}" for s in eval_splits)
        lines.append(f"| {rank} | {'+'.join(r['branches'])}{star} | {r['val']:.4f} | {cells} |")
    if champ_rank and champ_rank > 10:
        r = champ_row
        cells = " | ".join(f"{r[s]:.4f}" for s in eval_splits)
        lines.append(f"| {champ_rank} | {'+'.join(r['branches'])} **<- greedy** | {r['val']:.4f} | {cells} |")
    Path(f"{out}.md").write_text("\n".join(lines) + "\n")
    print(f"EXHAUSTIVE_DONE rank={champ_rank}/{len(rows)} -> {out}.md", flush=True)


if __name__ == "__main__":
    main()
