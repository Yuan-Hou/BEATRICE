"""Random-sample null distribution for k-subset fusion search when exhaustive
is infeasible (joint k=5: C(30,5)=142,506 subsets ~ 14 GPU-days).

Samples N distinct k-subsets uniformly (champion always included), trains each
with the standard 3-head-seed protocol (same eval_subset as the exhaustive
scripts), and reports the champion's percentile within the sampled val
distribution plus the random-selection stats. Resumable: appends to <out>.rows.jsonl
and skips already-evaluated subsets on restart.

Usage:
  python scripts/sampled_k_search.py --cache outputs/aligned/cached_fusion_vecs/joint_30br \
      --side context --k 5 --n 4000 --champion quad,hidden,attention,hid_prob_attn,q_attention \
      --out outputs/aligned/reports/sampled_k5_joint
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from search_fusion_combos import eval_subset, load_split  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--side", choices=("context", "question"), required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--eval-splits", default="test,popqa,nq,triviaqa")
    ap.add_argument("--champion", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample-seed", type=int, default=0)
    args = ap.parse_args()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    es = [x for x in args.eval_splits.split(",") if x]
    splits = {s: load_split(Path(args.cache), s, device) for s in ["train", "validation"] + es}
    av = sorted(splits["train"]["vecs"].keys())
    champion = tuple(sorted(args.champion.split(",")))
    assert all(b in av for b in champion), f"champion branches not all available: {champion}"

    rng = random.Random(args.sample_seed)
    picked = {champion}
    while len(picked) < args.n:
        picked.add(tuple(sorted(rng.sample(av, args.k))))
    subsets = sorted(picked)
    rng.shuffle(subsets)

    rows_path = Path(f"{args.out}.rows.jsonl")
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if rows_path.exists():
        for line in open(rows_path):
            r = json.loads(line)
            done[tuple(sorted(r["branches"]))] = r
        print(f"[resume] {len(done)} subsets already evaluated", flush=True)

    seeds = list(range(args.seeds))
    t0 = time.time()
    with open(rows_path, "a") as fh:
        for i, sub in enumerate(subsets):
            if sub in done:
                continue
            r = eval_subset(list(sub), splits, args.side, device, seeds, eval_splits=tuple(es))
            r["branches"] = list(sub)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            done[sub] = r
            if len(done) % 100 == 0:
                el = time.time() - t0
                print(f"  {len(done)}/{len(subsets)} done ({el:.0f}s, ~{el/max(1,len(done)-0):.1f}s/subset avg-run)", flush=True)

    vals = np.array([done[s]["val"] for s in subsets])
    cv = done[champion]["val"]
    pct = float((vals < cv).mean()) * 100
    n_ge = int((vals >= cv).sum())
    summary = {
        "side": args.side, "k": args.k, "n_sampled": len(subsets), "seeds": args.seeds,
        "champion": list(champion), "champion_val": cv,
        "champion_percentile": pct, "n_sampled_ge_champion": n_ge,
        "random_selection_val": {"mean": float(vals.mean()), "std": float(vals.std()),
                                 "median": float(np.median(vals)), "p90": float(np.percentile(vals, 90)),
                                 "max": float(vals.max())},
    }
    json.dump({"summary": summary}, open(f"{args.out}.json", "w"), indent=1)
    order = sorted(subsets, key=lambda s: -done[s]["val"])
    lines = [f"# Sampled k={args.k} subset search ({args.side}, N={len(subsets)}, seeds={args.seeds})", "",
             f"- champion `{list(champion)}` val={cv:.4f} -> percentile {pct:.2f}% "
             f"({n_ge}/{len(subsets)} sampled subsets >= champion)",
             f"- random-selection val: mean={vals.mean():.4f} +- {vals.std():.4f} (max {vals.max():.4f})", "",
             "| rank | subset | val | " + " | ".join(es) + " |", "|---|---|---|" + "---|" * len(es)]
    for rank, s in enumerate(order[:15], 1):
        r = done[s]
        mark = " **<- champion**" if s == champion else ""
        lines.append(f"| {rank} | {'+'.join(s)}{mark} | {r['val']:.4f} | "
                     + " | ".join(f"{r[m]:.4f}" for m in es) + " |")
    Path(f"{args.out}.md").write_text("\n".join(lines) + "\n")
    print(f"SAMPLED_SEARCH_DONE percentile={pct:.2f}% -> {args.out}.md", flush=True)


if __name__ == "__main__":
    main()
