"""Sharded, resumable exhaustive k-subset search (for pools too big for the
single-process exhaustive_k_search, e.g. joint k=4: C(30,4)=27,405 ~ 53 GPU-h).

Each shard enumerates the full sorted combination list and takes slice
[shard::nshards]; rows append to <out>_shard<i>.rows.jsonl (resume = skip done).
After all shards finish, run with --merge to compute the champion's global rank
and write <out>.{json,md}.

Usage (per shard):
  python scripts/exhaustive_sharded.py --cache ... --side context --k 4 \
      --shard 0 --nshards 4 --champion quad,hidden,attention,hid_attn \
      --out outputs/aligned/reports/exhaustive_k4_joint
Merge:
  python scripts/exhaustive_sharded.py --cache ... --side context --k 4 \
      --nshards 4 --champion ... --out ... --merge
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from search_fusion_combos import eval_subset, load_split  # noqa: E402


def all_combos(av, k):
    return list(itertools.combinations(sorted(av), k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--side", choices=("context", "question"), required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--eval-splits", default="test,popqa,nq,triviaqa")
    ap.add_argument("--champion", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    es = [x for x in args.eval_splits.split(",") if x]
    champion = tuple(sorted(args.champion.split(",")))

    if args.merge:
        rows = {}
        for i in range(args.nshards):
            p = Path(f"{args.out}_shard{i}.rows.jsonl")
            for line in open(p):
                r = json.loads(line)
                rows[tuple(sorted(r["branches"]))] = r
        import torch  # noqa: F401  (parity with shard path imports)
        # completeness check against the full enumeration
        splits = None
        from search_fusion_combos import load_split as _ls
        av = None
        # derive pool from any row width? enumerate needs branch list: read one shard's branches union
        pool = sorted({b for s in rows for b in s})
        combos = all_combos(pool, args.k)
        missing = [c for c in combos if c not in rows]
        if missing:
            print(f"MERGE_INCOMPLETE: {len(missing)} subsets missing (e.g. {missing[:2]})", flush=True)
            sys.exit(1)
        vals = np.array([rows[c]["val"] for c in combos])
        cv = rows[champion]["val"]
        rank = int((vals > cv).sum()) + 1
        order = sorted(combos, key=lambda c: -rows[c]["val"])
        summary = {"side": args.side, "k": args.k, "n_subsets": len(combos), "seeds": args.seeds,
                   "champion": list(champion), "champion_val": cv, "champion_rank": rank,
                   "gap_to_best": float(vals.max() - cv),
                   "random_selection_val": {"mean": float(vals.mean()), "std": float(vals.std()),
                                            "median": float(np.median(vals)), "p90": float(np.percentile(vals, 90))},
                   "n_within_5e4_of_best": int((vals >= vals.max() - 5e-4).sum()),
                   "n_within_2e3_of_best": int((vals >= vals.max() - 2e-3).sum())}
        json.dump({"summary": summary, "rows": [rows[c] for c in order[:200]]},
                  open(f"{args.out}.json", "w"), indent=1)
        lines = [f"# Exhaustive k={args.k} subset search ({args.side}, {len(combos)} subsets, sharded)", "",
                 f"- champion `{list(champion)}` val={cv:.4f} -> **rank {rank}/{len(combos)}**, "
                 f"gap to best = {vals.max()-cv:.4f}",
                 f"- random-selection val: mean={vals.mean():.4f} +- {vals.std():.4f}", "",
                 "| rank | subset | val | " + " | ".join(es) + " |", "|---|---|---|" + "---|" * len(es)]
        for rk, c in enumerate(order[:20], 1):
            r = rows[c]
            mark = " **<- champion**" if c == champion else ""
            lines.append(f"| {rk} | {'+'.join(c)}{mark} | {r['val']:.4f} | "
                         + " | ".join(f"{r[m]:.4f}" for m in es) + " |")
        Path(f"{args.out}.md").write_text("\n".join(lines) + "\n")
        print(f"MERGE_DONE rank={rank}/{len(combos)} -> {args.out}.md", flush=True)
        return

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = {s: load_split(Path(args.cache), s, device) for s in ["train", "validation"] + es}
    av = sorted(splits["train"]["vecs"].keys())
    combos = all_combos(av, args.k)[args.shard::args.nshards]
    rows_path = Path(f"{args.out}_shard{args.shard}.rows.jsonl")
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if rows_path.exists():
        for line in open(rows_path):
            done.add(tuple(sorted(json.loads(line)["branches"])))
        print(f"[resume] shard {args.shard}: {len(done)}/{len(combos)} already done", flush=True)
    t0, n0 = time.time(), len(done)
    with open(rows_path, "a") as fh:
        for c in combos:
            if c in done:
                continue
            r = eval_subset(list(c), splits, args.side, device, list(range(args.seeds)), eval_splits=tuple(es))
            r["branches"] = list(c)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            done.add(c)
            if len(done) % 200 == 0:
                el = time.time() - t0
                rate = el / max(1, len(done) - n0)
                print(f"  shard{args.shard}: {len(done)}/{len(combos)} ({el:.0f}s, {rate:.1f}s/subset)", flush=True)
    print(f"SHARD_DONE {args.shard}/{args.nshards} n={len(done)}", flush=True)


if __name__ == "__main__":
    main()
