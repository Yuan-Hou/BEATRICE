"""Pick Probing-RAG's theta on dev by offline replay of full-round traces.

Run after the dev record_all_rounds runs:
  cd <package_root> && python -m pgir.probingrag.tune_theta \
      --traces pgir/runs/flashrag/probingrag_tune_*/**/probingrag_trace.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from pgir.probingrag.pipeline import replay_theta
from pgir.tuning.trace import em, f1, load_golds

THETAS = [-2.0, -1.0, 0.0, 1.0, 2.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", nargs="+", required=True)
    ap.add_argument("--max-iter", type=int, default=4)
    args = ap.parse_args()

    paths = []
    for t in args.traces:
        paths.extend(glob.glob(t, recursive=True))
    traces = []
    for p in sorted(set(paths)):
        traces.extend(json.loads(l) for l in open(p))
    golds = load_golds(Path(__file__).resolve().parent.parent / "datasets")
    print(f"{len(traces)} traces from {len(set(paths))} files")

    best = None
    for theta in THETAS:
        final = replay_theta(traces, theta, args.max_iter)
        n = len(final)
        ems = sum(em(a, golds[i]) for i, a in final.items()) / n
        f1s = sum(f1(a, golds[i]) for i, a in final.items()) / n
        # cost: rounds executed until stop
        rounds = 0
        for tr in traces:
            stop = next((r["round"] for r in tr["rounds"]
                         if r["margin"] > theta or r["round"] == args.max_iter),
                        tr["rounds"][-1]["round"] if tr["rounds"] else 0)
            rounds += stop + 1
        print(f"theta={theta:+.1f}  EM={ems:.4f}  F1={f1s:.4f}  gen/item={rounds/n:.2f}")
        if best is None or ems > best[1]:
            best = (theta, ems)
    print(f"chosen theta = {best[0]} (dev EM {best[1]:.4f})")


if __name__ == "__main__":
    main()
