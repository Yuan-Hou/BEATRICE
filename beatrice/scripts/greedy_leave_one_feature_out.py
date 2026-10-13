#!/usr/bin/env python
"""Leave-one-feature-out greedy ablation.

For each of the four feature channels (H=hidden, P=probability, R=residual,
A=attention) we DROP every candidate encoder-branch that uses that channel and
re-run the exact same greedy forward selection (add the branch that most improves
validation AUROC; stop when the best gain <= 5e-4). This isolates how much the
greedy fusion loses when a whole feature family is unavailable.

Reuses the search machinery / stop rule from search_fusion_combos.py so numbers
are directly comparable to the full search.

Usage:
  python scripts/greedy_leave_one_feature_out.py \
      --cache outputs/cached_fusion_vecs/ctx_15br --side context \
      --eval-splits test,popqa,nq,triviaqa --out outputs/reports/loo_greedy_ctx
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from search_fusion_combos import eval_subset, load_split  # noqa: E402

_CH = {"hid": "H", "hidden": "H", "prob": "P", "probability": "P",
       "resid": "R", "residual": "R", "attn": "A", "attention": "A"}


def branch_channels(branch: str) -> set[str]:
    # joint caches prefix question-only branches with "q_"; the prefix is a
    # conditioning marker, not a feature channel — strip it before parsing.
    if branch.startswith("q_"):
        branch = branch[2:]
    if branch in ("quad", "quad1", "hid_prob_resid_attn"):
        return set("HPRA")
    return {_CH.get(tok, tok) for tok in branch.split("_")}


def greedy(available, splits, side, device, seeds, eval_splits, max_greedy=8):
    """Same greedy + 5e-4 stop rule as search_fusion_combos.main."""
    chosen, best_val, traj = [], -1.0, []
    while len(chosen) < max_greedy:
        cands = [b for b in available if b not in chosen]
        step = {b: eval_subset(chosen + [b], splits, side, device, seeds,
                               eval_splits=eval_splits) for b in cands}
        pick = max(step, key=lambda b: step[b]["val"])
        if step[pick]["val"] <= best_val + 5e-4:
            rej = {"added": pick, **step[pick]}
            return traj, rej
        chosen.append(pick)
        best_val = step[pick]["val"]
        traj.append({"added": pick, **step[pick], "branches": list(chosen)})
    return traj, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--side", choices=("context", "question"), default="question")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--eval-splits", default="test,popqa")
    ap.add_argument("--channels", default="H,P,R,A",
                    help="which feature channels to drop (comma-separated subset of H,P,R,A)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_splits = [x for x in args.eval_splits.split(",") if x]
    cache = Path(args.cache)
    splits = {s: load_split(cache, s, device)
              for s in ["train", "validation"] + eval_splits}
    available = sorted(splits["train"]["vecs"].keys())
    seeds = list(range(args.seeds))
    fname = {"H": "Hidden", "P": "Prob", "R": "Resid", "A": "Attn"}

    out = {"eval_splits": eval_splits, "runs": {}}

    # reference: full greedy (all channels available)
    print("[full] greedy over all branches", flush=True)
    traj, rej = greedy(available, splits, args.side, device, seeds, eval_splits)
    out["runs"]["full"] = {"trajectory": traj, "rejected_next": rej,
                           "champion": traj[-1] if traj else None,
                           "n_candidates": len(available)}
    print(f"  full champion k={len(traj)}: {traj[-1]['branches']} "
          f"val={traj[-1]['val']:.4f}", flush=True)

    # leave-one-feature-out
    for ch in [c for c in ["H", "P", "R", "A"] if c in args.channels.split(",")]:
        avail = [b for b in available if ch not in branch_channels(b)]
        print(f"[drop {ch}={fname[ch]}] {len(avail)} candidate branches: {avail}",
              flush=True)
        traj, rej = greedy(avail, splits, args.side, device, seeds, eval_splits)
        champ = traj[-1] if traj else None
        out["runs"][f"drop_{ch}"] = {
            "dropped": fname[ch], "n_candidates": len(avail),
            "trajectory": traj, "rejected_next": rej, "champion": champ}
        if champ:
            extra = " ".join(f"{k}={champ[k]:.4f}" for k in eval_splits)
            print(f"  drop-{ch} champion k={len(traj)}: {champ['branches']} "
                  f"val={champ['val']:.4f} {extra}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(f"{args.out}.json", "w"), indent=1)

    # markdown table
    hdr = " | ".join(eval_splits)
    lines = [f"# Leave-one-feature-out greedy ({args.side}, seeds={args.seeds})", "",
             f"| condition | greedy set | k | val | {hdr} |",
             "|---|---|---|---|" + "---|" * len(eval_splits)]
    order = [("full", "— (all features)")] + \
            [(f"drop_{c}", f"drop {fname[c]}") for c in ["H", "P", "R", "A"]]
    for key, label in order:
        r = out["runs"].get(key)
        if not r or not r["champion"]:
            continue
        c = r["champion"]
        bset = "+".join("".join(sorted(branch_channels(b))) for b in c["branches"])
        cells = " | ".join(f"{c[k]:.4f}" for k in eval_splits)
        lines.append(f"| {label} | {bset} | {len(c['branches'])} | "
                     f"{c['val']:.4f} | {cells} |")
    Path(f"{args.out}.md").write_text("\n".join(lines) + "\n")
    print(f"LOO_GREEDY_DONE -> {args.out}.md", flush=True)


if __name__ == "__main__":
    main()
