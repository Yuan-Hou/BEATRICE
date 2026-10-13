"""Fusion combination search over precomputed frozen-branch vectors.

Given a cache holding pooled vecs + logits for ALL trained encoders (singles, pairs,
triples, quad — any subset family), this searches how to combine them:

  1. natural families (each single, pairs6, triples4, quad, singles+pairs, ...),
  2. greedy forward selection on VALIDATION AUC (the "most efficient" minimal set),
  3. everything trained with multiple seeds (differences are ~0.003; we average).

Selection uses validation only; test/popqa are computed for reporting and never drive
any choice. Each candidate trains the standard frozen-fusion head (same class as the
real experiments, via forward_from_vecs) in seconds, so the search is cheap.

Usage:
  python scripts/search_fusion_combos.py --cache outputs/cached_fusion_vecs/qonly_all15 \
      --side question --out outputs/reports/qonly_combo_search
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import average_precision_score, roc_auc_score

from ragprobe.training.model import (
    FrozenContextQuestionAnswerEncoderFusionProbe,
    FrozenQuestionAnswerEncoderFusionProbe,
)

SINGLES = ["hidden", "probability", "residual", "attention"]
PAIRS = ["hid_prob", "hid_resid", "hid_attn", "prob_resid", "prob_attn", "resid_attn"]
TRIPLES = ["hid_prob_resid", "hid_prob_attn", "hid_resid_attn", "prob_resid_attn"]
QUAD = ["quad"]


def load_split(cache: Path, name: str, device):
    d = torch.load(cache / f"{name}.pt", map_location="cpu", weights_only=False)
    return {
        "vecs": {k: v.to(device) for k, v in d["vecs"].items()},
        "logits": {k: v.to(device) for k, v in d["logits"].items()},
        "y": d["labels"].reshape(-1).to(device=device, dtype=torch.float32),
    }


def train_eval_subset(subset, splits, side, device, seed, eval_splits=("test", "popqa"),
                      epochs=40, patience=10, batch_size=512, lr=2e-4, wd=1e-2):
    torch.manual_seed(seed)
    cls = (FrozenContextQuestionAnswerEncoderFusionProbe if side == "context"
           else FrozenQuestionAnswerEncoderFusionProbe)
    dim = splits["train"]["vecs"][subset[0]].shape[1]
    model = cls(encoders={}, modality_names=list(subset), encoder_dim=dim,
                hidden_dim=128, dropout=0.1, modality_dropout=0.25).to(device)
    tr, va = splits["train"], splits["validation"]
    n = tr["y"].shape[0]
    pos = float(tr["y"].sum())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(n - pos) / max(pos, 1)], device=device))
    opt = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)

    def _scores(split):
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model.forward_from_vecs(split["vecs"], split["logits"])).reshape(-1)
        return split["y"].cpu().numpy(), p.cpu().numpy()

    def _auc(split):
        y, p = _scores(split)
        return roc_auc_score(y, p)

    def _auprc_err(split):
        # AUPRC with "LLM answered WRONG" as positive (error detection): flip labels,
        # rank by -P(correct). Reporting only — epoch selection stays val-AUROC.
        y, p = _scores(split)
        return average_precision_score(1 - y, -p)

    best_va, best_state, no_imp = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            vb = {k: tr["vecs"][k][idx] for k in subset}
            lb = {k: tr["logits"][k][idx] for k in subset}
            opt.zero_grad(set_to_none=True)
            loss = criterion(model.forward_from_vecs(vb, lb), tr["y"][idx].unsqueeze(-1))
            loss.backward()
            opt.step()
        a = _auc(va)
        if a > best_va + 1e-4:
            best_va, no_imp = a, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= patience:
                break
    model.load_state_dict(best_state)
    out = {"val": best_va, "val_auprc": _auprc_err(va)}
    for s in eval_splits:
        out[s] = _auc(splits[s])
        out[f"{s}_auprc"] = _auprc_err(splits[s])
    return out


def eval_subset(subset, splits, side, device, seeds, eval_splits=("test", "popqa")):
    rs = [train_eval_subset(subset, splits, side, device, seed=s, eval_splits=eval_splits) for s in seeds]
    return {k: sum(r[k] for r in rs) / len(rs) for k in rs[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--side", choices=("context", "question"), default="question")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-greedy", type=int, default=8, help="max branches in greedy set")
    ap.add_argument("--eval-splits", default="test,popqa",
                    help="comma list of cache splits to report (beyond val)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache = Path(args.cache)
    eval_splits = [x for x in args.eval_splits.split(",") if x]
    splits = {s: load_split(cache, s, device) for s in ["train", "validation"] + eval_splits}
    available = sorted(splits["train"]["vecs"].keys())
    print(f"available branches ({len(available)}): {available}", flush=True)
    seeds = list(range(args.seeds))
    results: dict[str, dict] = {}

    def ev(name, subset):
        subset = [b for b in subset if b in available]
        if not subset or name in results:
            return
        t0 = time.time()
        r = eval_subset(subset, splits, args.side, device, seeds, eval_splits=eval_splits)
        r.update({"branches": subset, "k": len(subset)})
        results[name] = r
        extra = " ".join(f"{k}={r[k]:.4f}" for k in eval_splits)
        print(f"  {name:28s} k={len(subset):2d} val={r['val']:.4f} {extra} ({time.time()-t0:.0f}s)", flush=True)

    # ---- 1. natural families ----
    print("[families]", flush=True)
    for b in available:
        ev(f"single:{b}", [b])
    ev("singles4", SINGLES)
    ev("pairs6", PAIRS)
    ev("triples4", TRIPLES)
    ev("quad1", QUAD)
    ev("singles4_pairs6", SINGLES + PAIRS)
    ev("pairs6_triples4", PAIRS + TRIPLES)
    ev("triples4_quad", TRIPLES + QUAD)
    ev("pairs6_triples4_quad", PAIRS + TRIPLES + QUAD)
    ev("all15", SINGLES + PAIRS + TRIPLES + QUAD)

    # ---- 2. greedy forward selection on validation ----
    print("[greedy forward selection]", flush=True)
    chosen: list[str] = []
    best_val = -1.0
    trajectory = []
    while len(chosen) < args.max_greedy:
        cands = [b for b in available if b not in chosen]
        step = {}
        for b in cands:
            r = eval_subset(chosen + [b], splits, args.side, device, seeds, eval_splits=eval_splits)
            step[b] = r
        pick = max(step, key=lambda b: step[b]["val"])
        if step[pick]["val"] <= best_val + 5e-4:
            print(f"  stop: no gain > 5e-4 (best candidate {pick} val={step[pick]['val']:.4f})", flush=True)
            break
        chosen.append(pick)
        best_val = step[pick]["val"]
        trajectory.append({"added": pick, **step[pick], "branches": list(chosen)})
        extra = " ".join(f"{k}={step[pick][k]:.4f}" for k in eval_splits)
        print(f"  + {pick:18s} -> val={step[pick]['val']:.4f} {extra} (k={len(chosen)})", flush=True)
    results[f"greedy_k{len(chosen)}"] = {**trajectory[-1], "k": len(chosen)} if trajectory else {}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"families": results, "greedy_trajectory": trajectory},
              open(f"{out}.json", "w"), indent=1)

    hdr = " | ".join(eval_splits)
    lines = [f"# Fusion combination search ({args.side}, seeds={args.seeds})", "",
             f"| combo | k | val | {hdr} |", "|---|---|" + "---|" * (1 + len(eval_splits))]
    for name, r in sorted(results.items(), key=lambda kv: -(kv[1].get("val") or 0)):
        if not r:
            continue
        cells = " | ".join(f"{r[k]:.4f}" for k in eval_splits)
        lines.append(f"| {name} | {r['k']} | {r['val']:.4f} | {cells} |")
    lines += ["", "## Greedy trajectory (selection on validation only)", "",
              f"| step | added | k | val | {hdr} |", "|---|---|---|---|" + "---|" * len(eval_splits)]
    for i, t in enumerate(trajectory, 1):
        cells = " | ".join(f"{t[k]:.4f}" for k in eval_splits)
        lines.append(f"| {i} | {t['added']} | {len(t['branches'])} | {t['val']:.4f} | {cells} |")
    Path(f"{out}.md").write_text("\n".join(lines) + "\n")
    print(f"COMBO_SEARCH_DONE -> {out}.md", flush=True)


if __name__ == "__main__":
    main()
