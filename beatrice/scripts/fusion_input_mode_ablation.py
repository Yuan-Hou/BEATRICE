#!/usr/bin/env python
"""Fusion-head INPUT-MODE ablation.

The deployed frozen-fusion head concatenates, per frozen branch:
  [ 256-d pooled vec | scalar branch logit | pairwise |logit_i - logit_j| ]
and trains a small MLP on top. This asks what each input family actually buys by
retraining that exact head (same MLP, modality-dropout, optimiser, 5e-4 early stop)
with different input compositions:

  logit                - pure logit ensemble (stack on branch predictions only)
  logit+disagree       - logits + pairwise disagreements
  vec                  - pure vector ensemble (drop logits & disagreements)
  vec+logit            - vecs + logits
  vec+logit+disagree   - the current deployed head

Run over a few branch sets (greedy champion / singles4 / all15) so we see whether
logits/disagreements matter more when branch predictions are diverse.

Reuses the cache loader from search_fusion_combos.py.

Usage:
  python scripts/fusion_input_mode_ablation.py --cache outputs/cached_fusion_vecs/ctx_15br \
      --side context --eval-splits test,popqa,nq,triviaqa --out outputs/reports/fusion_inputmode_ctx
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from search_fusion_combos import load_split  # noqa: E402

MODES = ["logit", "logit+disagree", "vec", "vec+logit", "vec+logit+disagree"]

SETS = {
    "context": {
        "champion(H+P+A,H+P+R)": ["hid_prob_attn", "hid_prob_resid"],
        "singles4(H,P,R,A)": ["hidden", "probability", "residual", "attention"],
        "all15": None,  # filled with all available
    },
    "question": {
        "champion(H+R,H+P+A,H+A)": ["hid_resid", "hid_prob_attn", "hid_attn"],
        "singles4(H,P,R,A)": ["hidden", "probability", "residual", "attention"],
        "all15": None,
    },
}


class InputModeHead(nn.Module):
    def __init__(self, n_branches, encoder_dim, mode, hidden_dim=128,
                 dropout=0.1, modality_dropout=0.25):
        super().__init__()
        self.n = n_branches
        self.mode = mode
        self.modality_dropout = float(modality_dropout)
        self.use_vec = "vec" in mode
        self.use_logit = "logit" in mode
        self.use_dis = "disagree" in mode
        npair = n_branches * (n_branches - 1) // 2
        dim = ((encoder_dim * n_branches if self.use_vec else 0)
               + (n_branches if self.use_logit else 0)
               + (npair if self.use_dis else 0))
        self.final = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, hidden_dim * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def _drop(self, vecs, logits):
        # drop whole branches (both vec + logit), same scheme as the deployed head
        if not self.training or self.modality_dropout <= 0 or self.n <= 1:
            return vecs, logits
        b = logits.size(0)
        keep = torch.rand(b, self.n, device=logits.device) >= self.modality_dropout
        dropped = ~keep.any(dim=1)
        if dropped.any():
            r = torch.randint(self.n, (int(dropped.sum()),), device=logits.device)
            keep[dropped] = False
            keep[dropped, r] = True
        kf = keep.to(logits.dtype) / max(1.0 - self.modality_dropout, 1e-6)
        return [v * kf[:, i:i + 1] for i, v in enumerate(vecs)], logits * kf

    def forward(self, vecs, logits):
        vecs, logits = self._drop(vecs, logits)
        parts = []
        if self.use_vec:
            parts += vecs
        if self.use_logit:
            parts.append(logits)
        if self.use_dis:
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    parts.append(torch.abs(logits[:, i:i + 1] - logits[:, j:j + 1]))
        return self.final(torch.cat(parts, dim=-1))


def train_eval(subset, mode, splits, device, seed, eval_splits,
               epochs=40, patience=10, bs=512, lr=2e-4, wd=1e-2):
    torch.manual_seed(seed)
    dim = splits["train"]["vecs"][subset[0]].shape[1]
    model = InputModeHead(len(subset), dim, mode).to(device)
    tr, va = splits["train"], splits["validation"]
    n = tr["y"].shape[0]
    pos = float(tr["y"].sum())
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(n - pos) / max(pos, 1)], device=device))
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    def stack(split, idx=None):
        vs = [split["vecs"][k] if idx is None else split["vecs"][k][idx] for k in subset]
        lg = torch.cat([split["logits"][k] if idx is None else split["logits"][k][idx]
                        for k in subset], dim=-1)
        return vs, lg

    def scores(split):
        model.eval()
        with torch.no_grad():
            vs, lg = stack(split)
            p = torch.sigmoid(model(vs, lg)).reshape(-1)
        return split["y"].cpu().numpy(), p.cpu().numpy()

    def auc(split):
        y, p = scores(split)
        return roc_auc_score(y, p)

    def auprc_err(split):
        # AUPRC, "LLM wrong" = positive; reporting only, selection stays val-AUROC
        y, p = scores(split)
        return average_precision_score(1 - y, -p)

    best, best_state, no_imp = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            vs, lg = stack(tr, idx)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(vs, lg), tr["y"][idx].unsqueeze(-1))
            loss.backward()
            opt.step()
        a = auc(va)
        if a > best + 1e-4:
            best, no_imp = a, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= patience:
                break
    model.load_state_dict(best_state)
    out = {"val": best, "val_auprc": auprc_err(splits["validation"])}
    for s in eval_splits:
        out[s] = auc(splits[s])
        out[f"{s}_auprc"] = auprc_err(splits[s])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--side", choices=("context", "question"), default="context")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--eval-splits", default="test,popqa")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_splits = [x for x in args.eval_splits.split(",") if x]
    cache = Path(args.cache)
    splits = {s: load_split(cache, s, device)
              for s in ["train", "validation"] + eval_splits}
    available = sorted(splits["train"]["vecs"].keys())
    if any(b.startswith("q_") for b in available):
        # joint (ctx + closed-book) cache: ablate the joint champion's head inputs
        sets = {
            "joint_champion(AHP,HPR,q_HA)": ["hid_prob_attn", "hid_prob_resid", "q_hid_attn"],
            "ctx_champion(ref)": ["hid_prob_attn", "hid_prob_resid"],
            "all30": None,
        }
    else:
        sets = dict(SETS[args.side])
    for k, v in sets.items():
        if v is None:
            sets[k] = available

    results = {}
    for set_name, subset in sets.items():
        subset = [b for b in subset if b in available]
        results[set_name] = {}
        for mode in MODES:
            rs = [train_eval(subset, mode, splits, device, s, eval_splits)
                  for s in range(args.seeds)]
            agg = {k: float(sum(r[k] for r in rs) / len(rs)) for k in rs[0]}
            results[set_name][mode] = agg
            extra = " ".join(f"{k}={agg[k]:.4f}" for k in eval_splits)
            print(f"  {set_name:26s} {mode:20s} val={agg['val']:.4f} {extra}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(f"{args.out}.json", "w"), indent=1)

    hdr = " | ".join(eval_splits)
    lines = [f"# Fusion input-mode ablation ({args.side}, seeds={args.seeds})", ""]
    for set_name, modes in results.items():
        lines += [f"## {set_name}", "",
                  f"| input mode | val | {hdr} |",
                  "|---|---|" + "---|" * len(eval_splits)]
        for mode in MODES:
            r = modes[mode]
            cells = " | ".join(f"{r[k]:.4f}" for k in eval_splits)
            lines.append(f"| {mode} | {r['val']:.4f} | {cells} |")
        lines.append("")
    Path(f"{args.out}.md").write_text("\n".join(lines) + "\n")
    print(f"INPUTMODE_DONE -> {args.out}.md", flush=True)


if __name__ == "__main__":
    main()
