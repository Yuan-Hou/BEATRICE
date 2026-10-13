#!/usr/bin/env python
"""Per-dataset (musique/hotpotqa/twowiki) test breakdown for every compact-table row.

Same protocol as search_fusion_combos.eval_subset (real fusion head class, 3 seeds,
epochs=40/patience=10/bs=512/lr=2e-4/wd=1e-2, val-AUROC selection); reports the
3-seed mean of per-dataset AUROC and AUPRC (positive = LLM wrong) on the ID test.

Writes outputs/reports/compact_rows_perdataset.json:
  {"qonly": {row_key: {musique, musique_auprc, hotpotqa, ..., twowiki_auprc}}, "ctx": {...}}

Usage: CUDA_VISIBLE_DEVICES=<gpu> python scripts/compute_compact_perdataset.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, "src")
from ragprobe.training.model import (  # noqa: E402
    FrozenContextQuestionAnswerEncoderFusionProbe,
    FrozenQuestionAnswerEncoderFusionProbe,
)

R = Path("outputs/reports")
DS = ["musique", "hotpotqa", "twowiki"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load(cache, s):
    d = torch.load(f"outputs/cached_fusion_vecs/{cache}/{s}.pt",
                   map_location="cpu", weights_only=False)
    return {"vecs": {k: v.to(DEV) for k, v in d["vecs"].items()},
            "logits": {k: v.to(DEV) for k, v in d["logits"].items()},
            "y": d["labels"].reshape(-1).to(DEV).float(), "eids": d["example_ids"]}


def eval_perds(cache, subset, side, seeds=3):
    cls = (FrozenContextQuestionAnswerEncoderFusionProbe if side == "context"
           else FrozenQuestionAnswerEncoderFusionProbe)
    splits = {s: load(cache, s) for s in ["train", "validation", "test"]}
    tr, va, te = splits["train"], splits["validation"], splits["test"]
    dim = tr["vecs"][subset[0]].shape[1]
    n = tr["y"].shape[0]
    pos = float(tr["y"].sum())
    per_seed = []
    for sd in range(seeds):
        torch.manual_seed(sd)
        model = cls(encoders={}, modality_names=list(subset), encoder_dim=dim,
                    hidden_dim=128, dropout=0.1, modality_dropout=0.25).to(DEV)
        crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(n - pos) / max(pos, 1)], device=DEV))
        opt = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                          lr=2e-4, weight_decay=1e-2)

        def fwd(sp):
            return torch.sigmoid(model.forward_from_vecs(sp["vecs"], sp["logits"])).reshape(-1)

        best, bs_, ni = -1.0, None, 0
        for _ in range(40):
            model.train()
            perm = torch.randperm(n, device=DEV)
            for s in range(0, n, 512):
                idx = perm[s:s + 512]
                vb = {k: tr["vecs"][k][idx] for k in subset}
                lb = {k: tr["logits"][k][idx] for k in subset}
                opt.zero_grad(set_to_none=True)
                loss = crit(model.forward_from_vecs(vb, lb), tr["y"][idx].unsqueeze(-1))
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                a = roc_auc_score(va["y"].cpu(), fwd(va).cpu())
            if a > best + 1e-4:
                best, bs_, ni = a, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
            else:
                ni += 1
                if ni >= 10:
                    break
        model.load_state_dict(bs_)
        model.eval()
        with torch.no_grad():
            p = fwd(te).cpu().numpy()
        yy = te["y"].cpu().numpy()
        row = {}
        for ds in DS:
            mask = np.array([e.startswith(ds) for e in te["eids"]])
            row[ds] = roc_auc_score(yy[mask], p[mask])
            row[f"{ds}_auprc"] = average_precision_score(1 - yy[mask], -p[mask])
        per_seed.append(row)
    return {k: float(np.mean([r[k] for r in per_seed])) for k in per_seed[0]}


def main():
    qfam = json.loads((R / "qonly_combo_search_full.json").read_text())["families"]
    qtraj = json.loads((R / "qonly_combo_search_full.json").read_text())["greedy_trajectory"]
    ctraj = json.loads((R / "ctx_combo_search_15br.json").read_text())["greedy_trajectory"]
    jtraj = json.loads((R / "joint_combo_search_30br.json").read_text())["greedy_trajectory"]
    qloo = json.loads((R / "loo_greedy_qonly.json").read_text())["runs"]
    jloo = json.loads((R / "loo_greedy_joint.json").read_text())["runs"]
    jall = sorted(torch.load("outputs/cached_fusion_vecs/joint_30br/validation.pt",
                             weights_only=False)["vecs"].keys())
    call = [b for b in jall if not b.startswith("q_")]

    SINGLES = ["hidden", "probability", "residual", "attention"]
    jobs = {
        "qonly": [
            ("hidden_only", "qonly_all15", ["hidden"], "question"),
            ("drop_H", "qonly_all15", qloo["drop_H"]["champion"]["branches"], "question"),
            ("quad", "qonly_all15", ["quad"], "question"),
            ("singles4", "qonly_all15", SINGLES, "question"),
            ("all", "qonly_all15", call, "question"),
            ("full", "qonly_all15", qtraj[-1]["branches"], "question"),
        ],
        "ctx": [
            ("hidden_only", "ctx_15br", ["hidden"], "context"),
            ("drop_H", "joint_30br", jloo["drop_H"]["champion"]["branches"], "context"),
            ("quad", "ctx_15br", ["quad"], "context"),
            ("singles4", "ctx_15br", SINGLES, "context"),
            ("all", "joint_30br", jall, "context"),
            ("wo_closed_book", "ctx_15br", ctraj[-1]["branches"], "context"),
            ("full", "joint_30br", jtraj[-1]["branches"], "context"),
        ],
    }
    out = {}
    for side, items in jobs.items():
        out[side] = {}
        for key, cache, subset, s in items:
            r = eval_perds(cache, subset, s)
            out[side][key] = r
            print(f"[{side}/{key}] " + " ".join(f"{d}={r[d]:.4f}" for d in DS), flush=True)
    json.dump(out, open(R / "compact_rows_perdataset.json", "w"), indent=1)
    print("PERDATASET_DONE")


if __name__ == "__main__":
    main()
