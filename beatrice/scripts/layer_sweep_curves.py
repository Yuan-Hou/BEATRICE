"""Re-run the hidden-layer sweep and PERSIST the full per-layer AUROC curves.

Round 3 selected L17 for the fusion's hidden branch via the baseline layer sweep
(saplma_mean/sep best layer = L17 on validation), but only best_layer + val_auc were
saved — no curve. This script retrains a probe per layer and records AUROC on
validation, test_id and popqa_ood for every layer, giving the "AUROC vs layer" figure
that justifies the L17 choice in the paper.

Probes (the ones relevant to our hidden branch):
  - saplma_mean : MLP[256,128,64] on mean-pooled answer hidden  (paper-faithful SAPLMA ablation)
  - saplma_last : MLP on last-answer-token hidden               (SAPLMA original)
  - llmsknow    : LogisticRegression on exact-answer-token hidden

Data: data/layer_features/{noctx,withctx}/ (33 = embeddings + 32 blocks).
Out:  outputs/reports/layer_sweep_curves.json + layer_sweep_curves.md

Usage: CUDA_VISIBLE_DEVICES=7 python scripts/layer_sweep_curves.py [--settings noctx,withctx]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ragprobe.baselines.faithful import (
    LayerData,
    _mlp_predict,
    _train_mlp,
    auroc,
    load_layer_features,
)

GROUPS = ("validation", "test_id", "popqa_ood")


def sweep_saplma(train: LayerData, evalsets: dict, pooling: str, device: str):
    Xtr_all = train.hidden_last if pooling == "last" else train.hidden_mean
    n_layers = Xtr_all.shape[1]
    curves = {g: [] for g in GROUPS}
    for l in range(n_layers):
        t0 = time.time()
        net, sc, _ = _train_mlp(Xtr_all[:, l], train.label.astype(np.float32),
                                evalsets["validation"].hidden_mean[:, l], device=device)
        for g in GROUPS:
            ds = evalsets[g]
            X = (ds.hidden_last if pooling == "last" else ds.hidden_mean)[:, l]
            curves[g].append(float(auroc(ds.label, _mlp_predict(net, sc, X, device=device))))
        print(f"  saplma_{pooling} L{l:02d} val={curves['validation'][-1]:.4f} "
              f"test={curves['test_id'][-1]:.4f} ood={curves['popqa_ood'][-1]:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    return curves


def sweep_llmsknow(train: LayerData, evalsets: dict):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    n_layers = train.hidden_last.shape[1]
    curves = {g: [] for g in GROUPS}
    for l in range(n_layers):
        sc = StandardScaler().fit(train.hidden_last[:, l])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(train.hidden_last[:, l]), train.label)
        for g in GROUPS:
            ds = evalsets[g]
            p = clf.predict_proba(sc.transform(ds.hidden_last[:, l]))[:, 1]
            curves[g].append(float(auroc(ds.label, p)))
        print(f"  llmsknow L{l:02d} val={curves['validation'][-1]:.4f} "
              f"test={curves['test_id'][-1]:.4f} ood={curves['popqa_ood'][-1]:.4f}", flush=True)
    return curves


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", default="noctx,withctx")
    ap.add_argument("--out", default="outputs/reports/layer_sweep_curves")
    ap.add_argument("--base", default="data/layer_features")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results: dict = {}
    for setting in [s.strip() for s in args.settings.split(",") if s.strip()]:
        print(f"[{setting}] loading layer features ...", flush=True)
        train = load_layer_features(setting, "train", base=args.base)
        evalsets = {g: load_layer_features(setting, g, base=args.base) for g in GROUPS}
        print(f"[{setting}] train N={len(train.label)} layers={train.hidden_last.shape[1]}", flush=True)
        results[setting] = {}
        for pooling in ("mean", "last"):
            print(f"[{setting}] saplma_{pooling} sweep", flush=True)
            results[setting][f"saplma_{pooling}"] = sweep_saplma(train, evalsets, pooling, device)
        print(f"[{setting}] llmsknow sweep", flush=True)
        results[setting]["llmsknow"] = sweep_llmsknow(train, evalsets)
        del train, evalsets

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(f"{out}.json", "w"), indent=1)

    lines = ["# Hidden-layer sweep curves (AUROC vs layer)", ""]
    for setting, probes in results.items():
        lines.append(f"## {setting}")
        for probe, curves in probes.items():
            va = curves["validation"]
            best = int(np.argmax(va))
            lines.append(f"\n### {probe}  (best val layer = L{best}, val={va[best]:.4f}, "
                         f"test={curves['test_id'][best]:.4f}, ood={curves['popqa_ood'][best]:.4f})")
            lines.append("| layer | val | test | popqa |")
            lines.append("|---|---|---|---|")
            for l in range(len(va)):
                mark = " **" if l == best else " "
                lines.append(f"| L{l}{mark}| {va[l]:.4f} | {curves['test_id'][l]:.4f} | {curves['popqa_ood'][l]:.4f} |")
        lines.append("")
    open(f"{out}.md", "w").write("\n".join(lines))
    print(f"LAYER_SWEEP_DONE -> {out}.json / {out}.md", flush=True)


if __name__ == "__main__":
    main()
