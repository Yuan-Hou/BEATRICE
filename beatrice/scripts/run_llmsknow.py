"""LLMsKnow (Orgad 2024, 'LLMs Know More Than They Show') baseline.

Faithful to technion-cs-nlp/LLMsKnow `probe_specific_layer_token`: a plain
LogisticRegression on the hidden state of the EXACT-ANSWER token at a single layer
(swept, selected on validation AUROC). For our short-answer QA the answer span IS the
exact answer, so its last token = `hidden_last` (already extracted in data/layer_features).
This is the paper's core "truth concentrates on the exact-answer token" probe; it differs
from SAPLMA (MLP) only in the linear classifier and the exact-answer-token choice.

With-context trains on the original+rewritten augmentation (same as SAPLMA/fusion).

    python scripts/run_llmsknow.py --setting withctx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ragprobe.baselines import aligned_labels as AL
from ragprobe.baselines.faithful import (
    auroc, concat_layerdata, load_layer_features, run_llmsknow,
)
from ragprobe.utils.logging import log

GROUPS = ["validation", "test_id", "popqa_ood"]


def _write(records, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    global GROUPS
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", choices=["noctx", "withctx"], required=True)
    ap.add_argument("--layer-base", default="data/layer_features",
                    help="layer-features root (gemma: data/gemma4/layer_features)")
    ap.add_argument("--groups", nargs="*", default=GROUPS,
                    help="Which eval groups to process (default: %(default)s).")
    ap.add_argument("--out-dir", default=None,
                    help="output dir (default {out_root}/baselines/faithful/{setting})")
    args = ap.parse_args()
    GROUPS = list(args.groups)
    setting = args.setting
    out_dir = Path(args.out_dir or f"{AL.out_root()}/baselines/faithful/{setting}")

    train = AL.filter_layerdata(load_layer_features(setting, "train", base=args.layer_base), setting, "train")
    evalsets = {g: AL.filter_layerdata(load_layer_features(setting, g, base=args.layer_base), setting, g) for g in GROUPS}
    if setting == "withctx":
        try:
            train_rew = AL.filter_layerdata(load_layer_features(setting, "train_rewritten", base=args.layer_base),
                                            "rewritten", "train_rewritten")
            train = concat_layerdata(train, train_rew)
            log(f"  + rewritten aug: train -> {len(train.label)}")
        except FileNotFoundError:
            log("  (no rewritten aug found)")

    scores, best_layer, per = run_llmsknow(train, evalsets)
    log(f"[llmsknow] best_layer={best_layer} val_auc={per[best_layer]:.4f}")

    summary = []
    for g in GROUPS:
        ds = evalsets[g]; sc = scores[g]
        recs = [{
            "example_id": str(ds.example_id[i]), "dataset": str(ds.dataset[i]),
            "split": g, "label": int(ds.label[i]),
            "helpful_probability": float(sc[i]), "score": float(sc[i]),
        } for i in range(len(ds.label))]
        _write(recs, out_dir / f"faithful_{setting}__llmsknow__{g}.jsonl")
        summary.append((g, len(recs), float(ds.label.mean()), auroc(ds.label, sc)))

    print(f"\n# LLMsKnow — {setting}  (best_layer={best_layer})")
    print(f"{'group':12s} {'N':>5s} {'pos%':>6s} {'AUROC':>7s}")
    for g, n, pr, au in summary:
        print(f"{g:12s} {n:5d} {pr*100:5.1f}% {au:7.4f}")


if __name__ == "__main__":
    main()
