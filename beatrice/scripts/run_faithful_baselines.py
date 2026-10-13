"""Run the paper-faithful baselines (layer-swept SAPLMA/SEP, full LLM-Check, SE)
on the re-extracted per-layer features, and write predictions in the standard
schema so they fold into the same-sample comparison and evaluator.

Usage (vllm2 env, after extraction + SE cache finish):
    CUDA_VISIBLE_DEVICES=5 python scripts/run_faithful_baselines.py \
        --config configs/experiment/probe/baselines/baselines_question_noctx_qwen9b.yaml \
        --setting noctx --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ragprobe.baselines import aligned_labels as AL
from ragprobe.baselines.faithful import (
    auroc, concat_layerdata, load_layer_features, load_se_cache, run_llmcheck,
    run_saplma, run_sep, se_scores,
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
    ap.add_argument("--config", required=True)  # kept for symmetry / scope provenance
    ap.add_argument("--setting", choices=["noctx", "withctx"], required=True)
    ap.add_argument("--layer-base", default="data/layer_features",
                    help="layer-features root (gemma: data/gemma4/layer_features)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--groups", nargs="*", default=GROUPS,
                    help="Which eval groups to process (default: %(default)s).")
    args = ap.parse_args()
    GROUPS = list(args.groups)

    setting = args.setting
    out_dir = Path(args.out_dir or f"{AL.out_root()}/baselines/faithful/{setting}")

    log(f"loading layer features for {setting} ...")
    train = AL.filter_layerdata(load_layer_features(setting, "train", base=args.layer_base), setting, "train")
    evalsets = {g: AL.filter_layerdata(load_layer_features(setting, g, base=args.layer_base), setting, g) for g in GROUPS}
    train_se = load_se_cache(setting, "train")

    # With-context fusion trains on the rewritten-context augmentation; give the
    # trained baselines (SAPLMA) the same data for fairness. SEP/LLM-Check stay on
    # the original train (no SE labels on rewritten; training-free orientation).
    train_saplma = train
    sep_sources = [(train, train_se)]
    try:
        train_rew = AL.filter_layerdata(load_layer_features(setting, "train_rewritten", base=args.layer_base),
                                        "rewritten", "train_rewritten")
        train_saplma = concat_layerdata(train, train_rew)
        log(f"  + rewritten-context aug: SAPLMA train {len(train.label)} -> {len(train_saplma.label)}")
        # SEP also gets the augmentation IF rewritten SE labels were sampled.
        train_rew_se = load_se_cache(setting, "train_rewritten")
        if train_rew_se:
            sep_sources.append((train_rew, train_rew_se))
            log(f"  + rewritten SE labels: SEP train {len(train_se)} + {len(train_rew_se)}")
        else:
            log("  (no rewritten SE labels; SEP on original SE only)")
    except FileNotFoundError:
        log("  (no rewritten-context aug found; SAPLMA on original train only)")
    log(f"train N={len(train.label)} pos={train.label.mean():.3f}; "
        f"layers={train.hidden_last.shape[1]} d={train.hidden_last.shape[2]}; "
        f"attn_layers={None if train.attn_logdet is None else train.attn_logdet.shape[1]}; "
        f"train_se={len(train_se)}")

    methods = {}  # name -> {group: scores}
    meta = {}     # name -> note

    # SAPLMA (last-token primary, mean-pool ablation) — uses rewritten aug if present
    for pool in ("last", "mean"):
        sc, best, per = run_saplma(train_saplma, evalsets, pooling=pool, device=args.device)
        methods[f"saplma_{pool}"] = sc
        meta[f"saplma_{pool}"] = f"best_layer={best} (val_auc={per[best]:.4f})"
        log(f"[saplma_{pool}] {meta[f'saplma_{pool}']}")

    # SEP (logistic probe -> binarized SE), last-token; aug sources if present
    sc, best, per = run_sep(sep_sources, evalsets, pooling="last", device=args.device)
    methods["sep_probe"] = sc
    meta["sep_probe"] = f"best_layer={best} (val_auc={per[best]:.4f})" if best >= 0 else "DEGENERATE"
    log(f"[sep_probe] {meta['sep_probe']}")

    # LLM-Check family
    for name, sc in run_llmcheck(train, evalsets).items():
        methods[name] = sc

    # Semantic Entropy direct (discrete + proper) from cache
    methods["semantic_entropy_discrete"] = se_scores(evalsets, setting, "discrete")
    methods["semantic_entropy_proper"] = se_scores(evalsets, setting, "proper")

    # ---- write predictions + collect a summary ----
    summary = []
    for mname, by_group in methods.items():
        for g in GROUPS:
            ds = evalsets[g]
            scores = by_group[g]
            recs = [{
                "example_id": str(ds.example_id[i]), "dataset": str(ds.dataset[i]),
                "split": g, "label": int(ds.label[i]),
                "helpful_probability": float(scores[i]), "score": float(scores[i]),
            } for i in range(len(ds.label))]
            _write(recs, out_dir / f"faithful_{setting}__{mname}__{g}.jsonl")
            summary.append((mname, g, len(recs), float(ds.label.mean()), auroc(ds.label, scores)))

    # text summary
    lines = [f"# Faithful baselines — {setting}", ""]
    for k, v in meta.items():
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append(f"{'method':28s} {'group':12s} {'N':>5s} {'pos%':>6s} {'AUROC':>7s}")
    for mname, g, n, pr, au in sorted(summary, key=lambda x: (x[1], -x[4] if x[4]==x[4] else 0)):
        lines.append(f"{mname:28s} {g:12s} {n:5d} {pr*100:5.1f}% {au:7.4f}")
    text = "\n".join(lines)
    (out_dir / f"faithful_{setting}__summary.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
