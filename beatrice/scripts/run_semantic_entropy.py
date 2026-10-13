"""Compute Semantic Entropy (SE) and Semantic Entropy Probe (SEP) baselines from
the sampling pass, and write predictions in the standard schema so they slot into
``scripts/evaluate_predictions.py`` and the baseline-vs-fusion comparison.

Inputs (from scripts/sample_generations.py): ``data/se_samples/<setting>/<split>.jsonl``.
Hidden states for SEP come from the SAME features the baselines use (matched by
example_id), so SEP scores the identical examples as every other method.

Outputs (under --out-dir): one preds JSONL per method x split, with
``helpful_probability`` = P(correct):
  - semantic_entropy_discrete / semantic_entropy_proper : score = oriented -SE
  - sep_probe : score = 1 - P(high SE) from a hidden-state logistic probe

Usage:
    python scripts/run_semantic_entropy.py \
        --config configs/experiment/probe/baselines/baselines_question_noctx_qwen9b.yaml \
        --setting noctx --samples-dir data/se_samples/noctx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ragprobe.baselines.evaluate import format_report_table, metrics_overall_and_by_dataset
from ragprobe.baselines.extract import extract_split_signals
from ragprobe.baselines.semantic_entropy import (
    compute_se_for_samples, sep_predict_high_se_prob, train_sep_probe,
)
from ragprobe.training.dataset import load_feature_meta
from ragprobe.utils.config import load_yaml
from ragprobe.utils.io import read_jsonl, write_jsonl
from ragprobe.utils.logging import log


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _oriented_correctness_score(se_values: np.ndarray) -> np.ndarray:
    """High SE -> likely incorrect. Map -z(SE) through a sigmoid -> P(correct)-ish."""
    se = np.asarray(se_values, dtype=float)
    mu, sd = se.mean(), se.std() + 1e-8
    return _sigmoid(-(se - mu) / sd)


def _hidden_by_id(meta, segment, label_mode, positive_transition):
    sigs = extract_split_signals(meta, segment, label_mode, positive_transition)
    out = {}
    for s in sigs:
        if s.answer_hidden_mean is not None:
            out[s.example_id] = np.asarray(s.answer_hidden_mean, dtype=np.float32)
    return out


def _records(se_recs, scores):
    recs = []
    for r, sc in zip(se_recs, scores):
        recs.append({
            "example_id": r.example_id, "dataset": r.dataset, "split": r.split,
            "label": int(r.label), "helpful_probability": float(sc), "score": float(sc),
            "discrete_se": r.discrete_se, "proper_se": r.proper_se,
            "num_clusters": r.num_clusters, "k": r.k,
        })
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--setting", choices=["noctx", "withctx"], required=True)
    ap.add_argument("--samples-dir", default=None)
    ap.add_argument("--no-nli", action="store_true", help="Cluster by exact-match only (fast, no NLI).")
    ap.add_argument("--nli-model", default="microsoft/deberta-large-mnli")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]
    probe_cfg = cfg.get("probe", {})
    segment = probe_cfg.get("segment", "answer")
    label_mode = probe_cfg.get("label_mode", "without_context_correct")
    positive_transition = probe_cfg.get("positive_transition", "NY")
    threshold = float(probe_cfg.get("threshold", 0.5))

    samples_dir = Path(args.samples_dir or f"data/se_samples/{args.setting}")
    out_dir = Path(args.out_dir or f"outputs/predictions/baselines/se_{args.setting}")
    out_dir.mkdir(parents=True, exist_ok=True)
    use_nli = not args.no_nli

    # ---- Compute SE on every available split ----
    se_by_split = {}
    for path in sorted(samples_dir.glob("*.jsonl")):
        if path.name.endswith(".se.jsonl"):  # cache_semantic_entropy sidecars, not sample groups
            continue
        split_name = path.stem
        rows = list(read_jsonl(path))
        log(f"[SE] {split_name}: {len(rows)} examples, computing semantic entropy (nli={use_nli})...")
        se_by_split[split_name] = compute_se_for_samples(rows, use_nli=use_nli, nli_model=args.nli_model)

    eval_splits = [s for s in se_by_split if s != "train"]

    report = {}

    # ---- SE-direct baselines ----
    for split_name in eval_splits:
        recs = se_by_split[split_name]
        for key, attr in [("semantic_entropy_discrete", "discrete_se"),
                          ("semantic_entropy_proper", "proper_se")]:
            se_vals = np.array([getattr(r, attr) for r in recs])
            scores = _oriented_correctness_score(se_vals)
            out_recs = _records(recs, scores)
            write_jsonl(out_recs, str(out_dir / f"se_{args.setting}__{key}__{split_name}.jsonl"))
            report[f"{key} [{split_name}]"] = metrics_overall_and_by_dataset(
                out_recs, "helpful_probability", threshold)

    # ---- SEP probe (needs train SE labels + hidden states) ----
    if "train" in se_by_split:
        train_recs = se_by_split["train"]
        train_ids = {r.example_id for r in train_recs}
        # Pull hidden states for exactly the sampled train ids.
        train_meta_all = load_feature_meta(
            data_cfg["train_meta_paths"], splits=data_cfg.get("train_splits"),
            partition=data_cfg.get("train_partition"))
        train_meta = [m for m in train_meta_all
                      if str(m.get("example", m).get("id", "")) in train_ids]
        hid_train = _hidden_by_id(train_meta, segment, label_mode, positive_transition)

        Xtr, se_tr = [], []
        for r in train_recs:
            if r.example_id in hid_train:
                Xtr.append(hid_train[r.example_id])
                se_tr.append(r.discrete_se)
        Xtr = np.stack(Xtr) if Xtr else np.zeros((0, 1))
        se_tr = np.array(se_tr)
        log(f"[SEP] training probe on {len(Xtr)} examples (hidden_dim={Xtr.shape[1] if len(Xtr) else 0})")
        clf, scaler, thr = train_sep_probe(Xtr, se_tr)
        log(f"[SEP] SE median threshold={thr:.4f}; probe={'OK' if clf is not None else 'DEGENERATE'}")

        # Apply on eval splits using their hidden states.
        for split_name in eval_splits:
            recs = se_by_split[split_name]
            if split_name == "validation":
                meta = load_feature_meta(
                    data_cfg["validation_meta_paths"], splits=data_cfg.get("validation_splits"),
                    partition=data_cfg.get("validation_partition"))
            else:
                extra = next(e for e in data_cfg.get("eval_meta_sets", []) if e["name"] == split_name)
                meta = load_feature_meta(extra["meta_paths"], splits=extra.get("splits"),
                                         partition=extra.get("partition"))
            hid = _hidden_by_id(meta, segment, label_mode, positive_transition)
            kept = [r for r in recs if r.example_id in hid]
            if not kept:
                continue
            X = np.stack([hid[r.example_id] for r in kept])
            high_se_prob = sep_predict_high_se_prob(clf, scaler, X)
            scores = 1.0 - high_se_prob  # P(correct) = 1 - P(high SE)
            out_recs = _records(kept, scores)
            write_jsonl(out_recs, str(out_dir / f"se_{args.setting}__sep_probe__{split_name}.jsonl"))
            report[f"sep_probe [{split_name}]"] = metrics_overall_and_by_dataset(
                out_recs, "helpful_probability", threshold)

    (out_dir / f"se_{args.setting}__metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    table = format_report_table(report)
    (out_dir / f"se_{args.setting}__summary.txt").write_text(table + "\n", encoding="utf-8")
    log(f"Wrote SE/SEP metrics to {out_dir}")
    print("\n" + table)


if __name__ == "__main__":
    main()
