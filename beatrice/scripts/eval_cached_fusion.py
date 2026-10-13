"""Evaluate a frozen-fusion checkpoint from precomputed branch vectors.

Drop-in fast replacement for evaluate_probe.py on frozen-fusion architectures: the raw
path re-runs the 4 frozen branch encoders over 8GB/batch raw features (hours per eval);
here the branch vecs come from the precompute cache, and only the tiny trainable head
runs. No LLM embeddings or branch encoders are loaded at all. Writes the same
predictions jsonl (+ ROC json/png) and logs the same "Eval metrics" line.

Usage:
  python scripts/eval_cached_fusion.py --config <eval cfg> \
      --cache outputs/cached_fusion_vecs/ctx_L17rd_attn --split test
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from ragprobe.training.dataset import _get_example_id, load_feature_meta
from ragprobe.training.evaluator import _write_roc_outputs
from ragprobe.training.model import (
    FrozenContextQuestionAnswerEncoderFusionProbe,
    FrozenQuestionAnswerEncoderFusionProbe,
)
from ragprobe.training.trainer import compute_metrics
from ragprobe.utils.config import load_yaml
from ragprobe.utils.io import write_jsonl
from ragprobe.utils.logging import log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--split", required=True, help="cache split name ({split}.pt)")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    gpu_id = cfg.get("gpu_id")
    if gpu_id is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(cfg["checkpoint_path"], map_location="cpu", weights_only=False)
    names = list(ck["modality_names"])
    is_context = str(ck.get("architecture", "")).startswith("context_")
    cls = FrozenContextQuestionAnswerEncoderFusionProbe if is_context else FrozenQuestionAnswerEncoderFusionProbe
    model = cls(
        encoders={}, modality_names=names,
        encoder_dim=int(ck.get("encoder_dim") or (256 if is_context else 128)),
        hidden_dim=int(ck.get("hidden_dim", 128)),
        dropout=float(ck.get("dropout", 0.1)),
        modality_dropout=float(ck.get("modality_dropout", 0.25)),
    ).to(device)
    missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
    bad = [k for k in missing if not k.startswith("encoders.")]
    if bad or unexpected:
        raise RuntimeError(f"fusion head state mismatch: missing={bad} unexpected={list(unexpected)}")
    model.eval()

    cache = torch.load(Path(args.cache) / f"{args.split}.pt", map_location="cpu", weights_only=False)
    miss = [n for n in names if n not in cache["vecs"]]
    if miss:
        raise RuntimeError(f"cache {args.split}.pt lacks branches {miss}")
    vecs = {n: cache["vecs"][n].to(device) for n in names}
    logits = {n: cache["logits"][n].to(device) for n in names}
    labels = cache["labels"].reshape(-1).long().tolist()
    cache_ids = list(cache["example_ids"])

    data_cfg = cfg["data"]
    meta = load_feature_meta(data_cfg["meta_paths"], splits=data_cfg.get("splits"),
                             partition=data_cfg.get("partition"))
    if len(meta) != len(cache_ids):
        raise RuntimeError(f"meta N={len(meta)} != cache N={len(cache_ids)}")
    for i in (0, len(meta) // 2, len(meta) - 1):  # spot-check order alignment
        if _get_example_id(meta[i]) != cache_ids[i]:
            raise RuntimeError(f"cache/meta example order mismatch at {i}")

    with torch.no_grad():
        out = model.forward_from_vecs(vecs, logits, return_multitask=True)
        probs = torch.sigmoid(out["binary_logits"]).cpu().numpy().reshape(-1)
        modality_probs = torch.sigmoid(out["modality_logits"]).cpu().numpy()

    binary_target = ck.get("binary_target_name",
                           "with_context_correct" if is_context else "without_context_correct")
    records = []
    for i, item in enumerate(meta):
        example = item.get("example", {})
        rec = {
            "example_id": example.get("id") or item.get("example_id"),
            "dataset": example.get("dataset", item.get("dataset")),
            "split": example.get("split", item.get("split")),
            "question": example.get("question", item.get("question")),
            "answers": example.get("answers", item.get("answers")),
            "transition": item.get("derived", {}).get("transition", item.get("transition")),
            "label": int(labels[i]),
            "binary_target": binary_target,
            "binary_probability": float(probs[i]),
            "helpful_probability": float(probs[i]),
            "prediction": int(probs[i] >= 0.5),
            "modality_probabilities": {n: float(modality_probs[i, j]) for j, n in enumerate(names)},
            "source_meta": item,
        }
        if is_context:
            rec["with_context_correct_probability"] = float(probs[i])
            rec["with_context_correct_label"] = int(labels[i])
        else:
            rec["without_context_correct_probability"] = float(probs[i])
        records.append(rec)

    y_score = [float(p) for p in probs]
    metrics = compute_metrics(labels, y_score)
    output_cfg = cfg.get("output", {})
    prediction_path = output_cfg.get("prediction_path")
    if prediction_path:
        Path(prediction_path).parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(records, prediction_path)
        _write_roc_outputs(labels, y_score, output_cfg)
    log(f"Eval metrics: {metrics}")
    if prediction_path:
        log(f"Predictions saved to {prediction_path}")


if __name__ == "__main__":
    main()
