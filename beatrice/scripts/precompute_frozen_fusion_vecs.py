"""Precompute frozen-branch pooled vectors for fast fusion training.

The frozen late-fusion probe only feeds each branch's 256-d pooled `encode()` output (and
its scalar branch logit) into the trainable head. Since the branch encoders are frozen and
deterministic, we can run them over the data ONCE and cache those vectors, turning every
subsequent fusion/LOO epoch from ~70 min (shipping 8GB raw-feature batches) into seconds.

Given a *full* fusion config (all branch_checkpoints), this writes per-split caches:
  {out_dir}/{train,validation}.pt = {
     example_ids, modality_names, labels, rag_class, aux,
     vecs:   {branch: FloatTensor[N,256]},
     logits: {branch: FloatTensor[N,1]},
  }
LOO variants (no_hidden, ...) reuse this same cache by selecting their branch subset.

Usage:
  CUDA_VISIBLE_DEVICES=<gpu> python scripts/precompute_frozen_fusion_vecs.py \
     --config configs/experiment/probe/ctx_L17rd_attn/train_fusion_ctxL17rd_attn.yaml \
     --out outputs/cached_fusion_vecs/ctx_L17rd_attn
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from ragprobe.training.dataset import _get_example_id, load_feature_meta
from ragprobe.training.trainer import (
    _build_context_question_answer_frozen_fusion,
    _build_question_answer_frozen_fusion,
    _context_dataset_from_encoder_ckpt,
    _move_nested,
    _question_dataset_from_encoder_ckpt,
    extract_qwen_embeddings,
)
from ragprobe.utils.config import load_yaml


def _load_meta(data_cfg, which):
    return load_feature_meta(
        data_cfg[f"{which}_meta_paths"],
        splits=data_cfg.get(f"{which}_splits"),
        partition=data_cfg.get(f"{which}_partition"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="a FULL fusion config (all branch_checkpoints)")
    ap.add_argument("--out", required=True, help="output cache directory")
    ap.add_argument("--splits", default="train,validation")
    ap.add_argument("--eval-config", action="append", default=[],
                    help="name=path of an eval config; caches that config's data.meta_paths as {name}.pt (e.g. test=configs/.../eval_test.yaml)")
    ap.add_argument("--side", choices=("context", "question"), default="context",
                    help="which frozen-fusion family the branch checkpoints belong to")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    gpu_id = cfg.get("gpu_id")
    if gpu_id is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = cfg["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    embed_weights = extract_qwen_embeddings(model_name)

    if args.side == "context":
        model, specs, ckpts = _build_context_question_answer_frozen_fusion(cfg, embed_weights, device)
        dataset_factory = _context_dataset_from_encoder_ckpt
        n_inputs, label_i = 15, 15   # 15 inputs, then label / rag / aux / idx
        branch_logit = lambda enc, v: enc.binary_head(v)  # noqa: E731
    else:
        model, specs, ckpts = _build_question_answer_frozen_fusion(cfg, embed_weights, device)
        dataset_factory = _question_dataset_from_encoder_ckpt
        n_inputs, label_i = 10, 10   # 10 inputs, then label / idx
        branch_logit = lambda enc, v: enc.fusion[-1](v)  # noqa: E731
    model.eval()
    names = [s["name"] for s in specs]
    probe_cfg = cfg["probe"]
    pos_trans = probe_cfg.get("positive_transition", "NY")
    label_mode = probe_cfg.get("label_mode",
                               "with_context_correct" if args.side == "context" else "without_context_correct")
    print(f"[precompute] side={args.side} branches={names} device={device} encoder_dim={model.encoder_dim}", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, list]] = []
    for which in [s.strip() for s in args.splits.split(",") if s.strip()]:
        jobs.append((which, _load_meta(cfg["data"], which)))
    for spec in args.eval_config:
        name, path = spec.split("=", 1)
        ec = load_yaml(path)["data"]
        jobs.append((name, load_feature_meta(ec["meta_paths"], splits=ec.get("splits"),
                                             partition=ec.get("partition"))))

    for which, meta in jobs:
        loaders = {
            n: DataLoader(
                dataset_factory(meta, tokenizer, ckpts[n], pos_trans, label_mode),
                batch_size=args.batch_size, shuffle=False, num_workers=0)
            for n in names
        }
        vecs = {n: [] for n in names}
        logits = {n: [] for n in names}
        labels, rag, aux, eids = [], [], [], []
        n_batches = len(loaders[names[0]])
        for bi, batches in enumerate(zip(*[loaders[n] for n in names])):
            for k, n in enumerate(names):
                b = batches[k]
                inp = _move_nested(b[:n_inputs], device)  # float16 hidden -> float32, same as slow path
                with torch.no_grad():
                    v = model.encoders[n].encode(*inp)
                    lg = branch_logit(model.encoders[n], v)
                vecs[n].append(v.float().cpu())
                logits[n].append(lg.float().cpu())
            b0 = batches[0]
            labels.append(b0[label_i].reshape(-1).float())
            if args.side == "context":
                rag.append(b0[16].reshape(-1).long())
                aux.append(b0[17].float())
                idx_col = b0[18]
            else:
                idx_col = b0[11]
            for mi in idx_col.tolist():
                eids.append(_get_example_id(meta[int(mi)]))
            if bi % 100 == 0:
                print(f"  [{which}] batch {bi}/{n_batches}", flush=True)
        out = {
            "example_ids": eids,
            "modality_names": names,
            "labels": torch.cat(labels),
            "vecs": {n: torch.cat(vecs[n]) for n in names},
            "logits": {n: torch.cat(logits[n]) for n in names},
        }
        if rag:
            out["rag_class"] = torch.cat(rag)
            out["aux"] = torch.cat(aux)
        torch.save(out, out_dir / f"{which}.pt")
        print(f"[precompute] wrote {out_dir/f'{which}.pt'}  N={len(eids)}", flush=True)
    print("PRECOMPUTE_DONE", flush=True)


if __name__ == "__main__":
    main()
