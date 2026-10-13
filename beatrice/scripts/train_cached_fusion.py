"""Fast frozen-fusion trainer over precomputed branch vectors.

Trains exactly the same head as the slow path (FrozenContextQuestionAnswerEncoderFusionProbe.
final, via forward_from_vecs) but consumes the ~280MB cache from
precompute_frozen_fusion_vecs.py instead of shipping 8GB raw-feature batches each epoch, so a
run takes seconds. Saves a checkpoint in the identical format the slow trainer produces, so
scripts/evaluate_probe.py works unchanged. LOO variants automatically use their branch subset
(model.modality_names) of the shared full-branch cache.

Usage:
  CUDA_VISIBLE_DEVICES=<gpu> python scripts/train_cached_fusion.py \
     --config configs/experiment/probe/ctx_L17rd_attn/train_fusion_ctxL17rd_attn_no_hidden.yaml \
     --cache outputs/cached_fusion_vecs/ctx_L17rd_attn
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from ragprobe.training.trainer import (
    _build_context_question_answer_frozen_fusion,
    _build_question_answer_frozen_fusion,
    compute_metrics,
    extract_qwen_embeddings,
)
from ragprobe.utils.config import load_yaml
from ragprobe.utils.logging import log


def _load_cache(cache_dir: Path, which: str, names, device):
    d = torch.load(cache_dir / f"{which}.pt", map_location="cpu", weights_only=False)
    missing = [n for n in names if n not in d["vecs"]]
    if missing:
        raise ValueError(f"cache {which}.pt missing branches {missing}; precompute from a FULL config")
    vecs = {n: d["vecs"][n].to(device) for n in names}
    logits = {n: d["logits"][n].to(device) for n in names}
    labels = d["labels"].reshape(-1).to(device=device, dtype=torch.float32)
    return vecs, logits, labels, d["example_ids"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache", required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    probe_cfg = cfg["probe"]
    output_cfg = cfg["output"]

    gpu_id = cfg.get("gpu_id")
    if gpu_id is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    embed_weights = extract_qwen_embeddings(cfg["model_name"])
    architecture = probe_cfg.get("architecture", "context_question_answer_frozen_encoder_fusion")
    is_context = architecture.startswith("context_")
    builder = _build_context_question_answer_frozen_fusion if is_context else _build_question_answer_frozen_fusion
    model, branch_specs, _ = builder(cfg, embed_weights, device)
    names = model.modality_names
    label_mode = probe_cfg.get("label_mode", "with_context_correct" if is_context else "without_context_correct")
    binary_target_name = "with_context_correct" if is_context else "without_context_correct"
    log(f"Cached fusion branches: {names}")

    cache_dir = Path(args.cache)
    tr_vecs, tr_logits, tr_labels, _ = _load_cache(cache_dir, "train", names, device)
    va_vecs, va_logits, va_labels, va_ids = _load_cache(cache_dir, "validation", names, device)
    n_train = tr_labels.shape[0]
    num_pos = int(tr_labels.sum().item())
    pos_weight = (n_train - num_pos) / max(num_pos, 1)
    log(f"Cached fusion label_mode={label_mode}; positives={num_pos}; negatives={n_train-num_pos}; pos_weight={pos_weight:.4f}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable, lr=float(probe_cfg.get("learning_rate", 1e-4)),
                            weight_decay=float(probe_cfg.get("weight_decay", 1e-2)))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=float(probe_cfg.get("lr_scheduler_factor", 0.5)),
        patience=int(probe_cfg.get("lr_scheduler_patience", 4)),
        min_lr=float(probe_cfg.get("min_learning_rate", 1e-7)))
    epochs = int(probe_cfg.get("epochs", 40))
    patience = int(probe_cfg.get("early_stopping_patience", 10))
    min_delta = float(probe_cfg.get("early_stopping_min_delta", 0.0001))
    batch_size = int(probe_cfg.get("batch_size", 512))

    checkpoint_path = output_cfg["checkpoint_path"]
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    best_auc, best_epoch, no_improve = -1.0, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        total_loss = 0.0
        for s in range(0, n_train, batch_size):
            idx = perm[s:s + batch_size]
            vb = {n: tr_vecs[n][idx] for n in names}
            lb = {n: tr_logits[n][idx] for n in names}
            optimizer.zero_grad(set_to_none=True)
            out = model.forward_from_vecs(vb, lb, return_multitask=True)
            loss = criterion(out["binary_logits"], tr_labels[idx].unsqueeze(-1))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * idx.numel()

        model.eval()
        with torch.no_grad():
            out = model.forward_from_vecs(va_vecs, va_logits, return_multitask=True)
            probs = torch.sigmoid(out["binary_logits"]).cpu().numpy().reshape(-1)
        y_true = va_labels.cpu().numpy().astype(int).tolist()
        metrics = compute_metrics(y_true, [float(p) for p in probs])
        scheduler.step(metrics["auc"])
        log(f"Epoch {epoch:03d} | loss={total_loss/max(n_train,1):.4f} | auc={metrics['auc']:.4f} | "
            f"f1={metrics['f1']:.4f} | acc={metrics['accuracy']:.4f} | lr={optimizer.param_groups[0]['lr']:.2e}")

        if metrics["auc"] > best_auc + min_delta:
            best_auc, best_epoch, no_improve = metrics["auc"], epoch, 0
            torch.save({
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items() if not k.startswith("encoders.")},
                "model_name": cfg["model_name"],
                "architecture": architecture,
                "branch_checkpoints": branch_specs,
                "modality_names": names,
                "encoder_dim": getattr(model, "encoder_dim", None),
                "hidden_dim": model.hidden_dim,
                "fusion_num_res_blocks": int(probe_cfg.get("fusion_num_res_blocks", 3)),
                "dropout": float(probe_cfg.get("dropout", 0.1)),
                "modality_dropout": float(probe_cfg.get("modality_dropout", 0.25)),
                "positive_transition": probe_cfg.get("positive_transition", "NY"),
                "label_mode": label_mode,
                "binary_target_name": binary_target_name,
                "best_auc": best_auc,
                "best_epoch": best_epoch,
            }, checkpoint_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                log(f"Early stopping at epoch {epoch} (no improvement for {patience})")
                break

    log(f"Done. Best epoch={best_epoch}; best AUC={best_auc:.4f}")


if __name__ == "__main__":
    main()
