import gc
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from ragprobe.training import packed_store
from ragprobe.training.dataset import (
    RAG_CLASS_IGNORE_INDEX,
    RAG_SIGNAL_CLASSES,
    RAGFeatureDataset,
    QuestionAnswerSidecarDataset,
    ContextQuestionAnswerSidecarDataset,
    PairedQuestionContextSidecarDataset,
    fast_collate,
    _get_example_id,
    load_feature_meta,
    load_rag_signal_label_map,
    get_rag_signal_label,
    get_aux_binary_labels,
    get_binary_label,
    get_helpful_label,
    get_without_context_correct,
    get_with_context_correct,
    _get_transition,
)
from ragprobe.training.model import ContextQuestionAnswerModalityEnsembleProbe, ContextQuestionAnswerModalityGatedProbe, ContextQuestionAnswerResNetProbe, PairedQuestionContextResNetProbe, QuestionAnswerDualProbeCNN, FrozenQuestionAnswerEncoderFusionProbe, FrozenContextQuestionAnswerEncoderFusionProbe, FrozenQuestionAnswerTokenFusionProbe, QuestionAnswerModalityEnsembleProbe, QuestionAnswerModalityGatedProbe, QuestionAnswerResNetProbe, RAGProbeCNN
from ragprobe.utils.io import write_jsonl
from ragprobe.utils.logging import log



def _load_label_override(path):
    """jsonl {id, is_correct} -> {id: 0.0/1.0}; used by aligned-label experiments."""
    if not path:
        return None
    import json as _json
    out = {}
    with open(path) as f:
        for line in f:
            r = _json.loads(line)
            out[str(r["id"])] = 1.0 if r["is_correct"] else 0.0
    return out

def extract_qwen_embeddings(model_name):
    log(f"Extracting embedding weights from {model_name}")
    temp_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    weights = temp_model.get_input_embeddings().weight.detach().float().clone()
    del temp_model
    gc.collect()
    torch.cuda.empty_cache()
    return weights


def infer_num_attn_channels(meta_records, fallback=128):
    for item in meta_records:
        rag = item.get("feature_extraction", {}).get("rag", {})
        if "num_attention_channels" in rag:
            return int(rag["num_attention_channels"])
        feature_names = rag.get("feature_names") or []
        if feature_names:
            count = 0
            for name in feature_names:
                name = str(name)
                if "_attn_" in name or name.startswith("attn_"):
                    count += 1
                    continue
                break
            if count > 0:
                return int(count)
        if "num_feature_channels" in rag:
            # Legacy tensors saved [attention + p + entropy]. New tensors should
            # expose feature_names, so this path remains only for old metadata.
            return int(rag["num_feature_channels"]) - 2

    return fallback


def compute_metrics(y_true, y_score, threshold=0.5):
    y_pred = [1 if p >= threshold else 0 for p in y_score]
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred) if y_true else 0.0,
        "precision": precision_score(y_true, y_pred, zero_division=0) if y_true else 0.0,
        "recall": recall_score(y_true, y_pred, zero_division=0) if y_true else 0.0,
        "f1": f1_score(y_true, y_pred, zero_division=0) if y_true else 0.0,
    }
    try:
        metrics["auc"] = roc_auc_score(y_true, y_score)
    except ValueError:
        metrics["auc"] = 0.0
    return metrics


def compute_classification_metrics(y_true, y_pred):
    true_counter = Counter(y_true)
    pred_counter = Counter(y_pred)
    if not y_true:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "true_distribution": {},
            "pred_distribution": {},
        }

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "true_distribution": dict(true_counter),
        "pred_distribution": dict(pred_counter),
    }


def _transition_distribution(records):
    c = Counter()
    for item in records:
        c[_get_transition(item) or "UNKNOWN"] += 1
    return dict(c)


def _count_positive(records, positive_transition="NY", label_mode="transition_positive"):
    return sum(
        1
        for item in records
        if get_binary_label(item, positive_transition=positive_transition, label_mode=label_mode) > 0.5
    )


def _aux_binary_pos_weight(records):
    labels = torch.tensor([get_aux_binary_labels(item) for item in records], dtype=torch.float32)
    pos = labels.sum(dim=0).clamp_min(1.0)
    neg = (labels.size(0) - labels.sum(dim=0)).clamp_min(1.0)
    return neg / pos


def _rag_class_label_distribution(records, rag_class_labels, regression_transition="YN"):
    counter = Counter()
    for item in records:
        label = rag_class_labels.get(
            _get_example_id(item),
            get_rag_signal_label(item, regression_transition=regression_transition),
        )
        counter[int(label)] += 1
    return counter


def _make_rag_class_weights(label_counts, mode="sqrt_inverse", max_weight=10.0):
    if mode in {None, "none", "off", False}:
        return None

    valid_counts = {
        label: count
        for label, count in label_counts.items()
        if label != RAG_CLASS_IGNORE_INDEX and count > 0
    }
    if not valid_counts:
        return None

    total = sum(valid_counts.values())
    weights = torch.ones(len(RAG_SIGNAL_CLASSES), dtype=torch.float32)
    for label, count in valid_counts.items():
        if mode == "inverse":
            weight = total / (len(valid_counts) * count)
        elif mode == "sqrt_inverse":
            weight = (total / count) ** 0.5
        else:
            raise ValueError(f"Unknown rag_class_weighting mode: {mode}")
        weights[label] = min(float(weight), float(max_weight))

    weights = weights / weights.mean().clamp_min(1e-6)
    return weights


def _named_label_distribution(label_counts):
    out = {}
    for label, count in sorted(label_counts.items()):
        if label == RAG_CLASS_IGNORE_INDEX:
            name = "IGNORE"
        else:
            name = RAG_SIGNAL_CLASSES[int(label)]
        out[name] = int(count)
    return out


def _named_pred_distribution(distribution):
    return {
        RAG_SIGNAL_CLASSES[int(label)]: int(count)
        for label, count in sorted(distribution.items())
    }



def _example_id(item):
    return (item.get("example") or {}).get("id") or _get_example_id(item)


def _require_question_sidecars(records, split_name: str):
    # packed fast path covers every record -> sidecar files are not needed on disk
    if packed_store.enabled() and packed_store.resolve(records) is not None:
        return
    missing = []
    wrong_version = []
    for i, item in enumerate(records[:200]):
        rag = item.get("feature_extraction", {}).get("rag", {})
        sidecar = rag.get("sidecar_path")
        if not sidecar:
            missing.append((i, _example_id(item), rag.get("extractor_version"), rag.get("target_type")))
        elif not Path(sidecar).exists():
            missing.append((i, _example_id(item), rag.get("extractor_version"), sidecar))
        if rag.get("extractor_version") not in {"unified_answer_span_v1", "unified_answer_span_v2"}:
            wrong_version.append((i, _example_id(item), rag.get("extractor_version")))
    if missing:
        preview = "; ".join(str(x) for x in missing[:5])
        raise RuntimeError(
            f"QuestionAnswerResNet requires new question-only features with sidecar_path. "
            f"{split_name} has records without usable sidecars. Preview: {preview}. "
            "Re-run question-only feature extraction with the current extractor first, e.g. "
            "python scripts/extract_features.py --config configs/experiment/features/question/musique_qwen9b_question_features.yaml "
            "and the corresponding hotpotqa/twowiki question configs."
        )
    if wrong_version:
        preview = "; ".join(str(x) for x in wrong_version[:5])
        raise RuntimeError(
            f"QuestionAnswerResNet requires extractor_version=unified_answer_span_v1/v2. "
            f"{split_name} has older feature metadata. Preview: {preview}. "
            "Re-run question-only feature extraction with the current extractor."
        )


def _train_question_answer_resnet(config: dict, train_meta, val_meta, tokenizer, embed_weights):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    probe_cfg = config["probe"]
    model_name = config["model_name"]
    label_mode = probe_cfg.get("label_mode", "without_context_correct")
    binary_target_name = "without_context_correct" if label_mode == "without_context_correct" else "helpful"
    positive_transition = probe_cfg.get("positive_transition", "NY")

    _require_question_sidecars(train_meta, "train")
    _require_question_sidecars(val_meta, "validation")

    qa_ds_kwargs = dict(
        tokenizer=tokenizer,
        question_max_len=probe_cfg.get("question_max_len", probe_cfg.get("max_question_len", 128)),
        answer_max_len=probe_cfg.get("answer_max_len", 32),
        positive_transition=positive_transition,
        label_mode=label_mode,
        use_numeric_features=probe_cfg.get("use_numeric_features", True),
        use_probability_features=probe_cfg.get("use_probability_features", True),
        use_residual_features=probe_cfg.get("use_residual_features", True),
        use_attention_features=probe_cfg.get("use_attention_features", True),
        use_last_hidden=probe_cfg.get("use_last_hidden", True),
        use_attention_mass=probe_cfg.get("use_attention_mass", False),
        ablate_segments=tuple(probe_cfg.get("ablate_segments") or ()),
        label_override=_load_label_override(probe_cfg.get("label_override_path")),
        label_override_required=bool(probe_cfg.get("label_override_required", False)),
    )
    train_ds = QuestionAnswerSidecarDataset(train_meta, **qa_ds_kwargs)
    val_ds = QuestionAnswerSidecarDataset(val_meta, **qa_ds_kwargs)
    sample = train_ds[0]
    numeric_channels = int(sample[0].shape[0])
    attention_channels = int(sample[1].shape[0])
    llm_hidden_dim = int(sample[3].shape[-1])
    log(
        "QuestionAnswerResNet feature dims: "
        f"numeric_channels={numeric_channels}; attention_channels={attention_channels}; llm_hidden_dim={llm_hidden_dim}; "
        f"question_max_len={train_ds.question_max_len}; answer_max_len={train_ds.answer_max_len}"
    )

    _collate = fast_collate if os.environ.get("PROBE_FAST_COLLATE") == "1" else None
    if _collate is not None:
        # batch-direct packed fetch (no-op when PROBE_PACKED_ROOT is off/unresolved)
        train_ds._packed_batch = val_ds._packed_batch = True
    train_loader = DataLoader(
        train_ds,
        batch_size=probe_cfg["batch_size"],
        shuffle=True,
        num_workers=probe_cfg.get("num_workers", 4),
        pin_memory=bool(probe_cfg.get("pin_memory", torch.cuda.is_available())),
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=probe_cfg["batch_size"],
        shuffle=False,
        num_workers=probe_cfg.get("num_workers", 4),
        pin_memory=bool(probe_cfg.get("pin_memory", torch.cuda.is_available())),
        collate_fn=_collate,
    )

    num_pos = _count_positive(train_meta, positive_transition=positive_transition, label_mode=label_mode)
    num_neg = len(train_meta) - num_pos
    pos_weight = num_neg / max(num_pos, 1)
    log(f"QuestionAnswerResNet label_mode={label_mode}; positives={num_pos}; negatives={num_neg}; pos_weight={pos_weight:.4f}")

    qa_architecture = probe_cfg.get("architecture", "question_answer_resnet")
    if qa_architecture == "question_answer_modality_gated":
        qa_model_cls = QuestionAnswerModalityGatedProbe
    elif qa_architecture == "question_answer_modality_ensemble":
        qa_model_cls = QuestionAnswerModalityEnsembleProbe
    else:
        qa_model_cls = QuestionAnswerResNetProbe
    log(f"QuestionAnswer model class: {qa_model_cls.__name__}")
    model = qa_model_cls(
        embed_weights=embed_weights,
        numeric_channels=numeric_channels,
        attention_channels=attention_channels,
        hidden_dim=probe_cfg.get("hidden_dim", 128),
        embed_proj_dim=probe_cfg.get("embed_proj_dim", 32),
        llm_hidden_dim=llm_hidden_dim,
        hidden_proj_dim=probe_cfg.get("hidden_proj_dim", probe_cfg.get("embed_proj_dim", 32)),
        num_res_blocks=probe_cfg.get("num_res_blocks", 3),
        dropout=probe_cfg.get("dropout", 0.1),
        noise_std=probe_cfg.get("noise_std", 0.02),
        use_token_embedding=probe_cfg.get("use_token_embedding", True),
        use_last_hidden=probe_cfg.get("use_last_hidden", True),
        **({"conv_kernel_size": int(probe_cfg.get("conv_kernel_size", 3)),
            "encoder_pool_first": str(probe_cfg.get("encoder_pool_first", "off"))}
           if qa_model_cls is QuestionAnswerResNetProbe else {}),
        **({
            "aux_hidden_dim": probe_cfg.get("aux_hidden_dim"),
            "aux_num_res_blocks": probe_cfg.get("aux_num_res_blocks", 2),
            "prob_feature_channels": probe_cfg.get("prob_feature_channels", 5),
            "use_probability_features": probe_cfg.get("use_probability_features", True),
            "use_residual_features": probe_cfg.get("use_residual_features", True),
            "use_attention_features": probe_cfg.get("use_attention_features", True),
            "modality_dropout": probe_cfg.get("modality_dropout", 0.1),
        } if qa_model_cls in {QuestionAnswerModalityGatedProbe, QuestionAnswerModalityEnsembleProbe} else {}),
    ).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(probe_cfg["learning_rate"]),
        weight_decay=float(probe_cfg.get("weight_decay", 1e-2)),
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    modality_aux_loss_weight = float(probe_cfg.get("modality_aux_loss_weight", 0.3))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(probe_cfg.get("lr_scheduler_factor", 0.5)),
        patience=int(probe_cfg.get("lr_scheduler_patience", 4)),
        min_lr=float(probe_cfg.get("min_learning_rate", 1e-7)),
    )

    checkpoint_path = Path(config["output"]["checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    val_prediction_path = config.get("output", {}).get("val_prediction_path")
    best_auc = -1.0
    best_epoch = 0
    no_improve = 0
    patience = int(probe_cfg.get("early_stopping_patience", 12))
    min_delta = float(probe_cfg.get("early_stopping_min_delta", 1e-4))
    show_progress = bool(probe_cfg.get("show_progress", sys.stderr.isatty()))

    _resid_off = None
    _resid_off_path = probe_cfg.get("residual_offset_path")
    if _resid_off_path:
        import json as _json
        _lut = _json.load(open(_resid_off_path))
        def _off_vec(meta):
            miss = 0; vals = []
            for it in meta:
                eid = _example_id(it)
                if eid in _lut:
                    vals.append(float(_lut[eid]))
                else:
                    miss += 1; vals.append(0.0)
            return torch.tensor(vals, device=device, dtype=torch.float32), miss
        _oft, _mt = _off_vec(train_meta)
        _ofv, _mv = _off_vec(val_meta)
        _resid_off = (_oft, _ofv)
        log(f"[residual-boost] offset from {_resid_off_path}: train_miss={_mt}/{len(train_meta)} val_miss={_mv}/{len(val_meta)}")

    _train_iter = packed_store.maybe_cuda_prefetch(train_loader, train_ds, device)
    for epoch in range(1, int(probe_cfg["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(_train_iter, desc=f"Epoch {epoch:03d} train", leave=False, disable=not show_progress):
            q_num, q_attn, q_tok, q_hidden, q_mask, a_num, a_attn, a_tok, a_hidden, a_mask, targets, _bidx = batch
            q_num = q_num.to(device)
            q_attn = q_attn.to(device)
            q_tok = q_tok.to(device)
            q_hidden = q_hidden.to(device=device, dtype=torch.float32)
            q_mask = q_mask.to(device)
            a_num = a_num.to(device)
            a_attn = a_attn.to(device)
            a_tok = a_tok.to(device)
            a_hidden = a_hidden.to(device=device, dtype=torch.float32)
            a_mask = a_mask.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            if qa_model_cls is QuestionAnswerModalityEnsembleProbe:
                outputs = model(q_num, q_attn, q_tok, q_hidden, q_mask, a_num, a_attn, a_tok, a_hidden, a_mask, return_multitask=True)
                logits = outputs["binary_logits"]
                if _resid_off is not None:
                    logits = logits + _resid_off[0][_bidx.to(device)].unsqueeze(-1)
                modality_targets = targets.expand_as(outputs["modality_logits"])
                modality_loss = criterion(outputs["modality_logits"], modality_targets)
                loss = criterion(logits, targets) + modality_aux_loss_weight * modality_loss
            else:
                logits = model(q_num, q_attn, q_tok, q_hidden, q_mask, a_num, a_attn, a_tok, a_hidden, a_mask)
                if _resid_off is not None:
                    logits = logits + _resid_off[0][_bidx.to(device)].unsqueeze(-1)
                loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        model.eval()
        y_true, y_score = [], []
        val_pred_records = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch:03d} val", leave=False, disable=not show_progress):
                q_num, q_attn, q_tok, q_hidden, q_mask, a_num, a_attn, a_tok, a_hidden, a_mask, targets, indices = batch
                q_num = q_num.to(device)
                q_attn = q_attn.to(device)
                q_tok = q_tok.to(device)
                q_hidden = q_hidden.to(device=device, dtype=torch.float32)
                q_mask = q_mask.to(device)
                a_num = a_num.to(device)
                a_attn = a_attn.to(device)
                a_tok = a_tok.to(device)
                a_hidden = a_hidden.to(device=device, dtype=torch.float32)
                a_mask = a_mask.to(device)
                targets = targets.to(device)
                if qa_model_cls is QuestionAnswerModalityEnsembleProbe:
                    outputs = model(q_num, q_attn, q_tok, q_hidden, q_mask, a_num, a_attn, a_tok, a_hidden, a_mask, return_multitask=True)
                    logits = outputs["binary_logits"]
                    modality_probs = torch.sigmoid(outputs["modality_logits"]).cpu().numpy().tolist()
                    modality_names = list(outputs.get("modality_names", []))
                else:
                    logits = model(q_num, q_attn, q_tok, q_hidden, q_mask, a_num, a_attn, a_tok, a_hidden, a_mask)
                    modality_probs = None
                    modality_names = []
                if _resid_off is not None:
                    logits = logits + _resid_off[1][indices.to(device)].unsqueeze(-1)
                probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy().tolist()
                labels = targets.squeeze(-1).cpu().numpy().tolist()
                if not isinstance(probs, list):
                    probs = [probs]
                if not isinstance(labels, list):
                    labels = [labels]
                y_score.extend(float(x) for x in probs)
                y_true.extend(float(x) for x in labels)
                for prob, label, idx in zip(probs, labels, indices.numpy().tolist()):
                    item = val_meta[int(idx)]
                    ex = item.get("example") or {}
                    val_pred_records.append({
                        "idx": int(idx),
                        "example_id": _example_id(item),
                        "dataset": ex.get("dataset"),
                        "split": ex.get("split"),
                        "question": ex.get("question"),
                        "transition": _get_transition(item),
                        "label": int(label),
                        "without_context_correct_probability": float(prob),
                        "helpful_probability": float(prob),
                        "prediction": int(float(prob) >= 0.5),
                    })
        metrics = compute_metrics(y_true, y_score)
        avg_loss = total_loss / max(len(train_loader), 1)
        log(f"Epoch {epoch:03d} | loss={avg_loss:.4f} | auc={metrics['auc']:.4f} | f1={metrics['f1']:.4f} | acc={metrics['accuracy']:.4f} | lr={optimizer.param_groups[0]['lr']:.2e}")
        scheduler.step(metrics["auc"])

        if metrics["auc"] > best_auc + min_delta:
            best_auc = metrics["auc"]
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "state_dict": model.state_dict(),
                "model_name": model_name,
                "architecture": qa_architecture,
                "numeric_channels": numeric_channels,
                "attention_channels": attention_channels,
                "hidden_dim": probe_cfg.get("hidden_dim", 128),
                "embed_proj_dim": probe_cfg.get("embed_proj_dim", 32),
                "llm_hidden_dim": llm_hidden_dim,
                "hidden_proj_dim": probe_cfg.get("hidden_proj_dim", probe_cfg.get("embed_proj_dim", 32)),
                "num_res_blocks": probe_cfg.get("num_res_blocks", 3),
                "dropout": probe_cfg.get("dropout", 0.1),
                "question_max_len": train_ds.question_max_len,
                "answer_max_len": train_ds.answer_max_len,
                "positive_transition": positive_transition,
                "label_mode": label_mode,
                "binary_target_name": binary_target_name,
                "use_numeric_features": probe_cfg.get("use_numeric_features", True),
                "use_probability_features": probe_cfg.get("use_probability_features", True),
                "use_residual_features": probe_cfg.get("use_residual_features", True),
                "use_attention_features": probe_cfg.get("use_attention_features", True),
                "use_token_embedding": probe_cfg.get("use_token_embedding", True),
                "use_last_hidden": probe_cfg.get("use_last_hidden", True),
                "use_attention_mass": probe_cfg.get("use_attention_mass", False),
                "ablate_segments": list(probe_cfg.get("ablate_segments") or []),
                "conv_kernel_size": int(probe_cfg.get("conv_kernel_size", 3)),
                "encoder_pool_first": str(probe_cfg.get("encoder_pool_first", "off")),
                "aux_hidden_dim": probe_cfg.get("aux_hidden_dim"),
                "aux_num_res_blocks": probe_cfg.get("aux_num_res_blocks", 2),
                "prob_feature_channels": probe_cfg.get("prob_feature_channels", 5),
                "modality_dropout": probe_cfg.get("modality_dropout", 0.1),
                "modality_aux_loss_weight": modality_aux_loss_weight,
                "best_auc": best_auc,
                "best_epoch": best_epoch,
            }, checkpoint_path)
            log(f"Saved new best checkpoint: {checkpoint_path}")
            if val_prediction_path:
                write_jsonl(val_pred_records, val_prediction_path)
        else:
            no_improve += 1
            log(f"No AUC improvement for {no_improve}/{patience}; best_epoch={best_epoch}, best_auc={best_auc:.4f}")
        if patience > 0 and no_improve >= patience:
            log(f"Early stopping at epoch={epoch}; best_epoch={best_epoch}; best_auc={best_auc:.4f}")
            break
    log(f"Done. Best epoch={best_epoch}; best AUC={best_auc:.4f}")


def _require_context_sidecars(records, split_name: str):
    # packed fast path covers every record -> sidecar files are not needed on disk
    if packed_store.enabled() and packed_store.resolve(records) is not None:
        return
    missing = []
    wrong = []
    for i, item in enumerate(records[:200]):
        rag = item.get("feature_extraction", {}).get("rag", {})
        sidecar = rag.get("sidecar_path")
        target_type = rag.get("target_type")
        if not sidecar or not Path(sidecar).exists():
            missing.append((i, _example_id(item), rag.get("extractor_version"), sidecar))
        if rag.get("extractor_version") not in {"unified_answer_span_v1", "unified_answer_span_v2"} or target_type == "question":
            wrong.append((i, _example_id(item), rag.get("extractor_version"), target_type))
    if missing:
        preview = "; ".join(str(x) for x in missing[:5])
        raise RuntimeError(
            f"ContextQuestionAnswerResNet requires new with-context features with sidecar_path. "
            f"{split_name} has missing sidecars. Preview: {preview}. Re-run with-context feature extraction first."
        )
    if wrong:
        preview = "; ".join(str(x) for x in wrong[:5])
        raise RuntimeError(
            f"ContextQuestionAnswerResNet requires extractor_version=unified_answer_span_v1/v2 and target_type=context. "
            f"{split_name} has incompatible metadata. Preview: {preview}."
        )




def _load_matching_state_dict(model, checkpoint_state_dict):
    model_state = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in checkpoint_state_dict.items():
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape):
            compatible[key] = value
        else:
            skipped.append(key)
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    return missing, unexpected, skipped


def _build_question_answer_encoder(checkpoint_path: str, embed_weights, device: str, noise_std: float = 0.0):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    qa_architecture = ckpt.get("architecture", "question_answer_resnet")
    if qa_architecture == "question_answer_modality_gated":
        qa_model_cls = QuestionAnswerModalityGatedProbe
    elif qa_architecture == "question_answer_modality_ensemble":
        qa_model_cls = QuestionAnswerModalityEnsembleProbe
    else:
        qa_model_cls = QuestionAnswerResNetProbe
    log(f"QuestionAnswer encoder class: {qa_model_cls.__name__}")
    model = qa_model_cls(
        embed_weights=embed_weights,
        numeric_channels=int(ckpt["numeric_channels"]),
        attention_channels=int(ckpt["attention_channels"]),
        hidden_dim=int(ckpt.get("hidden_dim", 128)),
        embed_proj_dim=int(ckpt.get("embed_proj_dim", 32)),
        llm_hidden_dim=int(ckpt.get("llm_hidden_dim", embed_weights.shape[1])),
        hidden_proj_dim=int(ckpt.get("hidden_proj_dim", ckpt.get("embed_proj_dim", 32))),
        num_res_blocks=int(ckpt.get("num_res_blocks", 3)),
        dropout=float(ckpt.get("dropout", 0.1)),
        noise_std=noise_std,
        use_token_embedding=ckpt.get("use_token_embedding", False),
        use_last_hidden=ckpt.get("use_last_hidden", True),
        **({
            "aux_hidden_dim": ckpt.get("aux_hidden_dim"),
            "aux_num_res_blocks": ckpt.get("aux_num_res_blocks", 2),
            "prob_feature_channels": ckpt.get("prob_feature_channels", 5),
            "use_probability_features": ckpt.get("use_probability_features", True),
            "use_residual_features": ckpt.get("use_residual_features", True),
            "use_attention_features": ckpt.get("use_attention_features", True),
            "modality_dropout": ckpt.get("modality_dropout", 0.0),
        } if qa_model_cls in {QuestionAnswerModalityGatedProbe, QuestionAnswerModalityEnsembleProbe} else {}),
        **({"conv_kernel_size": int(ckpt.get("conv_kernel_size", 3)),
            "encoder_pool_first": str(ckpt.get("encoder_pool_first", "off"))}
           if qa_model_cls is QuestionAnswerResNetProbe else {}),
    ).to(device)
    missing, unexpected, skipped = _load_matching_state_dict(model, ckpt["state_dict"])
    log(f"Loaded question encoder from {checkpoint_path}; missing={len(missing)} unexpected={len(unexpected)} skipped={len(skipped)}")
    return model, ckpt, int(ckpt.get("hidden_dim", 128))


def _build_context_question_answer_encoder(checkpoint_path: str, embed_weights, device: str, noise_std: float = 0.0):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    classes = tuple(ckpt.get("rag_signal_classes", RAG_SIGNAL_CLASSES))
    cqa_architecture = ckpt.get("architecture", "context_question_answer_resnet")
    if cqa_architecture == "context_question_answer_modality_gated":
        cqa_model_cls = ContextQuestionAnswerModalityGatedProbe
    elif cqa_architecture == "context_question_answer_modality_ensemble":
        cqa_model_cls = ContextQuestionAnswerModalityEnsembleProbe
    else:
        cqa_model_cls = ContextQuestionAnswerResNetProbe
    log(f"ContextQuestionAnswer encoder class: {cqa_model_cls.__name__}")
    model = cqa_model_cls(
        embed_weights=embed_weights,
        numeric_channels=int(ckpt["numeric_channels"]),
        attention_channels=int(ckpt["attention_channels"]),
        hidden_dim=int(ckpt.get("hidden_dim", 128)),
        embed_proj_dim=int(ckpt.get("embed_proj_dim", 32)),
        llm_hidden_dim=int(ckpt.get("llm_hidden_dim", embed_weights.shape[1])),
        hidden_proj_dim=int(ckpt.get("hidden_proj_dim", ckpt.get("embed_proj_dim", 32))),
        num_res_blocks=int(ckpt.get("num_res_blocks", 3)),
        dropout=float(ckpt.get("dropout", 0.1)),
        noise_std=noise_std,
        use_token_embedding=ckpt.get("use_token_embedding", False),
        use_last_hidden=ckpt.get("use_last_hidden", True),
        num_rag_classes=len(classes),
        **({
            "aux_hidden_dim": ckpt.get("aux_hidden_dim"),
            "aux_num_res_blocks": ckpt.get("aux_num_res_blocks", 2),
            "prob_feature_channels": ckpt.get("prob_feature_channels", 5),
            "use_probability_features": ckpt.get("use_probability_features", True),
            "use_residual_features": ckpt.get("use_residual_features", True),
            "use_attention_features": ckpt.get("use_attention_features", True),
            "modality_dropout": ckpt.get("modality_dropout", 0.0),
        } if cqa_model_cls in {ContextQuestionAnswerModalityGatedProbe, ContextQuestionAnswerModalityEnsembleProbe} else {}),
        **({"conv_kernel_size": int(ckpt.get("conv_kernel_size", 3))}
           if cqa_model_cls is ContextQuestionAnswerResNetProbe else {}),
    ).to(device)
    missing, unexpected, skipped = _load_matching_state_dict(model, ckpt["state_dict"])
    log(f"Loaded context encoder from {checkpoint_path}; missing={len(missing)} unexpected={len(unexpected)} skipped={len(skipped)}")
    return model, ckpt, int(ckpt.get("hidden_dim", 128)) * 2


def _move_nested(batch, device):
    out = []
    for x in batch:
        if torch.is_tensor(x):
            if torch.is_floating_point(x):
                # Transfer first (keeping the source dtype, e.g. fp16), then cast to
                # fp32 ON THE TARGET DEVICE. A direct `.to(device, dtype=float32)` on a
                # CPU->CUDA move casts fp16->fp32 on the CPU first (single-threaded when
                # BLAS/OMP threads are capped), which turns a 1GB hidden tensor into a
                # multi-minute stall under CPU contention. Casting on-device is identical
                # numerically (fp16->fp32 is exact) and independent of CPU thread count.
                out.append(x.to(device=device).to(torch.float32))
            else:
                out.append(x.to(device))
        else:
            out.append(x)
    return tuple(out)


def _move_branch_batches(branch_batches, device):
    return {name: _move_nested(batch, device) for name, batch in branch_batches.items()}


def _question_dataset_from_encoder_ckpt(meta, tokenizer, ckpt, positive_transition, label_mode):
    return QuestionAnswerSidecarDataset(
        meta,
        tokenizer,
        question_max_len=int(ckpt.get("question_max_len", 128)),
        answer_max_len=int(ckpt.get("answer_max_len", 32)),
        positive_transition=positive_transition,
        label_mode=label_mode,
        use_numeric_features=bool(ckpt.get("use_numeric_features", True)),
        use_probability_features=bool(ckpt.get("use_probability_features", True)),
        use_residual_features=bool(ckpt.get("use_residual_features", True)),
        use_attention_features=bool(ckpt.get("use_attention_features", True)),
        use_last_hidden=bool(ckpt.get("use_last_hidden", True)),
        use_attention_mass=bool(ckpt.get("use_attention_mass", False)),
        ablate_segments=tuple(ckpt.get("ablate_segments") or ()),
    )


class QuestionAnswerFrozenFusionDataset(Dataset):
    def __init__(self, branch_datasets):
        self.branch_datasets = dict(branch_datasets)
        if not self.branch_datasets:
            raise ValueError("branch_datasets must not be empty")
        lengths = {name: len(ds) for name, ds in self.branch_datasets.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"branch datasets have different lengths: {lengths}")
        self.names = list(self.branch_datasets)
        self.length = next(iter(lengths.values()))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        branches = {}
        label = None
        item_idx = None
        for name in self.names:
            sample = self.branch_datasets[name][idx]
            branches[name] = sample[:10]
            if label is None:
                label = sample[10]
                item_idx = sample[11]
        return branches, label, item_idx


def _context_dataset_from_encoder_ckpt(meta, tokenizer, ckpt, positive_transition, label_mode):
    return ContextQuestionAnswerSidecarDataset(
        meta,
        tokenizer,
        context_max_len=int(ckpt.get("context_max_len", ckpt.get("max_context_len", 2048))),
        question_max_len=int(ckpt.get("question_max_len", 128)),
        answer_max_len=int(ckpt.get("answer_max_len", 32)),
        positive_transition=positive_transition,
        label_mode=label_mode,
        rag_class_labels_by_id={},
        regression_transition=ckpt.get("rag_class_regression_transition", "YN"),
        use_numeric_features=bool(ckpt.get("use_numeric_features", True)),
        use_probability_features=bool(ckpt.get("use_probability_features", True)),
        use_residual_features=bool(ckpt.get("use_residual_features", True)),
        use_attention_features=bool(ckpt.get("use_attention_features", True)),
        use_last_hidden=bool(ckpt.get("use_last_hidden", True)),
        use_attention_mass=bool(ckpt.get("use_attention_mass", False)),
        ablate_segments=tuple(ckpt.get("ablate_segments") or ()),
        ablate_channels=tuple(ckpt.get("ablate_channels") or ()),
    )


class ContextQuestionAnswerFrozenFusionDataset(Dataset):
    def __init__(self, branch_datasets):
        self.branch_datasets = dict(branch_datasets)
        if not self.branch_datasets:
            raise ValueError("branch_datasets must not be empty")
        lengths = {name: len(ds) for name, ds in self.branch_datasets.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"branch datasets have different lengths: {lengths}")
        self.names = list(self.branch_datasets)
        self.length = next(iter(lengths.values()))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        branches = {}
        label = None
        item_idx = None
        for name in self.names:
            sample = self.branch_datasets[name][idx]
            branches[name] = sample[:15]
            if label is None:
                label = sample[15]
                item_idx = sample[18]
        return branches, label, item_idx


def _question_fusion_branch_specs(probe_cfg):
    specs = probe_cfg.get("branch_checkpoints") or []
    if isinstance(specs, dict):
        specs = [{"name": name, "checkpoint_path": path} for name, path in specs.items()]
    out = []
    for spec in specs:
        name = str(spec.get("name") or Path(spec["checkpoint_path"]).parent.name).replace("only_", "")
        out.append({"name": name, "checkpoint_path": str(spec["checkpoint_path"])})
    if not out:
        raise ValueError("probe.branch_checkpoints is required for question_answer_frozen_encoder_fusion")
    return out



def _question_fusion_needs_embedding_weights(branch_specs):
    for spec in branch_specs:
        ckpt = torch.load(spec["checkpoint_path"], map_location="cpu", weights_only=False)
        if bool(ckpt.get("use_token_embedding", False)):
            return True
    return False


def _embedding_weights_for_question_fusion(model_name, branch_specs):
    if _question_fusion_needs_embedding_weights(branch_specs):
        return extract_qwen_embeddings(model_name)
    log("Frozen fusion branches do not use token embeddings; using dummy embedding weights")
    return torch.empty(1, 1)

def _build_question_answer_frozen_fusion(config, embed_weights, device):
    probe_cfg = config["probe"]
    specs = _question_fusion_branch_specs(probe_cfg)
    encoders = {}
    ckpts = {}
    encoder_dim = None
    for spec in specs:
        model, ckpt, _ = _build_question_answer_encoder(spec["checkpoint_path"], embed_weights, device, noise_std=0.0)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        encoders[spec["name"]] = model
        ckpts[spec["name"]] = ckpt
        dim = int(ckpt.get("hidden_dim", 128))
        if encoder_dim is None:
            encoder_dim = dim
        elif encoder_dim != dim:
            raise ValueError(f"All branch encoders must expose the same encode dim; got {encoder_dim} and {dim}")
    model = FrozenQuestionAnswerEncoderFusionProbe(
        encoders=encoders,
        modality_names=[spec["name"] for spec in specs],
        encoder_dim=int(encoder_dim or probe_cfg.get("hidden_dim", 128)),
        hidden_dim=int(probe_cfg.get("fusion_hidden_dim", probe_cfg.get("hidden_dim", 128))),
        dropout=float(probe_cfg.get("dropout", 0.1)),
        modality_dropout=float(probe_cfg.get("modality_dropout", 0.25)),
    ).to(device)
    return model, specs, ckpts


def _build_context_question_answer_frozen_fusion(config, embed_weights, device):
    probe_cfg = config["probe"]
    specs = _question_fusion_branch_specs(probe_cfg)
    encoders, ckpts = {}, {}
    encoder_dim = None
    for spec in specs:
        model, ckpt, dim = _build_context_question_answer_encoder(spec["checkpoint_path"], embed_weights, device, noise_std=0.0)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        encoders[spec["name"]] = model
        ckpts[spec["name"]] = ckpt
        if encoder_dim is None:
            encoder_dim = dim
        elif encoder_dim != dim:
            raise ValueError(f"All context branch encoders must expose the same encode dim; got {encoder_dim} and {dim}")
    model = FrozenContextQuestionAnswerEncoderFusionProbe(
        encoders=encoders,
        modality_names=[spec["name"] for spec in specs],
        encoder_dim=int(encoder_dim or 256),
        hidden_dim=int(probe_cfg.get("fusion_hidden_dim", probe_cfg.get("hidden_dim", 128))),
        dropout=float(probe_cfg.get("dropout", 0.1)),
        modality_dropout=float(probe_cfg.get("modality_dropout", 0.25)),
    ).to(device)
    return model, specs, ckpts


def _build_question_answer_frozen_token_fusion(config, embed_weights, device):
    probe_cfg = config["probe"]
    specs = _question_fusion_branch_specs(probe_cfg)
    encoders = {}
    ckpts = {}
    for spec in specs:
        model, ckpt, _ = _build_question_answer_encoder(spec["checkpoint_path"], embed_weights, device, noise_std=0.0)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        encoders[spec["name"]] = model
        ckpts[spec["name"]] = ckpt
    model = FrozenQuestionAnswerTokenFusionProbe(
        encoders=encoders,
        modality_names=[spec["name"] for spec in specs],
        hidden_dim=int(probe_cfg.get("fusion_hidden_dim", probe_cfg.get("hidden_dim", 128))),
        num_res_blocks=int(probe_cfg.get("fusion_num_res_blocks", 3)),
        dropout=float(probe_cfg.get("dropout", 0.1)),
        modality_dropout=float(probe_cfg.get("modality_dropout", 0.25)),
    ).to(device)
    return model, specs, ckpts


def _binary_pos_weight_from_labels(records, label_fn):
    positives = sum(1 for item in records if label_fn(item))
    negatives = len(records) - positives
    return positives, negatives, negatives / max(positives, 1)


def _train_paired_question_context_resnet(config: dict, train_q_meta, train_c_meta, val_q_meta, val_c_meta, tokenizer, embed_weights):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    probe_cfg = config["probe"]
    output_cfg = config.get("output", {})
    model_name = config["model_name"]

    _require_question_sidecars(train_q_meta, "train question")
    _require_question_sidecars(val_q_meta, "validation question")
    _require_context_sidecars(train_c_meta, "train context")
    _require_context_sidecars(val_c_meta, "validation context")

    ds_kwargs = dict(
        tokenizer=tokenizer,
        question_max_len=probe_cfg.get("question_max_len", 128),
        answer_max_len=probe_cfg.get("answer_max_len", 32),
        context_max_len=probe_cfg.get("context_max_len", probe_cfg.get("max_context_len", 2048)),
        use_attention_mass=probe_cfg.get("use_attention_mass", False),
    )
    train_ds = PairedQuestionContextSidecarDataset(train_q_meta, train_c_meta, **ds_kwargs)
    val_ds = PairedQuestionContextSidecarDataset(val_q_meta, val_c_meta, **ds_kwargs)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(f"No paired records loaded: train={len(train_ds)} val={len(val_ds)}")
    log(f"PairedQuestionContext records: train={len(train_ds)} val={len(val_ds)}")

    # __getitem__ does 3 torch.load() disk reads per sample, so the loader is I/O-bound;
    # workers + prefetch overlap those reads with GPU compute (num_workers=0 starves it).
    _nw = int(probe_cfg.get("num_workers", 0))
    _loader_extra = {}
    if _nw > 0:
        _loader_extra = {
            "persistent_workers": True,
            "prefetch_factor": int(probe_cfg.get("prefetch_factor", 4)),
        }
    train_loader = DataLoader(
        train_ds,
        batch_size=probe_cfg["batch_size"],
        shuffle=True,
        num_workers=_nw,
        pin_memory=bool(probe_cfg.get("pin_memory", torch.cuda.is_available())),
        **_loader_extra,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=probe_cfg["batch_size"],
        shuffle=False,
        num_workers=_nw,
        pin_memory=bool(probe_cfg.get("pin_memory", torch.cuda.is_available())),
        **_loader_extra,
    )

    q_ckpt_path = probe_cfg["question_checkpoint_path"]
    c_ckpt_path = probe_cfg["context_checkpoint_path"]
    q_encoder, q_ckpt, q_dim = _build_question_answer_encoder(q_ckpt_path, embed_weights, device, noise_std=0.0)
    c_encoder, c_ckpt, c_dim = _build_context_question_answer_encoder(c_ckpt_path, embed_weights, device, noise_std=0.0)
    model = PairedQuestionContextResNetProbe(
        question_encoder=q_encoder,
        context_encoder=c_encoder,
        question_encoder_dim=q_dim,
        context_encoder_dim=c_dim,
        hidden_dim=int(probe_cfg.get("hidden_dim", 128)),
        dropout=float(probe_cfg.get("dropout", 0.1)),
        freeze_encoders=bool(probe_cfg.get("freeze_encoders", True)),
    ).to(device)

    train_q_items = [p[0] for p in train_ds.pairs]
    train_c_items = [p[1] for p in train_ds.pairs]
    no_pos, no_neg, no_pos_weight = _binary_pos_weight_from_labels(train_q_items, get_without_context_correct)
    ctx_pos, ctx_neg, ctx_pos_weight = _binary_pos_weight_from_labels(train_c_items, get_with_context_correct)
    log(
        "Paired labels: "
        f"without_context positives={no_pos} negatives={no_neg} pos_weight={no_pos_weight:.4f}; "
        f"with_context positives={ctx_pos} negatives={ctx_neg} pos_weight={ctx_pos_weight:.4f}"
    )
    no_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([no_pos_weight], device=device))
    ctx_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([ctx_pos_weight], device=device))
    ctx_loss_weight = float(probe_cfg.get("with_context_loss_weight", 1.0))
    no_loss_weight = float(probe_cfg.get("without_context_loss_weight", 1.0))
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(probe_cfg["learning_rate"]),
        weight_decay=float(probe_cfg.get("weight_decay", 1e-2)),
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(probe_cfg.get("lr_scheduler_factor", 0.5)),
        patience=int(probe_cfg.get("lr_scheduler_patience", 4)),
        min_lr=float(probe_cfg.get("min_learning_rate", 1e-7)),
    )
    checkpoint_path = Path(output_cfg["checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    val_prediction_path = output_cfg.get("val_prediction_path")
    show_progress = bool(probe_cfg.get("show_progress", sys.stderr.isatty()))
    patience = int(probe_cfg.get("early_stopping_patience", 10))
    min_delta = float(probe_cfg.get("early_stopping_min_delta", 1e-4))
    best_score = -1.0
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, int(probe_cfg["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        for qa_batch, cqa_batch, no_targets, ctx_targets, _ in tqdm(train_loader, desc=f"Epoch {epoch:03d} train", leave=False, disable=not show_progress):
            qa_batch = _move_nested(qa_batch, device)
            cqa_batch = _move_nested(cqa_batch, device)
            no_targets = no_targets.to(device)
            ctx_targets = ctx_targets.to(device)
            optimizer.zero_grad()
            outputs = model(qa_batch, cqa_batch)
            no_loss = no_criterion(outputs["without_context_logits"], no_targets)
            ctx_loss = ctx_criterion(outputs["with_context_logits"], ctx_targets)
            loss = no_loss_weight * no_loss + ctx_loss_weight * ctx_loss
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        model.eval()
        no_true, no_score = [], []
        ctx_true, ctx_score = [], []
        val_records = []
        with torch.no_grad():
            for qa_batch, cqa_batch, no_targets, ctx_targets, indices in tqdm(val_loader, desc=f"Epoch {epoch:03d} val", leave=False, disable=not show_progress):
                qa_batch = _move_nested(qa_batch, device)
                cqa_batch = _move_nested(cqa_batch, device)
                outputs = model(qa_batch, cqa_batch)
                no_probs = torch.sigmoid(outputs["without_context_logits"]).squeeze(-1).cpu().numpy().tolist()
                ctx_probs = torch.sigmoid(outputs["with_context_logits"]).squeeze(-1).cpu().numpy().tolist()
                no_labels = no_targets.squeeze(-1).numpy().tolist()
                ctx_labels = ctx_targets.squeeze(-1).numpy().tolist()
                if not isinstance(no_probs, list): no_probs = [no_probs]
                if not isinstance(ctx_probs, list): ctx_probs = [ctx_probs]
                if not isinstance(no_labels, list): no_labels = [no_labels]
                if not isinstance(ctx_labels, list): ctx_labels = [ctx_labels]
                no_score.extend(float(x) for x in no_probs)
                ctx_score.extend(float(x) for x in ctx_probs)
                no_true.extend(float(x) for x in no_labels)
                ctx_true.extend(float(x) for x in ctx_labels)
                for no_p, ctx_p, no_y, ctx_y, idx in zip(no_probs, ctx_probs, no_labels, ctx_labels, indices.numpy().tolist()):
                    q_item, c_item = val_ds.pairs[int(idx)]
                    ex = c_item.get("example") or q_item.get("example") or {}
                    val_records.append({
                        "idx": int(idx),
                        "example_id": _get_example_id(c_item),
                        "dataset": ex.get("dataset"),
                        "split": ex.get("split"),
                        "question": ex.get("question"),
                        "transition": _get_transition(c_item),
                        "without_context_correct_label": int(no_y),
                        "with_context_correct_label": int(ctx_y),
                        "without_context_correct_probability": float(no_p),
                        "with_context_correct_probability": float(ctx_p),
                        "without_context_prediction": int(float(no_p) >= 0.5),
                        "with_context_prediction": int(float(ctx_p) >= 0.5),
                    })
        no_metrics = compute_metrics(no_true, no_score)
        ctx_metrics = compute_metrics(ctx_true, ctx_score)
        mean_auc = (float(no_metrics["auc"]) + float(ctx_metrics["auc"])) / 2.0
        avg_loss = total_loss / max(len(train_loader), 1)
        log(
            f"Epoch {epoch:03d} | loss={avg_loss:.4f} | mean_auc={mean_auc:.4f} "
            f"| noctx_auc={no_metrics['auc']:.4f} noctx_f1={no_metrics['f1']:.4f} noctx_acc={no_metrics['accuracy']:.4f} "
            f"| ctx_auc={ctx_metrics['auc']:.4f} ctx_f1={ctx_metrics['f1']:.4f} ctx_acc={ctx_metrics['accuracy']:.4f} "
            f"| lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        scheduler.step(mean_auc)
        if mean_auc > best_score + min_delta:
            best_score = mean_auc
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "state_dict": model.state_dict(),
                "model_name": model_name,
                "architecture": "paired_question_context_resnet",
                "question_checkpoint_path": q_ckpt_path,
                "context_checkpoint_path": c_ckpt_path,
                "question_encoder_dim": q_dim,
                "context_encoder_dim": c_dim,
                "hidden_dim": int(probe_cfg.get("hidden_dim", 128)),
                "dropout": float(probe_cfg.get("dropout", 0.1)),
                "freeze_encoders": bool(probe_cfg.get("freeze_encoders", True)),
                "question_encoder_checkpoint_meta": {k: q_ckpt.get(k) for k in ["architecture", "label_mode", "best_auc", "best_epoch"]},
                "context_encoder_checkpoint_meta": {k: c_ckpt.get(k) for k in ["architecture", "label_mode", "best_auc", "best_epoch"]},
                "best_mean_auc": best_score,
                "best_epoch": best_epoch,
            }, checkpoint_path)
            log(f"Saved new best paired checkpoint: {checkpoint_path}")
            if val_prediction_path:
                write_jsonl(val_records, val_prediction_path)
        else:
            no_improve += 1
            log(f"No mean AUC improvement for {no_improve}/{patience}; best_epoch={best_epoch}, best_mean_auc={best_score:.4f}")
        if patience > 0 and no_improve >= patience:
            log(f"Early stopping at epoch={epoch}; best_epoch={best_epoch}; best_mean_auc={best_score:.4f}")
            break
    log(f"Done. Best epoch={best_epoch}; best mean AUC={best_score:.4f}")

def _train_context_question_answer_resnet(config: dict, train_meta, val_meta, tokenizer, embed_weights):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    probe_cfg = config["probe"]
    data_cfg = config["data"]
    model_name = config["model_name"]
    positive_transition = probe_cfg.get("positive_transition", "NY")
    label_mode = probe_cfg.get("label_mode", "transition_positive")
    binary_target_name = "with_context_correct" if label_mode == "with_context_correct" else "helpful"
    regression_transition = probe_cfg.get("rag_class_regression_transition", "YN")

    _require_context_sidecars(train_meta, "train")
    _require_context_sidecars(val_meta, "validation")

    rag_class_labels = load_rag_signal_label_map(
        data_cfg.get("rag_class_meta_paths", []),
        regression_transition=regression_transition,
    )
    rag_class_loss_weight = float(probe_cfg.get("rag_class_loss_weight", 0.0 if not rag_class_labels else 0.2))
    aux_binary_loss_weight = float(probe_cfg.get("aux_binary_loss_weight", 0.5))
    train_rag_class_counts = _rag_class_label_distribution(train_meta, rag_class_labels, regression_transition=regression_transition)
    val_rag_class_counts = _rag_class_label_distribution(val_meta, rag_class_labels, regression_transition=regression_transition)
    log(f"Train RAG signal distribution: {_named_label_distribution(train_rag_class_counts)}")
    log(f"Validation RAG signal distribution: {_named_label_distribution(val_rag_class_counts)}")
    rag_class_weights = _make_rag_class_weights(
        train_rag_class_counts,
        mode=probe_cfg.get("rag_class_weighting", "sqrt_inverse"),
        max_weight=float(probe_cfg.get("rag_class_max_weight", 10.0)),
    )

    ds_kwargs = dict(
        tokenizer=tokenizer,
        context_max_len=probe_cfg.get("context_max_len", probe_cfg.get("max_context_len", 2048)),
        question_max_len=probe_cfg.get("question_max_len", 128),
        answer_max_len=probe_cfg.get("answer_max_len", 32),
        positive_transition=positive_transition,
        label_mode=label_mode,
        rag_class_labels_by_id=rag_class_labels,
        regression_transition=regression_transition,
        use_numeric_features=probe_cfg.get("use_numeric_features", True),
        use_probability_features=probe_cfg.get("use_probability_features", True),
        use_residual_features=probe_cfg.get("use_residual_features", True),
        use_attention_features=probe_cfg.get("use_attention_features", True),
        use_last_hidden=probe_cfg.get("use_last_hidden", True),
        use_attention_mass=probe_cfg.get("use_attention_mass", False),
        ablate_segments=tuple(probe_cfg.get("ablate_segments") or ()),
        label_override=_load_label_override(probe_cfg.get("label_override_path")),
        rewritten_label_override=_load_label_override(probe_cfg.get("rewritten_label_override_path")),
        label_override_required=bool(probe_cfg.get("label_override_required", False)),
        ablate_channels=tuple(probe_cfg.get("ablate_channels") or ()),
    )
    train_ds = ContextQuestionAnswerSidecarDataset(train_meta, **ds_kwargs)
    val_ds = ContextQuestionAnswerSidecarDataset(val_meta, **ds_kwargs)

    # Residual-boosting hook: if a per-example hidden logit offset is supplied, the
    # encoder is trained to CORRECT that offset (loss & val selection use
    # encoder_logit + offset), i.e. gradient boosting on top of the hidden probe.
    _resid_off = None
    _resid_off_path = probe_cfg.get("residual_offset_path")
    if _resid_off_path:
        import json as _json
        _lut = _json.load(open(_resid_off_path))
        def _off_vec(meta):
            miss = 0; vals = []
            for it in meta:
                eid = _example_id(it)
                if eid in _lut:
                    vals.append(float(_lut[eid]))
                else:
                    miss += 1; vals.append(0.0)
            return torch.tensor(vals, device=device, dtype=torch.float32), miss
        _off_train, _mt = _off_vec(train_meta)
        _off_val, _mv = _off_vec(val_meta)
        _resid_off = (_off_train, _off_val)
        log(f"[residual-boost] offset from {_resid_off_path}: train_miss={_mt}/{len(train_meta)} val_miss={_mv}/{len(val_meta)}")

    sample = train_ds[0]
    numeric_channels = int(sample[0].shape[0])
    attention_channels = int(sample[1].shape[0])
    llm_hidden_dim = int(sample[3].shape[-1])
    log(
        "ContextQuestionAnswerResNet feature dims: "
        f"numeric_channels={numeric_channels}; attention_channels={attention_channels}; llm_hidden_dim={llm_hidden_dim}; "
        f"context_max_len={train_ds.context_max_len}; question_max_len={train_ds.question_max_len}; answer_max_len={train_ds.answer_max_len}"
    )

    _collate = fast_collate if os.environ.get("PROBE_FAST_COLLATE") == "1" else None
    if _collate is not None:
        # batch-direct packed fetch (no-op when PROBE_PACKED_ROOT is off/unresolved)
        train_ds._packed_batch = val_ds._packed_batch = True
    train_loader = DataLoader(train_ds, batch_size=probe_cfg["batch_size"], shuffle=True, num_workers=probe_cfg.get("num_workers", 4), pin_memory=bool(probe_cfg.get("pin_memory", torch.cuda.is_available())), collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=probe_cfg["batch_size"], shuffle=False, num_workers=probe_cfg.get("num_workers", 4), pin_memory=bool(probe_cfg.get("pin_memory", torch.cuda.is_available())), collate_fn=_collate)

    num_pos = _count_positive(train_meta, positive_transition=positive_transition, label_mode=label_mode)
    num_neg = len(train_meta) - num_pos
    pos_weight = num_neg / max(num_pos, 1)
    log(f"ContextQuestionAnswerResNet label_mode={label_mode}; positives={num_pos}; negatives={num_neg}; pos_weight={pos_weight:.4f}")

    cqa_architecture = probe_cfg.get("architecture", "context_question_answer_resnet")
    if cqa_architecture == "context_question_answer_modality_gated":
        cqa_model_cls = ContextQuestionAnswerModalityGatedProbe
    elif cqa_architecture == "context_question_answer_modality_ensemble":
        cqa_model_cls = ContextQuestionAnswerModalityEnsembleProbe
    else:
        cqa_model_cls = ContextQuestionAnswerResNetProbe
    log(f"ContextQuestionAnswer model class: {cqa_model_cls.__name__}")
    model = cqa_model_cls(
        embed_weights=embed_weights,
        numeric_channels=numeric_channels,
        attention_channels=attention_channels,
        hidden_dim=probe_cfg.get("hidden_dim", 128),
        embed_proj_dim=probe_cfg.get("embed_proj_dim", 32),
        llm_hidden_dim=llm_hidden_dim,
        hidden_proj_dim=probe_cfg.get("hidden_proj_dim", probe_cfg.get("embed_proj_dim", 32)),
        num_res_blocks=probe_cfg.get("num_res_blocks", 3),
        dropout=probe_cfg.get("dropout", 0.1),
        noise_std=probe_cfg.get("noise_std", 0.02),
        num_rag_classes=len(RAG_SIGNAL_CLASSES),
        use_token_embedding=probe_cfg.get("use_token_embedding", True),
        use_last_hidden=probe_cfg.get("use_last_hidden", True),
        **({"conv_kernel_size": int(probe_cfg.get("conv_kernel_size", 3))}
           if cqa_model_cls is ContextQuestionAnswerResNetProbe else {}),
        **({
            "aux_hidden_dim": probe_cfg.get("aux_hidden_dim"),
            "aux_num_res_blocks": probe_cfg.get("aux_num_res_blocks", 2),
            "prob_feature_channels": probe_cfg.get("prob_feature_channels", 5),
            "use_probability_features": probe_cfg.get("use_probability_features", True),
            "use_residual_features": probe_cfg.get("use_residual_features", True),
            "use_attention_features": probe_cfg.get("use_attention_features", True),
            "modality_dropout": probe_cfg.get("modality_dropout", 0.1),
        } if cqa_model_cls in {ContextQuestionAnswerModalityGatedProbe, ContextQuestionAnswerModalityEnsembleProbe} else {}),
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=float(probe_cfg["learning_rate"]), weight_decay=float(probe_cfg.get("weight_decay", 1e-2)))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    modality_aux_loss_weight = float(probe_cfg.get("modality_aux_loss_weight", 0.3))
    aux_binary_criterion = nn.BCEWithLogitsLoss(pos_weight=_aux_binary_pos_weight(train_meta).to(device))
    rag_class_criterion = nn.CrossEntropyLoss(weight=rag_class_weights.to(device) if rag_class_weights is not None else None, ignore_index=RAG_CLASS_IGNORE_INDEX)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(probe_cfg.get("lr_scheduler_factor", 0.5)),
        patience=int(probe_cfg.get("lr_scheduler_patience", 4)),
        min_lr=float(probe_cfg.get("min_learning_rate", 1e-7)),
    )

    checkpoint_path = Path(config["output"]["checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    val_prediction_path = config.get("output", {}).get("val_prediction_path")
    best_auc = -1.0
    best_epoch = 0
    no_improve = 0
    patience = int(probe_cfg.get("early_stopping_patience", 12))
    min_delta = float(probe_cfg.get("early_stopping_min_delta", 1e-4))
    show_progress = bool(probe_cfg.get("show_progress", sys.stderr.isatty()))

    _train_iter = packed_store.maybe_cuda_prefetch(train_loader, train_ds, device)
    for epoch in range(1, int(probe_cfg["epochs"]) + 1):
        model.train()
        total_loss = total_binary_loss = total_aux_binary_loss = total_rag_class_loss = 0.0
        rag_batches = 0
        for batch in tqdm(_train_iter, desc=f"Epoch {epoch:03d} train", leave=False, disable=not show_progress):
            (c_num, c_attn, c_tok, c_hidden, c_mask, q_num, q_attn, q_tok, q_hidden, q_mask, a_num, a_attn, a_tok, a_hidden, a_mask, targets, rag_targets, aux_targets, _bidx) = batch
            c_num, c_attn, c_tok, c_hidden, c_mask = c_num.to(device), c_attn.to(device), c_tok.to(device), c_hidden.to(device=device, dtype=torch.float32), c_mask.to(device)
            q_num, q_attn, q_tok, q_hidden, q_mask = q_num.to(device), q_attn.to(device), q_tok.to(device), q_hidden.to(device=device, dtype=torch.float32), q_mask.to(device)
            a_num, a_attn, a_tok, a_hidden, a_mask = a_num.to(device), a_attn.to(device), a_tok.to(device), a_hidden.to(device=device, dtype=torch.float32), a_mask.to(device)
            targets, rag_targets, aux_targets = targets.to(device), rag_targets.to(device), aux_targets.to(device)
            optimizer.zero_grad()
            outputs = model(c_num, c_attn, c_tok, c_hidden, c_mask, q_num, q_attn, q_tok, q_hidden, q_mask, a_num, a_attn, a_tok, a_hidden, a_mask, return_multitask=True)
            _binlogit = outputs["binary_logits"]
            if _resid_off is not None:
                _binlogit = _binlogit + _resid_off[0][_bidx.to(device)].unsqueeze(-1)
            binary_loss = criterion(_binlogit, targets)
            if "modality_logits" in outputs:
                modality_targets = targets.expand_as(outputs["modality_logits"])
                modality_loss = criterion(outputs["modality_logits"], modality_targets)
            else:
                modality_loss = outputs["binary_logits"].sum() * 0.0
            aux_binary_loss = aux_binary_criterion(outputs["aux_binary_logits"], aux_targets)
            valid_rag = (rag_targets != RAG_CLASS_IGNORE_INDEX).sum().item()
            if valid_rag > 0 and rag_class_loss_weight > 0:
                rag_class_loss = rag_class_criterion(outputs["rag_class_logits"], rag_targets)
                total_rag_class_loss += float(rag_class_loss.item())
                rag_batches += 1
            else:
                rag_class_loss = outputs["rag_class_logits"].sum() * 0.0
            loss = binary_loss + modality_aux_loss_weight * modality_loss + aux_binary_loss_weight * aux_binary_loss + rag_class_loss_weight * rag_class_loss
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            total_binary_loss += float(binary_loss.item())
            total_aux_binary_loss += float(aux_binary_loss.item())

        model.eval()
        y_true, y_score = [], []
        rag_class_true, rag_class_pred = [], []
        val_pred_records = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch:03d} val", leave=False, disable=not show_progress):
                (c_num, c_attn, c_tok, c_hidden, c_mask, q_num, q_attn, q_tok, q_hidden, q_mask, a_num, a_attn, a_tok, a_hidden, a_mask, targets, rag_targets, aux_targets, indices) = batch
                c_num, c_attn, c_tok, c_hidden, c_mask = c_num.to(device), c_attn.to(device), c_tok.to(device), c_hidden.to(device=device, dtype=torch.float32), c_mask.to(device)
                q_num, q_attn, q_tok, q_hidden, q_mask = q_num.to(device), q_attn.to(device), q_tok.to(device), q_hidden.to(device=device, dtype=torch.float32), q_mask.to(device)
                a_num, a_attn, a_tok, a_hidden, a_mask = a_num.to(device), a_attn.to(device), a_tok.to(device), a_hidden.to(device=device, dtype=torch.float32), a_mask.to(device)
                targets, rag_targets, aux_targets = targets.to(device), rag_targets.to(device), aux_targets.to(device)
                outputs = model(c_num, c_attn, c_tok, c_hidden, c_mask, q_num, q_attn, q_tok, q_hidden, q_mask, a_num, a_attn, a_tok, a_hidden, a_mask, return_multitask=True)
                _vbinlogit = outputs["binary_logits"]
                if _resid_off is not None:
                    _vbinlogit = _vbinlogit + _resid_off[1][indices.to(device)].unsqueeze(-1)
                probs = torch.sigmoid(_vbinlogit).squeeze(-1).cpu().numpy().tolist()
                labels = targets.squeeze(-1).cpu().numpy().tolist()
                if not isinstance(probs, list): probs = [probs]
                if not isinstance(labels, list): labels = [labels]
                y_score.extend(float(x) for x in probs)
                y_true.extend(float(x) for x in labels)
                class_preds = torch.argmax(outputs["rag_class_logits"], dim=-1).cpu().numpy().tolist()
                class_labels = rag_targets.cpu().numpy().tolist()
                class_probs = torch.softmax(outputs["rag_class_logits"], dim=-1).cpu().numpy().tolist()
                aux_probs = torch.sigmoid(outputs["aux_binary_logits"]).cpu().numpy().tolist()
                aux_labels = aux_targets.cpu().numpy().tolist()
                for true_class, pred_class in zip(class_labels, class_preds):
                    if true_class != RAG_CLASS_IGNORE_INDEX:
                        rag_class_true.append(int(true_class)); rag_class_pred.append(int(pred_class))
                for prob, label, true_class, pred_class, cls_prob, ap, al, idx in zip(probs, labels, class_labels, class_preds, class_probs, aux_probs, aux_labels, indices.numpy().tolist()):
                    item = val_meta[int(idx)]
                    val_pred_records.append({
                        "idx": int(idx),
                        "example_id": _example_id(item),
                        "transition": _get_transition(item),
                        "label": int(label),
                        "binary_target": binary_target_name,
                        "binary_probability": float(prob),
                        "helpful_probability": float(prob),
                        "prediction": int(float(prob) >= 0.5),
                        "with_context_correct_probability": float(prob) if label_mode == "with_context_correct" else float(ap[0]),
                        "aux_with_context_correct_probability": float(ap[0]),
                        "rag_harmful_probability": float(ap[1]),
                        "with_context_correct_label": int(al[0]),
                        "rag_harmful_label": int(al[1]),
                        "rag_signal_label": int(true_class),
                        "rag_signal_label_name": RAG_SIGNAL_CLASSES[int(true_class)] if true_class != RAG_CLASS_IGNORE_INDEX else None,
                        "rag_signal_prediction": int(pred_class),
                        "rag_signal_prediction_name": RAG_SIGNAL_CLASSES[int(pred_class)],
                        "rag_signal_probabilities": {name: float(value) for name, value in zip(RAG_SIGNAL_CLASSES, cls_prob)},
                    })
        metrics = compute_metrics(y_true, y_score)
        rag_metrics = compute_classification_metrics(rag_class_true, rag_class_pred)
        avg_loss = total_loss / max(len(train_loader), 1)
        log(
            f"Epoch {epoch:03d} | loss={avg_loss:.4f} | binary_loss={total_binary_loss/max(len(train_loader),1):.4f} "
            f"| aux_binary_loss={total_aux_binary_loss/max(len(train_loader),1):.4f} | rag_class_loss={total_rag_class_loss/max(rag_batches,1):.4f} "
            f"| auc={metrics['auc']:.4f} | f1={metrics['f1']:.4f} | acc={metrics['accuracy']:.4f} "
            f"| rag_acc={rag_metrics['accuracy']:.4f} | rag_macro_f1={rag_metrics['macro_f1']:.4f} | lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        scheduler.step(metrics["auc"])
        if metrics["auc"] > best_auc + min_delta:
            best_auc = metrics["auc"]
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "state_dict": model.state_dict(),
                "model_name": model_name,
                "architecture": cqa_architecture,
                "numeric_channels": numeric_channels,
                "attention_channels": attention_channels,
                "hidden_dim": probe_cfg.get("hidden_dim", 128),
                "embed_proj_dim": probe_cfg.get("embed_proj_dim", 32),
                "llm_hidden_dim": llm_hidden_dim,
                "hidden_proj_dim": probe_cfg.get("hidden_proj_dim", probe_cfg.get("embed_proj_dim", 32)),
                "num_res_blocks": probe_cfg.get("num_res_blocks", 3),
                "dropout": probe_cfg.get("dropout", 0.1),
                "context_max_len": train_ds.context_max_len,
                "question_max_len": train_ds.question_max_len,
                "answer_max_len": train_ds.answer_max_len,
                "max_context_len": train_ds.context_max_len,
                "positive_transition": positive_transition,
                "label_mode": label_mode,
                "binary_target_name": binary_target_name,
                "rag_signal_classes": RAG_SIGNAL_CLASSES,
                "use_numeric_features": probe_cfg.get("use_numeric_features", True),
                "use_probability_features": probe_cfg.get("use_probability_features", True),
                "use_residual_features": probe_cfg.get("use_residual_features", True),
                "use_attention_features": probe_cfg.get("use_attention_features", True),
                "use_token_embedding": probe_cfg.get("use_token_embedding", True),
                "use_last_hidden": probe_cfg.get("use_last_hidden", True),
                "use_attention_mass": probe_cfg.get("use_attention_mass", False),
                "ablate_segments": list(probe_cfg.get("ablate_segments") or []),
                "ablate_channels": list(probe_cfg.get("ablate_channels") or []),
                "conv_kernel_size": int(probe_cfg.get("conv_kernel_size", 3)),
                "aux_hidden_dim": probe_cfg.get("aux_hidden_dim"),
                "aux_num_res_blocks": probe_cfg.get("aux_num_res_blocks", 2),
                "prob_feature_channels": probe_cfg.get("prob_feature_channels", 5),
                "modality_dropout": probe_cfg.get("modality_dropout", 0.1),
                "modality_aux_loss_weight": modality_aux_loss_weight,
                "aux_binary_loss_weight": aux_binary_loss_weight,
                "rag_class_loss_weight": rag_class_loss_weight,
                "rag_class_regression_transition": regression_transition,
                "best_auc": best_auc,
                "best_epoch": best_epoch,
            }, checkpoint_path)
            log(f"Saved new best checkpoint: {checkpoint_path}")
            if val_prediction_path:
                write_jsonl(val_pred_records, val_prediction_path)
        else:
            no_improve += 1
            log(f"No AUC improvement for {no_improve}/{patience}; best_epoch={best_epoch}, best_auc={best_auc:.4f}")
        if patience > 0 and no_improve >= patience:
            log(f"Early stopping at epoch={epoch}; best_epoch={best_epoch}; best_auc={best_auc:.4f}")
            break
    log(f"Done. Best epoch={best_epoch}; best AUC={best_auc:.4f}")


def _configure_torch_threads(probe_cfg: dict):
    num_threads = int(probe_cfg.get("torch_num_threads", 4))
    num_interop_threads = int(probe_cfg.get("torch_num_interop_threads", 1))
    num_threads = max(1, num_threads)
    num_interop_threads = max(1, num_interop_threads)
    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(num_interop_threads)
    except RuntimeError:
        pass
    log(f"Torch CPU threads: intraop={torch.get_num_threads()}; interop={num_interop_threads}")


def _train_question_answer_frozen_encoder_fusion(config: dict, train_meta, val_meta, tokenizer, embed_weights):
    probe_cfg = config["probe"]
    output_cfg = config["output"]
    device = config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    positive_transition = probe_cfg.get("positive_transition", "NY")
    label_mode = probe_cfg.get("label_mode", "without_context_correct")
    checkpoint_path = output_cfg["checkpoint_path"]
    val_prediction_path = output_cfg.get("val_prediction_path")
    architecture = probe_cfg.get("architecture", "question_answer_frozen_encoder_fusion")
    is_context_fusion = architecture == "context_question_answer_frozen_encoder_fusion"
    binary_target_name = "with_context_correct" if is_context_fusion else ("without_context_correct" if label_mode == "without_context_correct" else "helpful")

    if is_context_fusion:
        model, branch_specs, branch_ckpts = _build_context_question_answer_frozen_fusion(config, embed_weights, device)
        dataset_factory = _context_dataset_from_encoder_ckpt
        wrapper_cls = ContextQuestionAnswerFrozenFusionDataset
    elif architecture == "question_answer_frozen_token_fusion":
        model, branch_specs, branch_ckpts = _build_question_answer_frozen_token_fusion(config, embed_weights, device)
        dataset_factory = _question_dataset_from_encoder_ckpt
        wrapper_cls = QuestionAnswerFrozenFusionDataset
    else:
        model, branch_specs, branch_ckpts = _build_question_answer_frozen_fusion(config, embed_weights, device)
        dataset_factory = _question_dataset_from_encoder_ckpt
        wrapper_cls = QuestionAnswerFrozenFusionDataset
    branch_names = [spec["name"] for spec in branch_specs]
    log(f"Frozen {'context' if is_context_fusion else 'question'} fusion branches: {branch_names}")

    train_branch_datasets = {name: dataset_factory(train_meta, tokenizer, branch_ckpts[name], positive_transition, label_mode) for name in branch_names}
    val_branch_datasets = {name: dataset_factory(val_meta, tokenizer, branch_ckpts[name], positive_transition, label_mode) for name in branch_names}
    train_ds = wrapper_cls(train_branch_datasets)
    val_ds = wrapper_cls(val_branch_datasets)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(probe_cfg.get("batch_size", 512)),
        shuffle=True,
        num_workers=int(probe_cfg.get("num_workers", 0)),
        pin_memory=bool(probe_cfg.get("pin_memory", torch.cuda.is_available())),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(probe_cfg.get("batch_size", 512)),
        shuffle=False,
        num_workers=int(probe_cfg.get("num_workers", 0)),
        pin_memory=bool(probe_cfg.get("pin_memory", torch.cuda.is_available())),
    )

    num_pos = _count_positive(train_meta, positive_transition=positive_transition, label_mode=label_mode)
    num_neg = len(train_meta) - num_pos
    pos_weight = num_neg / max(num_pos, 1)
    log(f"Frozen fusion label_mode={label_mode}; positives={num_pos}; negatives={num_neg}; pos_weight={pos_weight:.4f}")

    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    optimizer = optim.AdamW(trainable_parameters, lr=float(probe_cfg.get("learning_rate", 1e-4)), weight_decay=float(probe_cfg.get("weight_decay", 1e-2)))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(probe_cfg.get("lr_scheduler_factor", 0.5)),
        patience=int(probe_cfg.get("lr_scheduler_patience", 4)),
        min_lr=float(probe_cfg.get("min_learning_rate", 1e-7)),
    )
    epochs = int(probe_cfg.get("epochs", 40))
    patience = int(probe_cfg.get("early_stopping_patience", 10))
    min_delta = float(probe_cfg.get("early_stopping_min_delta", 0.0001))
    show_progress = bool(probe_cfg.get("show_progress", True))
    best_auc = -1.0
    best_epoch = 0
    no_improve = 0
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    if val_prediction_path:
        Path(val_prediction_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0
        for branch_batches, targets, _idx in tqdm(train_loader, desc=f"Epoch {epoch:03d} train", leave=False, disable=not show_progress):
            branch_batches = _move_branch_batches(branch_batches, device)
            targets = targets.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(branch_batches, return_multitask=True)
            loss = criterion(outputs["binary_logits"], targets)
            loss.backward()
            optimizer.step()
            bs = int(targets.size(0))
            total_loss += float(loss.item()) * bs
            total_seen += bs

        model.eval()
        y_true, y_score = [], []
        val_pred_records = []
        with torch.no_grad():
            for branch_batches, targets, indices in tqdm(val_loader, desc=f"Epoch {epoch:03d} val", leave=False, disable=not show_progress):
                branch_batches = _move_branch_batches(branch_batches, device)
                targets = targets.to(device=device, dtype=torch.float32)
                outputs = model(branch_batches, return_multitask=True)
                probs = torch.sigmoid(outputs["binary_logits"]).cpu().numpy().reshape(-1)
                modality_probs = torch.sigmoid(outputs["modality_logits"]).cpu().numpy().tolist()
                labels = targets.cpu().numpy().reshape(-1).astype(int).tolist()
                y_true.extend(labels)
                y_score.extend([float(p) for p in probs])
                for local_i, meta_idx in enumerate(indices.tolist() if torch.is_tensor(indices) else indices):
                    item = val_meta[int(meta_idx)]
                    rec = {
                        "example_id": _get_example_id(item),
                        "dataset": item.get("example", {}).get("dataset", item.get("dataset")),
                        "split": item.get("example", {}).get("split", item.get("split")),
                        "question": item.get("example", {}).get("question", item.get("question")),
                        "transition": _get_transition(item),
                        "label": int(labels[local_i]),
                        "binary_target": binary_target_name,
                        "binary_probability": float(probs[local_i]),
                        "helpful_probability": float(probs[local_i]),
                        "prediction": int(probs[local_i] >= 0.5),
                        "modality_probabilities": {name: float(value) for name, value in zip(branch_names, modality_probs[local_i])},
                        "source_meta": item,
                    }
                    if is_context_fusion:
                        rec["with_context_correct_probability"] = float(probs[local_i])
                        rec["with_context_correct_label"] = int(labels[local_i])
                    else:
                        rec["without_context_correct_probability"] = float(probs[local_i])
                    val_pred_records.append(rec)
        metrics = compute_metrics(y_true, y_score)
        scheduler.step(metrics["auc"])
        avg_loss = total_loss / max(total_seen, 1)
        lr = optimizer.param_groups[0]["lr"]
        log(f"Epoch {epoch:03d} | loss={avg_loss:.4f} | auc={metrics['auc']:.4f} | f1={metrics['f1']:.4f} | acc={metrics['accuracy']:.4f} | lr={lr:.2e}")
        if metrics["auc"] > best_auc + min_delta:
            best_auc = metrics["auc"]
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items() if not k.startswith("encoders.")},
                "model_name": config["model_name"],
                "architecture": architecture,
                "branch_checkpoints": branch_specs,
                "modality_names": branch_names,
                "encoder_dim": getattr(model, "encoder_dim", None),
                "hidden_dim": model.hidden_dim,
                "fusion_num_res_blocks": int(probe_cfg.get("fusion_num_res_blocks", 3)),
                "dropout": float(probe_cfg.get("dropout", 0.1)),
                "modality_dropout": float(probe_cfg.get("modality_dropout", 0.25)),
                "positive_transition": positive_transition,
                "label_mode": label_mode,
                "binary_target_name": binary_target_name,
                "best_auc": best_auc,
                "best_epoch": best_epoch,
            }, checkpoint_path)
            log(f"Saved new best checkpoint: {checkpoint_path}")
            if val_prediction_path:
                write_jsonl(val_pred_records, val_prediction_path)
        else:
            no_improve += 1
            log(f"No AUC improvement for {no_improve}/{patience}; best_epoch={best_epoch}, best_auc={best_auc:.4f}")
        if patience > 0 and no_improve >= patience:
            log(f"Early stopping at epoch={epoch}; best_epoch={best_epoch}; best_auc={best_auc:.4f}")
            break
    log(f"Done. Best epoch={best_epoch}; best AUC={best_auc:.4f}")
    return {"best_auc": best_auc, "best_epoch": best_epoch, "checkpoint_path": checkpoint_path}

def train_probe(config: dict):
    gpu_id = config.get("gpu_id")
    if gpu_id is not None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = config["model_name"]
    probe_cfg = config["probe"]
    data_cfg = config["data"]
    _configure_torch_threads(probe_cfg)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    architecture = probe_cfg.get("architecture", "gated_cnn")
    if architecture == "paired_question_context_resnet":
        train_q_meta = load_feature_meta(data_cfg["question_train_meta_paths"], splits=data_cfg.get("train_splits"), partition=data_cfg.get("train_partition"))
        val_q_meta = load_feature_meta(data_cfg["question_validation_meta_paths"], splits=data_cfg.get("validation_splits"), partition=data_cfg.get("validation_partition"))
        train_c_meta = load_feature_meta(data_cfg["context_train_meta_paths"], splits=data_cfg.get("train_splits"), partition=data_cfg.get("train_partition"))
        val_c_meta = load_feature_meta(data_cfg["context_validation_meta_paths"], splits=data_cfg.get("validation_splits"), partition=data_cfg.get("validation_partition"))
        log(f"Question train records: {len(train_q_meta)}; context train records: {len(train_c_meta)}")
        log(f"Question validation records: {len(val_q_meta)}; context validation records: {len(val_c_meta)}")
        if not train_q_meta or not val_q_meta or not train_c_meta or not val_c_meta:
            raise RuntimeError("No paired training/validation records loaded before joining.")
        log(f"Question train transition distribution: {_transition_distribution(train_q_meta)}")
        log(f"Context train transition distribution: {_transition_distribution(train_c_meta)}")
        embed_weights = extract_qwen_embeddings(model_name)
        log("Probe architecture: paired_question_context_resnet; using pretrained question/context encoders")
        return _train_paired_question_context_resnet(config, train_q_meta, train_c_meta, val_q_meta, val_c_meta, tokenizer, embed_weights)

    train_meta = load_feature_meta(
        data_cfg["train_meta_paths"],
        splits=data_cfg.get("train_splits"),
        partition=data_cfg.get("train_partition"),
    )
    val_meta = load_feature_meta(
        data_cfg["validation_meta_paths"],
        splits=data_cfg.get("validation_splits"),
        partition=data_cfg.get("validation_partition"),
    )

    log(f"Train records: {len(train_meta)}")
    log(f"Validation records: {len(val_meta)}")

    if not train_meta:
        raise RuntimeError("No training records loaded.")
    if not val_meta:
        raise RuntimeError("No validation records loaded.")

    positive_transition = probe_cfg.get("positive_transition", "NY")
    label_mode = probe_cfg.get("label_mode", "transition_positive")
    regression_transition = probe_cfg.get("rag_class_regression_transition", "YN")

    train_dist = _transition_distribution(train_meta)
    val_dist = _transition_distribution(val_meta)
    log(f"Train transition distribution: {train_dist}")
    log(f"Validation transition distribution: {val_dist}")

    num_pos = _count_positive(train_meta, positive_transition=positive_transition, label_mode=label_mode)
    num_neg = len(train_meta) - num_pos

    if num_pos == 0:
        raise RuntimeError(
            "No positive training examples found. "
            f"positive_transition={positive_transition}, label_mode={label_mode}, "
            f"train_transition_distribution={train_dist}. "
            "This usually means the chosen split has no NY examples, or the metadata "
            "does not contain either derived.transition or legacy is_correct_* fields."
        )

    pos_weight = num_neg / max(num_pos, 1)

    log(
        f"Positive transition: {positive_transition}; label_mode={label_mode}; "
        f"positives={num_pos}; negatives={num_neg}; pos_weight={pos_weight:.4f}"
    )

    if architecture in {"question_answer_frozen_encoder_fusion", "question_answer_frozen_token_fusion", "context_question_answer_frozen_encoder_fusion"}:
        branch_specs = _question_fusion_branch_specs(probe_cfg)
        embed_weights = _embedding_weights_for_question_fusion(model_name, branch_specs)
        log(f"Probe architecture: {architecture}; using frozen question encoder fusion training path")
        return _train_question_answer_frozen_encoder_fusion(config, train_meta, val_meta, tokenizer, embed_weights)

    embed_weights = extract_qwen_embeddings(model_name)
    if architecture in {"question_answer_resnet", "question_answer_modality_gated", "question_answer_modality_ensemble"}:
        log(f"Probe architecture: {architecture}; using sidecar question-answer training path")
        return _train_question_answer_resnet(config, train_meta, val_meta, tokenizer, embed_weights)
    if architecture in {"context_question_answer_resnet", "context_question_answer_modality_gated", "context_question_answer_modality_ensemble"}:
        log(f"Probe architecture: {architecture}; using sidecar context-question-answer training path")
        return _train_context_question_answer_resnet(config, train_meta, val_meta, tokenizer, embed_weights)

    num_attn_channels = infer_num_attn_channels(train_meta)
    rag_class_labels = load_rag_signal_label_map(
        data_cfg.get("rag_class_meta_paths", []),
        regression_transition=regression_transition,
    )
    rag_class_loss_weight = float(
        probe_cfg.get(
            "rag_class_loss_weight",
            1.0 if rag_class_labels else 0.0,
        )
    )
    log(
        "RAG signal auxiliary task: "
        f"classes={list(RAG_SIGNAL_CLASSES)}, "
        f"labels_from_files={len(rag_class_labels)}, "
        f"loss_weight={rag_class_loss_weight:.4f}, "
        f"regression_transition={regression_transition}"
    )

    train_rag_class_counts = _rag_class_label_distribution(
        train_meta,
        rag_class_labels,
        regression_transition=regression_transition,
    )
    val_rag_class_counts = _rag_class_label_distribution(
        val_meta,
        rag_class_labels,
        regression_transition=regression_transition,
    )
    log(f"Train RAG signal distribution: {_named_label_distribution(train_rag_class_counts)}")
    log(f"Validation RAG signal distribution: {_named_label_distribution(val_rag_class_counts)}")

    rag_class_weighting = probe_cfg.get("rag_class_weighting", "sqrt_inverse")
    rag_class_max_weight = float(probe_cfg.get("rag_class_max_weight", 10.0))
    rag_class_weights = _make_rag_class_weights(
        train_rag_class_counts,
        mode=rag_class_weighting,
        max_weight=rag_class_max_weight,
    )
    if rag_class_weights is not None:
        log(
            "RAG signal class weights: "
            f"{dict(zip(RAG_SIGNAL_CLASSES, [round(float(x), 4) for x in rag_class_weights.tolist()]))}"
        )

    train_ds = RAGFeatureDataset(
        train_meta,
        tokenizer,
        max_len=probe_cfg["max_context_len"],
        positive_transition=positive_transition,
        label_mode=label_mode,
        rag_class_labels_by_id=rag_class_labels,
        regression_transition=regression_transition,
    )
    val_ds = RAGFeatureDataset(
        val_meta,
        tokenizer,
        max_len=probe_cfg["max_context_len"],
        positive_transition=positive_transition,
        label_mode=label_mode,
        rag_class_labels_by_id=rag_class_labels,
        regression_transition=regression_transition,
    )

    _collate = fast_collate if os.environ.get("PROBE_FAST_COLLATE") == "1" else None
    train_loader = DataLoader(
        train_ds,
        batch_size=probe_cfg["batch_size"],
        shuffle=True,
        num_workers=probe_cfg.get("num_workers", 4),
        pin_memory=bool(probe_cfg.get("pin_memory", torch.cuda.is_available())),
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=probe_cfg["batch_size"],
        shuffle=False,
        num_workers=probe_cfg.get("num_workers", 4),
        pin_memory=bool(probe_cfg.get("pin_memory", torch.cuda.is_available())),
        collate_fn=_collate,
    )

    
    model_cls = QuestionAnswerDualProbeCNN if architecture == "question_answer_dual_cnn" else RAGProbeCNN
    log(f"Probe architecture: {architecture}; model_cls={model_cls.__name__}; num_attn_channels={num_attn_channels}; prob_channels={probe_cfg.get('prob_channels', 3)}")
    model_kwargs = {
        "embed_weights": embed_weights,
        "num_attn_channels": num_attn_channels,
        "prob_channels": probe_cfg.get("prob_channels", 3),
        "hidden_dim": probe_cfg.get("hidden_dim", 64),
        "noise_std": probe_cfg.get("noise_std", 0.05),
        "dropout": probe_cfg.get("dropout", 0.15),
        "head_dropout": probe_cfg.get("head_dropout", 0.1),
        "use_head_gate": probe_cfg.get("use_head_gate", True),
        "num_rag_classes": len(RAG_SIGNAL_CLASSES),
    }
    if model_cls is QuestionAnswerDualProbeCNN:
        model_kwargs["use_attention_branch"] = probe_cfg.get("use_attention_branch", True)
    model = model_cls(**model_kwargs).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(probe_cfg["learning_rate"]),
        weight_decay=float(probe_cfg.get("weight_decay", 1e-2)),
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    aux_binary_loss_weight = float(probe_cfg.get("aux_binary_loss_weight", 0.5))
    aux_pos_weight = _aux_binary_pos_weight(train_meta).to(device)
    aux_binary_criterion = nn.BCEWithLogitsLoss(pos_weight=aux_pos_weight)
    log(
        "Aux binary tasks: names=['with_context_correct', 'rag_harmful'], "
        f"loss_weight={aux_binary_loss_weight:.4f}, "
        f"pos_weight={[round(float(x), 4) for x in aux_pos_weight.detach().cpu().tolist()]}"
    )
    rag_class_criterion = nn.CrossEntropyLoss(
        weight=rag_class_weights.to(device) if rag_class_weights is not None else None,
        ignore_index=RAG_CLASS_IGNORE_INDEX,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(probe_cfg.get("lr_scheduler_factor", 0.5)),
        patience=int(probe_cfg.get("lr_scheduler_patience", 4)),
        min_lr=float(probe_cfg.get("min_learning_rate", 1e-7)),
    )

    checkpoint_path = Path(config["output"]["checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_auc = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    early_stopping_patience = int(probe_cfg.get("early_stopping_patience", 12))
    early_stopping_min_delta = float(probe_cfg.get("early_stopping_min_delta", 1e-4))
    show_progress = bool(probe_cfg.get("show_progress", sys.stderr.isatty()))
    val_prediction_path = config.get("output", {}).get("val_prediction_path")

    for epoch in range(1, int(probe_cfg["epochs"]) + 1):
        model.train()
        total_loss = 0.0

        total_binary_loss = 0.0
        total_aux_binary_loss = 0.0
        total_rag_class_loss = 0.0
        train_rag_class_batches = 0

        for features, tokens, targets, rag_class_targets, aux_binary_targets, _ in tqdm(
            train_loader,
            desc=f"Epoch {epoch:03d} train",
            leave=False,
            disable=not show_progress,
        ):
            features = features.to(device)
            tokens = tokens.to(device)
            targets = targets.to(device)
            rag_class_targets = rag_class_targets.to(device)
            aux_binary_targets = aux_binary_targets.to(device)
            pad_mask = tokens == tokenizer.pad_token_id

            optimizer.zero_grad()
            if rag_class_loss_weight > 0 or aux_binary_loss_weight > 0:
                outputs = model(features, tokens, pad_mask=pad_mask, return_multitask=True)
                logits = outputs["binary_logits"]
                binary_loss = criterion(logits, targets)
                aux_binary_loss = aux_binary_criterion(outputs["aux_binary_logits"], aux_binary_targets)
                valid_rag_targets = (rag_class_targets != RAG_CLASS_IGNORE_INDEX).sum().item()
                if valid_rag_targets > 0:
                    rag_class_loss = rag_class_criterion(outputs["rag_class_logits"], rag_class_targets)
                else:
                    rag_class_loss = outputs["rag_class_logits"].sum() * 0.0
                loss = binary_loss + aux_binary_loss_weight * aux_binary_loss + rag_class_loss_weight * rag_class_loss
                if valid_rag_targets > 0:
                    total_rag_class_loss += float(rag_class_loss.item())
                    train_rag_class_batches += 1
            else:
                logits = model(features, tokens, pad_mask=pad_mask)
                binary_loss = criterion(logits, targets)
                aux_binary_loss = binary_loss * 0.0
                loss = binary_loss
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_binary_loss += float(binary_loss.item())
            total_aux_binary_loss += float(aux_binary_loss.item())

        model.eval()
        y_true, y_score = [], []
        rag_class_true, rag_class_pred = [], []
        val_pred_records = []

        with torch.no_grad():
            for features, tokens, targets, rag_class_targets, aux_binary_targets, indices in tqdm(
                val_loader,
                desc=f"Epoch {epoch:03d} val",
                leave=False,
                disable=not show_progress,
            ):
                features = features.to(device)
                tokens = tokens.to(device)
                rag_class_targets = rag_class_targets.to(device)
                aux_binary_targets = aux_binary_targets.to(device)
                pad_mask = tokens == tokenizer.pad_token_id

                outputs = model(features, tokens, pad_mask=pad_mask, return_multitask=True)
                logits = outputs["binary_logits"]
                rag_class_logits = outputs["rag_class_logits"]
                aux_binary_probs = torch.sigmoid(outputs["aux_binary_logits"]).cpu().numpy().tolist()
                probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy().tolist()
                if not isinstance(probs, list):
                    probs = [probs]

                class_preds = torch.argmax(rag_class_logits, dim=-1).cpu().numpy().tolist()
                class_labels = rag_class_targets.cpu().numpy().tolist()
                if not isinstance(class_preds, list):
                    class_preds = [class_preds]
                if not isinstance(class_labels, list):
                    class_labels = [class_labels]

                labels = targets.squeeze(-1).numpy().tolist()
                if not isinstance(labels, list):
                    labels = [labels]

                y_score.extend(probs)
                y_true.extend(labels)

                for true_class, pred_class in zip(class_labels, class_preds):
                    if true_class != RAG_CLASS_IGNORE_INDEX:
                        rag_class_true.append(int(true_class))
                        rag_class_pred.append(int(pred_class))

                aux_binary_labels = aux_binary_targets.cpu().numpy().tolist()
                for prob, label, true_class, pred_class, aux_probs, aux_labels, idx in zip(
                    probs,
                    labels,
                    class_labels,
                    class_preds,
                    aux_binary_probs,
                    aux_binary_labels,
                    indices.numpy().tolist(),
                ):
                    item = val_meta[idx]
                    val_pred_records.append({
                        "idx": int(idx),
                        "transition": _get_transition(item),
                        "label": int(label),
                        "helpful_probability": float(prob),
                        "with_context_correct_probability": float(aux_probs[0]),
                        "rag_harmful_probability": float(aux_probs[1]),
                        "with_context_correct_label": int(aux_labels[0]),
                        "rag_harmful_label": int(aux_labels[1]),
                        "rag_signal_label": int(true_class),
                        "rag_signal_label_name": (
                            RAG_SIGNAL_CLASSES[int(true_class)]
                            if true_class != RAG_CLASS_IGNORE_INDEX
                            else None
                        ),
                        "rag_signal_prediction": int(pred_class),
                        "rag_signal_prediction_name": RAG_SIGNAL_CLASSES[int(pred_class)],
                    })

        metrics = compute_metrics(y_true, y_score)
        rag_class_metrics = compute_classification_metrics(rag_class_true, rag_class_pred)
        avg_loss = total_loss / max(len(train_loader), 1)
        avg_binary_loss = total_binary_loss / max(len(train_loader), 1)
        avg_aux_binary_loss = total_aux_binary_loss / max(len(train_loader), 1)
        avg_rag_class_loss = total_rag_class_loss / max(train_rag_class_batches, 1)
        log(
            f"Epoch {epoch:03d} | loss={avg_loss:.4f} | binary_loss={avg_binary_loss:.4f} "
            f"| aux_binary_loss={avg_aux_binary_loss:.4f} | rag_class_loss={avg_rag_class_loss:.4f} | auc={metrics['auc']:.4f} "
            f"| f1={metrics['f1']:.4f} | rag_acc={rag_class_metrics['accuracy']:.4f} "
            f"| rag_macro_f1={rag_class_metrics['macro_f1']:.4f} "
            f"| lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        log(
            "RAG signal val prediction distribution: "
            f"{_named_pred_distribution(rag_class_metrics['pred_distribution'])}"
        )

        scheduler.step(metrics["auc"])

        if metrics["auc"] > best_auc + early_stopping_min_delta:
            best_auc = metrics["auc"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_name": model_name,
                    "architecture": architecture,
                    "num_attn_channels": num_attn_channels,
                    "prob_channels": probe_cfg.get("prob_channels", 3),
                    "use_attention_branch": probe_cfg.get("use_attention_branch", True),
                    "hidden_dim": probe_cfg.get("hidden_dim", 64),
                    "max_context_len": probe_cfg["max_context_len"],
                    "positive_transition": positive_transition,
                    "label_mode": label_mode,
                    "rag_signal_classes": RAG_SIGNAL_CLASSES,
                    "aux_binary_loss_weight": aux_binary_loss_weight,
                    "rag_class_loss_weight": rag_class_loss_weight,
                    "rag_class_regression_transition": regression_transition,
                    "rag_class_weighting": rag_class_weighting,
                    "rag_class_max_weight": rag_class_max_weight,
                    "rag_class_weights": rag_class_weights.tolist() if rag_class_weights is not None else None,
                    "best_auc": best_auc,
                    "best_epoch": best_epoch,
                },
                checkpoint_path,
            )
            log(f"Saved new best checkpoint: {checkpoint_path}")

            if val_prediction_path:
                write_jsonl(val_pred_records, val_prediction_path)
        else:
            epochs_without_improvement += 1
            log(
                f"No AUC improvement for {epochs_without_improvement}/"
                f"{early_stopping_patience} epochs; best_epoch={best_epoch}, best_auc={best_auc:.4f}"
            )

        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            log(
                f"Early stopping at epoch={epoch}; "
                f"best_epoch={best_epoch}; best_auc={best_auc:.4f}"
            )
            break

    log(f"Done. Best epoch={best_epoch}; best AUC={best_auc:.4f}")
