from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import roc_curve, auc
from transformers import AutoTokenizer

from ragprobe.training.dataset import (
    RAG_CLASS_IGNORE_INDEX,
    RAG_SIGNAL_CLASSES,
    RAGFeatureDataset,
    QuestionAnswerSidecarDataset,
    ContextQuestionAnswerSidecarDataset,
    load_feature_meta,
    load_rag_signal_label_map,
)
from ragprobe.training.model import ContextQuestionAnswerModalityEnsembleProbe, ContextQuestionAnswerModalityGatedProbe, ContextQuestionAnswerResNetProbe, FrozenQuestionAnswerEncoderFusionProbe, FrozenContextQuestionAnswerEncoderFusionProbe, FrozenQuestionAnswerTokenFusionProbe, QuestionAnswerDualProbeCNN, QuestionAnswerModalityEnsembleProbe, QuestionAnswerModalityGatedProbe, QuestionAnswerResNetProbe, RAGProbeCNN
from ragprobe.training.trainer import (
    compute_classification_metrics,
    extract_qwen_embeddings,
    compute_metrics,
    _build_question_answer_frozen_fusion,
    _build_context_question_answer_frozen_fusion,
    _build_question_answer_frozen_token_fusion,
    _question_dataset_from_encoder_ckpt,
    _context_dataset_from_encoder_ckpt,
    QuestionAnswerFrozenFusionDataset,
    ContextQuestionAnswerFrozenFusionDataset,
    _move_branch_batches,
    _embedding_weights_for_question_fusion,
)
from ragprobe.utils.io import write_jsonl
from ragprobe.utils.logging import log


def _load_matching_state_dict(model, checkpoint_state_dict):
    model_state = model.state_dict()
    compatible_state = {}
    skipped_keys = []

    for key, value in checkpoint_state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            compatible_state[key] = value
        else:
            skipped_keys.append(key)

    missing_keys, unexpected_keys = model.load_state_dict(compatible_state, strict=False)
    return missing_keys, unexpected_keys, skipped_keys



def _write_roc_outputs(y_true, y_score, output_cfg):
    roc_json_path = output_cfg.get("roc_curve_path") or output_cfg.get("roc_json_path")
    roc_png_path = output_cfg.get("roc_plot_path")
    if not roc_json_path and not roc_png_path:
        pred_path = output_cfg.get("prediction_path")
        if pred_path:
            roc_json_path = str(Path(pred_path).with_suffix(".roc.json"))
            roc_png_path = str(Path(pred_path).with_suffix(".roc.png"))
    if not y_true or len(set(int(x) for x in y_true)) < 2:
        roc_data = {"fpr": [], "tpr": [], "thresholds": [], "auc": 0.0, "error": "need both classes"}
    else:
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_data = {
            "fpr": [float(x) for x in fpr],
            "tpr": [float(x) for x in tpr],
            "thresholds": [float(x) for x in thresholds],
            "auc": float(auc(fpr, tpr)),
        }
    if roc_json_path:
        path = Path(roc_json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(roc_data, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"ROC curve data saved to {path}")
    if roc_png_path and roc_data.get("fpr"):
        try:
            import matplotlib.pyplot as plt
            path = Path(roc_png_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            plt.figure(figsize=(5, 5))
            plt.plot(roc_data["fpr"], roc_data["tpr"], label=f"AUC={roc_data['auc']:.4f}")
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(path, dpi=160)
            plt.close()
            log(f"ROC curve plot saved to {path}")
        except Exception as exc:
            log(f"Could not save ROC plot: {exc}")
    return roc_data


def _evaluate_question_answer_resnet(config, checkpoint, meta, tokenizer, embed_weights, device):
    data_cfg = config["data"]
    probe_cfg = config["probe"]
    architecture = checkpoint.get("architecture", probe_cfg.get("architecture", "question_answer_resnet"))
    if architecture == "question_answer_modality_gated":
        model_cls = QuestionAnswerModalityGatedProbe
    elif architecture == "question_answer_modality_ensemble":
        model_cls = QuestionAnswerModalityEnsembleProbe
    else:
        model_cls = QuestionAnswerResNetProbe
    log(f"QuestionAnswer eval model class: {model_cls.__name__}")
    model = model_cls(
        embed_weights=embed_weights,
        numeric_channels=int(checkpoint["numeric_channels"]),
        attention_channels=int(checkpoint["attention_channels"]),
        hidden_dim=int(checkpoint.get("hidden_dim", probe_cfg.get("hidden_dim", 128))),
        embed_proj_dim=int(checkpoint.get("embed_proj_dim", probe_cfg.get("embed_proj_dim", 32))),
        llm_hidden_dim=int(checkpoint.get("llm_hidden_dim", probe_cfg.get("llm_hidden_dim", embed_weights.shape[1]))),
        hidden_proj_dim=int(checkpoint.get("hidden_proj_dim", probe_cfg.get("hidden_proj_dim", checkpoint.get("embed_proj_dim", probe_cfg.get("embed_proj_dim", 32))))),
        num_res_blocks=int(checkpoint.get("num_res_blocks", probe_cfg.get("num_res_blocks", 3))),
        dropout=float(checkpoint.get("dropout", probe_cfg.get("dropout", 0.1))),
        noise_std=0.0,
        use_token_embedding=checkpoint.get("use_token_embedding", probe_cfg.get("use_token_embedding", True)),
        use_last_hidden=checkpoint.get("use_last_hidden", probe_cfg.get("use_last_hidden", True)),
        **({
            "aux_hidden_dim": checkpoint.get("aux_hidden_dim", probe_cfg.get("aux_hidden_dim")),
            "aux_num_res_blocks": checkpoint.get("aux_num_res_blocks", probe_cfg.get("aux_num_res_blocks", 2)),
            "prob_feature_channels": checkpoint.get("prob_feature_channels", probe_cfg.get("prob_feature_channels", 5)),
            "use_probability_features": checkpoint.get("use_probability_features", probe_cfg.get("use_probability_features", True)),
            "use_residual_features": checkpoint.get("use_residual_features", probe_cfg.get("use_residual_features", True)),
            "use_attention_features": checkpoint.get("use_attention_features", probe_cfg.get("use_attention_features", True)),
            "modality_dropout": 0.0,
        } if model_cls in {QuestionAnswerModalityGatedProbe, QuestionAnswerModalityEnsembleProbe} else {}),
    ).to(device)
    missing_keys, unexpected_keys, skipped_keys = _load_matching_state_dict(model, checkpoint["state_dict"])
    if missing_keys:
        log(f"Checkpoint missing model keys: {missing_keys}")
    if unexpected_keys:
        log(f"Checkpoint unexpected model keys: {unexpected_keys}")
    if skipped_keys:
        log(f"Checkpoint skipped incompatible model keys: {skipped_keys}")
    model.eval()

    ds = QuestionAnswerSidecarDataset(
        meta,
        tokenizer,
        question_max_len=checkpoint.get("question_max_len", probe_cfg.get("question_max_len", 128)),
        answer_max_len=checkpoint.get("answer_max_len", probe_cfg.get("answer_max_len", 32)),
        positive_transition=checkpoint.get("positive_transition", probe_cfg.get("positive_transition", "NY")),
        label_mode=checkpoint.get("label_mode", probe_cfg.get("label_mode", "without_context_correct")),
        use_numeric_features=checkpoint.get("use_numeric_features", probe_cfg.get("use_numeric_features", True)),
        use_probability_features=checkpoint.get("use_probability_features", probe_cfg.get("use_probability_features", True)),
        use_residual_features=checkpoint.get("use_residual_features", probe_cfg.get("use_residual_features", True)),
        use_attention_features=checkpoint.get("use_attention_features", probe_cfg.get("use_attention_features", True)),
        use_last_hidden=checkpoint.get("use_last_hidden", probe_cfg.get("use_last_hidden", True)),
        use_attention_mass=checkpoint.get("use_attention_mass", probe_cfg.get("use_attention_mass", False)),
    )
    loader = DataLoader(ds, batch_size=probe_cfg.get("batch_size", 1), shuffle=False, num_workers=0)
    y_true, y_score = [], []
    prediction_records = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Probe eval"):
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
            if model_cls is QuestionAnswerModalityEnsembleProbe:
                outputs = model(q_num, q_attn, q_tok, q_hidden, q_mask, a_num, a_attn, a_tok, a_hidden, a_mask, return_multitask=True)
                logits = outputs["binary_logits"]
                modality_probs_batch = torch.sigmoid(outputs["modality_logits"]).cpu().numpy().tolist()
                modality_names = list(outputs.get("modality_names", []))
            else:
                logits = model(q_num, q_attn, q_tok, q_hidden, q_mask, a_num, a_attn, a_tok, a_hidden, a_mask)
                modality_probs_batch = None
                modality_names = []
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy().tolist()
            labels = targets.squeeze(-1).numpy().tolist()
            if not isinstance(probs, list):
                probs = [probs]
            if not isinstance(labels, list):
                labels = [labels]
            if modality_probs_batch is None:
                modality_probs_iter = [None] * len(probs)
            else:
                modality_probs_iter = modality_probs_batch
            for prob, label, mprob, idx in zip(probs, labels, modality_probs_iter, indices.numpy().tolist()):
                item = meta[int(idx)]
                ex = item.get("example") or {}
                y_true.append(float(label))
                y_score.append(float(prob))
                rec = {
                    "example_id": ex.get("id"),
                    "dataset": ex.get("dataset"),
                    "split": ex.get("split"),
                    "question": ex.get("question"),
                    "answers": ex.get("answers"),
                    "transition": item.get("derived", {}).get("transition"),
                    "label": int(label),
                    "without_context_correct_probability": float(prob),
                    "helpful_probability": float(prob),
                    "prediction": int(float(prob) >= 0.5),
                    "source_meta": item,
                }
                if mprob is not None:
                    rec["modality_probabilities"] = {name: float(value) for name, value in zip(modality_names, mprob)}
                prediction_records.append(rec)
    metrics = compute_metrics(y_true, y_score)
    roc_data = _write_roc_outputs(y_true, y_score, config.get("output", {}))
    output_path = config["output"]["prediction_path"]
    write_jsonl(prediction_records, output_path)
    log(f"Eval metrics: {metrics}")
    log(f"Predictions saved to {output_path}")
    return prediction_records, {"binary": metrics, "roc_curve": roc_data}


def _evaluate_context_question_answer_resnet(config, checkpoint, meta, tokenizer, embed_weights, device, rag_class_labels):
    probe_cfg = config["probe"]
    rag_signal_classes = tuple(checkpoint.get("rag_signal_classes", RAG_SIGNAL_CLASSES))
    architecture = checkpoint.get("architecture", probe_cfg.get("architecture", "context_question_answer_resnet"))
    if architecture == "context_question_answer_modality_gated":
        model_cls = ContextQuestionAnswerModalityGatedProbe
    elif architecture == "context_question_answer_modality_ensemble":
        model_cls = ContextQuestionAnswerModalityEnsembleProbe
    else:
        model_cls = ContextQuestionAnswerResNetProbe
    log(f"ContextQuestionAnswer eval model class: {model_cls.__name__}")
    model = model_cls(
        embed_weights=embed_weights,
        numeric_channels=int(checkpoint["numeric_channels"]),
        attention_channels=int(checkpoint["attention_channels"]),
        hidden_dim=int(checkpoint.get("hidden_dim", probe_cfg.get("hidden_dim", 128))),
        embed_proj_dim=int(checkpoint.get("embed_proj_dim", probe_cfg.get("embed_proj_dim", 32))),
        llm_hidden_dim=int(checkpoint.get("llm_hidden_dim", probe_cfg.get("llm_hidden_dim", embed_weights.shape[1]))),
        hidden_proj_dim=int(checkpoint.get("hidden_proj_dim", probe_cfg.get("hidden_proj_dim", checkpoint.get("embed_proj_dim", probe_cfg.get("embed_proj_dim", 32))))),
        num_res_blocks=int(checkpoint.get("num_res_blocks", probe_cfg.get("num_res_blocks", 3))),
        dropout=float(checkpoint.get("dropout", probe_cfg.get("dropout", 0.1))),
        noise_std=0.0,
        num_rag_classes=len(rag_signal_classes),
        use_token_embedding=checkpoint.get("use_token_embedding", probe_cfg.get("use_token_embedding", True)),
        use_last_hidden=checkpoint.get("use_last_hidden", probe_cfg.get("use_last_hidden", True)),
        **({
            "aux_hidden_dim": checkpoint.get("aux_hidden_dim", probe_cfg.get("aux_hidden_dim")),
            "aux_num_res_blocks": checkpoint.get("aux_num_res_blocks", probe_cfg.get("aux_num_res_blocks", 2)),
            "prob_feature_channels": checkpoint.get("prob_feature_channels", probe_cfg.get("prob_feature_channels", 5)),
            "use_probability_features": checkpoint.get("use_probability_features", probe_cfg.get("use_probability_features", True)),
            "use_residual_features": checkpoint.get("use_residual_features", probe_cfg.get("use_residual_features", True)),
            "use_attention_features": checkpoint.get("use_attention_features", probe_cfg.get("use_attention_features", True)),
            "modality_dropout": 0.0,
        } if model_cls in {ContextQuestionAnswerModalityGatedProbe, ContextQuestionAnswerModalityEnsembleProbe} else {}),
    ).to(device)
    missing_keys, unexpected_keys, skipped_keys = _load_matching_state_dict(model, checkpoint["state_dict"])
    if missing_keys:
        log(f"Checkpoint missing model keys: {missing_keys}")
    if unexpected_keys:
        log(f"Checkpoint unexpected model keys: {unexpected_keys}")
    if skipped_keys:
        log(f"Checkpoint skipped incompatible model keys: {skipped_keys}")
    model.eval()
    label_mode = checkpoint.get("label_mode", probe_cfg.get("label_mode", "transition_positive"))
    binary_target_name = checkpoint.get("binary_target_name") or ("with_context_correct" if label_mode == "with_context_correct" else "helpful")

    ds = ContextQuestionAnswerSidecarDataset(
        meta,
        tokenizer,
        context_max_len=checkpoint.get("context_max_len", checkpoint.get("max_context_len", probe_cfg.get("max_context_len", 2048))),
        question_max_len=checkpoint.get("question_max_len", probe_cfg.get("question_max_len", 128)),
        answer_max_len=checkpoint.get("answer_max_len", probe_cfg.get("answer_max_len", 32)),
        positive_transition=checkpoint.get("positive_transition", probe_cfg.get("positive_transition", "NY")),
        label_mode=label_mode,
        rag_class_labels_by_id=rag_class_labels,
        regression_transition=checkpoint.get("rag_class_regression_transition", probe_cfg.get("rag_class_regression_transition", "YN")),
        use_numeric_features=checkpoint.get("use_numeric_features", probe_cfg.get("use_numeric_features", True)),
        use_probability_features=checkpoint.get("use_probability_features", probe_cfg.get("use_probability_features", True)),
        use_residual_features=checkpoint.get("use_residual_features", probe_cfg.get("use_residual_features", True)),
        use_attention_features=checkpoint.get("use_attention_features", probe_cfg.get("use_attention_features", True)),
        use_last_hidden=checkpoint.get("use_last_hidden", probe_cfg.get("use_last_hidden", True)),
        use_attention_mass=checkpoint.get("use_attention_mass", probe_cfg.get("use_attention_mass", False)),
    )
    loader = DataLoader(ds, batch_size=probe_cfg.get("batch_size", 1), shuffle=False, num_workers=0)
    y_true, y_score = [], []
    rag_class_true, rag_class_pred = [], []
    prediction_records = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Probe eval"):
            (c_num, c_attn, c_tok, c_hidden, c_mask, q_num, q_attn, q_tok, q_hidden, q_mask, a_num, a_attn, a_tok, a_hidden, a_mask, targets, rag_targets, aux_targets, indices) = batch
            c_num, c_attn, c_tok, c_hidden, c_mask = c_num.to(device), c_attn.to(device), c_tok.to(device), c_hidden.to(device=device, dtype=torch.float32), c_mask.to(device)
            q_num, q_attn, q_tok, q_hidden, q_mask = q_num.to(device), q_attn.to(device), q_tok.to(device), q_hidden.to(device=device, dtype=torch.float32), q_mask.to(device)
            a_num, a_attn, a_tok, a_hidden, a_mask = a_num.to(device), a_attn.to(device), a_tok.to(device), a_hidden.to(device=device, dtype=torch.float32), a_mask.to(device)
            rag_targets = rag_targets.to(device)
            aux_targets = aux_targets.to(device)
            outputs = model(c_num, c_attn, c_tok, c_hidden, c_mask, q_num, q_attn, q_tok, q_hidden, q_mask, a_num, a_attn, a_tok, a_hidden, a_mask, return_multitask=True)
            if "modality_logits" in outputs:
                modality_probs_batch = torch.sigmoid(outputs["modality_logits"]).cpu().numpy().tolist()
                modality_names = list(outputs.get("modality_names", []))
            else:
                modality_probs_batch = None
                modality_names = []
            probs = torch.sigmoid(outputs["binary_logits"]).squeeze(-1).cpu().numpy().tolist()
            labels = targets.squeeze(-1).numpy().tolist()
            if not isinstance(probs, list): probs = [probs]
            if not isinstance(labels, list): labels = [labels]
            rag_probs = torch.softmax(outputs["rag_class_logits"], dim=-1).cpu().numpy().tolist()
            rag_preds = torch.argmax(outputs["rag_class_logits"], dim=-1).cpu().numpy().tolist()
            rag_labels = rag_targets.cpu().numpy().tolist()
            aux_probs = torch.sigmoid(outputs["aux_binary_logits"]).cpu().numpy().tolist()
            aux_labels = aux_targets.cpu().numpy().tolist()
            for true_class, pred_class in zip(rag_labels, rag_preds):
                if true_class != RAG_CLASS_IGNORE_INDEX:
                    rag_class_true.append(int(true_class)); rag_class_pred.append(int(pred_class))
            if modality_probs_batch is None:
                modality_probs_iter = [None] * len(probs)
            else:
                modality_probs_iter = modality_probs_batch
            for prob, label, class_prob, true_class, pred_class, ap, al, mprob, idx in zip(probs, labels, rag_probs, rag_labels, rag_preds, aux_probs, aux_labels, modality_probs_iter, indices.numpy().tolist()):
                item = meta[int(idx)]
                ex = item.get("example") or {}
                y_true.append(float(label)); y_score.append(float(prob))
                rec = {
                    "example_id": ex.get("id"),
                    "dataset": ex.get("dataset"),
                    "split": ex.get("split"),
                    "question": ex.get("question"),
                    "answers": ex.get("answers"),
                    "transition": item.get("derived", {}).get("transition"),
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
                    "rag_signal_label_name": rag_signal_classes[int(true_class)] if true_class != RAG_CLASS_IGNORE_INDEX else None,
                    "rag_signal_prediction": int(pred_class),
                    "rag_signal_prediction_name": rag_signal_classes[int(pred_class)],
                    "rag_signal_probabilities": {name: float(value) for name, value in zip(rag_signal_classes, class_prob)},
                    "source_meta": item,
                }
                if mprob is not None:
                    rec["modality_probabilities"] = {name: float(value) for name, value in zip(modality_names, mprob)}
                prediction_records.append(rec)
    metrics = compute_metrics(y_true, y_score)
    roc_data = _write_roc_outputs(y_true, y_score, config.get("output", {}))
    rag_class_metrics = compute_classification_metrics(rag_class_true, rag_class_pred)
    output_path = config["output"]["prediction_path"]
    write_jsonl(prediction_records, output_path)
    log(f"Eval metrics: {metrics}")
    log(f"RAG signal eval metrics: {rag_class_metrics}")
    log(f"Predictions saved to {output_path}")
    return prediction_records, {"binary": metrics, "rag_signal": rag_class_metrics, "roc_curve": roc_data}


def _evaluate_question_answer_frozen_encoder_fusion(config, checkpoint, meta, tokenizer, embed_weights, device):
    probe_cfg = config["probe"]
    output_cfg = config.get("output", {})
    positive_transition = checkpoint.get("positive_transition", probe_cfg.get("positive_transition", "NY"))
    label_mode = checkpoint.get("label_mode", probe_cfg.get("label_mode", "without_context_correct"))
    eval_config = {
        "model_name": config["model_name"],
        "probe": {
            **probe_cfg,
            "branch_checkpoints": checkpoint.get("branch_checkpoints", probe_cfg.get("branch_checkpoints", [])),
            "fusion_hidden_dim": checkpoint.get("hidden_dim", probe_cfg.get("fusion_hidden_dim", probe_cfg.get("hidden_dim", 128))),
            "dropout": checkpoint.get("dropout", probe_cfg.get("dropout", 0.1)),
            "fusion_num_res_blocks": checkpoint.get("fusion_num_res_blocks", probe_cfg.get("fusion_num_res_blocks", 3)),
            "modality_dropout": 0.0,
        },
    }
    architecture = checkpoint.get("architecture", "question_answer_frozen_encoder_fusion")
    is_context_fusion = architecture == "context_question_answer_frozen_encoder_fusion"
    if is_context_fusion:
        model, branch_specs, branch_ckpts = _build_context_question_answer_frozen_fusion(eval_config, embed_weights, device)
    elif architecture == "question_answer_frozen_token_fusion":
        model, branch_specs, branch_ckpts = _build_question_answer_frozen_token_fusion(eval_config, embed_weights, device)
    else:
        model, branch_specs, branch_ckpts = _build_question_answer_frozen_fusion(eval_config, embed_weights, device)
    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
    log(f"Loaded frozen fusion head; missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    branch_names = [spec["name"] for spec in branch_specs]
    dataset_factory = _context_dataset_from_encoder_ckpt if is_context_fusion else _question_dataset_from_encoder_ckpt
    wrapper_cls = ContextQuestionAnswerFrozenFusionDataset if is_context_fusion else QuestionAnswerFrozenFusionDataset
    branch_datasets = {name: dataset_factory(meta, tokenizer, branch_ckpts[name], positive_transition, label_mode) for name in branch_names}
    ds = wrapper_cls(branch_datasets)
    loader = DataLoader(
        ds,
        batch_size=int(probe_cfg.get("batch_size", 512)),
        shuffle=False,
        num_workers=int(probe_cfg.get("num_workers", 0)),
        pin_memory=bool(probe_cfg.get("pin_memory", torch.cuda.is_available())),
    )
    records = []
    y_true, y_score = [], []
    with torch.no_grad():
        for branch_batches, targets, indices in tqdm(loader, desc="Probe eval"):
            branch_batches = _move_branch_batches(branch_batches, device)
            targets = targets.to(device=device, dtype=torch.float32)
            outputs = model(branch_batches, return_multitask=True)
            probs = torch.sigmoid(outputs["binary_logits"]).cpu().numpy().reshape(-1)
            modality_probs = torch.sigmoid(outputs["modality_logits"]).cpu().numpy().tolist()
            labels = targets.cpu().numpy().reshape(-1).astype(int).tolist()
            y_true.extend(labels)
            y_score.extend([float(p) for p in probs])
            idx_list = indices.tolist() if torch.is_tensor(indices) else list(indices)
            for local_i, meta_idx in enumerate(idx_list):
                item = meta[int(meta_idx)]
                example = item.get("example", {})
                rec = {
                    "example_id": example.get("id") or item.get("example_id"),
                    "dataset": example.get("dataset", item.get("dataset")),
                    "split": example.get("split", item.get("split")),
                    "question": example.get("question", item.get("question")),
                    "answers": example.get("answers", item.get("answers")),
                    "transition": item.get("derived", {}).get("transition", item.get("transition")),
                    "label": int(labels[local_i]),
                    "binary_target": checkpoint.get("binary_target_name", "with_context_correct" if is_context_fusion else "without_context_correct"),
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
                records.append(rec)
    metrics = compute_metrics(y_true, y_score)
    prediction_path = output_cfg.get("prediction_path")
    if prediction_path:
        write_jsonl(records, prediction_path)
        _write_roc_outputs(y_true, y_score, output_cfg)
    log(f"Eval metrics: {metrics}")
    if prediction_path:
        log(f"Predictions saved to {prediction_path}")
    return metrics

def evaluate_probe(config: dict):
    gpu_id = config.get("gpu_id")
    if gpu_id is not None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = config["model_name"]
    data_cfg = config["data"]
    probe_cfg = config["probe"]

    checkpoint = torch.load(config["checkpoint_path"], map_location="cpu", weights_only=False)
    architecture = checkpoint.get("architecture", probe_cfg.get("architecture", "gated_cnn"))
    num_attn_channels = checkpoint.get("num_attn_channels")
    hidden_dim = checkpoint.get("hidden_dim", probe_cfg.get("hidden_dim", 64))
    prob_channels = checkpoint.get("prob_channels", probe_cfg.get("prob_channels", 3))
    use_attention_branch = checkpoint.get("use_attention_branch", probe_cfg.get("use_attention_branch", True))
    max_context_len = checkpoint.get("max_context_len", probe_cfg.get("max_context_len", probe_cfg.get("context_max_len", 1024)))
    positive_transition = checkpoint.get("positive_transition", "NY")
    label_mode = checkpoint.get("label_mode", probe_cfg.get("label_mode", "transition_positive"))
    rag_signal_classes = tuple(checkpoint.get("rag_signal_classes", RAG_SIGNAL_CLASSES))
    regression_transition = checkpoint.get(
        "rag_class_regression_transition",
        probe_cfg.get("rag_class_regression_transition", "YN"),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    meta = load_feature_meta(data_cfg["meta_paths"], splits=data_cfg.get("splits"), partition=data_cfg.get("partition"))
    log(f"Loaded eval records: {len(meta)}")
    rag_class_labels = load_rag_signal_label_map(
        data_cfg.get("rag_class_meta_paths", []),
        regression_transition=regression_transition,
    )

    if architecture in {"question_answer_frozen_encoder_fusion", "question_answer_frozen_token_fusion", "context_question_answer_frozen_encoder_fusion"}:
        branch_specs = checkpoint.get("branch_checkpoints", probe_cfg.get("branch_checkpoints", []))
        embed_weights = _embedding_weights_for_question_fusion(model_name, branch_specs)
        log(f"Probe architecture: {architecture}; using frozen late-fusion evaluator")
        return _evaluate_question_answer_frozen_encoder_fusion(config, checkpoint, meta, tokenizer, embed_weights, device)

    embed_weights = extract_qwen_embeddings(model_name)
    if architecture in {"question_answer_resnet", "question_answer_modality_gated", "question_answer_modality_ensemble"}:
        log(f"Probe architecture: {architecture}; using sidecar evaluator")
        return _evaluate_question_answer_resnet(config, checkpoint, meta, tokenizer, embed_weights, device)
    if architecture in {"context_question_answer_resnet", "context_question_answer_modality_gated", "context_question_answer_modality_ensemble"}:
        log(f"Probe architecture: {architecture}; using sidecar evaluator")
        return _evaluate_context_question_answer_resnet(config, checkpoint, meta, tokenizer, embed_weights, device, rag_class_labels)

    if num_attn_channels is None:
        raise KeyError("checkpoint is missing num_attn_channels for legacy gated_cnn evaluation")
    model_cls = QuestionAnswerDualProbeCNN if architecture == "question_answer_dual_cnn" else RAGProbeCNN
    log(f"Probe architecture: {architecture}; model_cls={model_cls.__name__}; num_attn_channels={num_attn_channels}; prob_channels={prob_channels}")
    model_kwargs = {
        "embed_weights": embed_weights,
        "num_attn_channels": num_attn_channels,
        "prob_channels": prob_channels,
        "hidden_dim": hidden_dim,
        "noise_std": 0.0,
        "num_rag_classes": len(rag_signal_classes),
    }
    if model_cls is QuestionAnswerDualProbeCNN:
        model_kwargs["use_attention_branch"] = use_attention_branch
    model = model_cls(**model_kwargs).to(device)
    missing_keys, unexpected_keys, skipped_keys = _load_matching_state_dict(
        model,
        checkpoint["state_dict"],
    )
    if missing_keys:
        log(f"Checkpoint missing model keys: {missing_keys}")
    if unexpected_keys:
        log(f"Checkpoint unexpected model keys: {unexpected_keys}")
    if skipped_keys:
        log(f"Checkpoint skipped incompatible model keys: {skipped_keys}")
    model.eval()

    ds = RAGFeatureDataset(
        meta,
        tokenizer,
        max_len=max_context_len,
        positive_transition=positive_transition,
        label_mode=label_mode,
        rag_class_labels_by_id=rag_class_labels,
        regression_transition=regression_transition,
    )
    loader = DataLoader(ds, batch_size=probe_cfg.get("batch_size", 1), shuffle=False, num_workers=0)

    y_true, y_score = [], []
    rag_class_true, rag_class_pred = [], []
    prediction_records = []

    with torch.no_grad():
        for features, tokens, targets, rag_class_targets, aux_binary_targets, indices in tqdm(loader, desc="Probe eval"):
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

            rag_class_probs = torch.softmax(rag_class_logits, dim=-1).cpu().numpy().tolist()
            if rag_class_probs and not isinstance(rag_class_probs[0], list):
                rag_class_probs = [rag_class_probs]

            class_preds = torch.argmax(rag_class_logits, dim=-1).cpu().numpy().tolist()
            class_labels = rag_class_targets.cpu().numpy().tolist()
            if not isinstance(class_preds, list):
                class_preds = [class_preds]
            if not isinstance(class_labels, list):
                class_labels = [class_labels]

            labels = targets.squeeze(-1).numpy().tolist()
            if not isinstance(labels, list):
                labels = [labels]

            for true_class, pred_class in zip(class_labels, class_preds):
                if true_class != RAG_CLASS_IGNORE_INDEX:
                    rag_class_true.append(int(true_class))
                    rag_class_pred.append(int(pred_class))

            aux_binary_labels = aux_binary_targets.cpu().numpy().tolist()
            for prob, label, true_class, pred_class, class_prob, aux_probs, aux_labels, idx in zip(
                probs,
                labels,
                class_labels,
                class_preds,
                rag_class_probs,
                aux_binary_probs,
                aux_binary_labels,
                indices.numpy().tolist(),
            ):
                item = meta[idx]
                y_true.append(float(label))
                y_score.append(float(prob))
                prediction_records.append({
                    "example_id": item["example"]["id"],
                    "dataset": item["example"].get("dataset"),
                    "split": item["example"].get("split"),
                    "question": item["example"]["question"],
                    "answers": item["example"]["answers"],
                    "transition": item.get("derived", {}).get("transition"),
                    "label": int(label),
                    "helpful_probability": float(prob),
                    "prediction": int(prob >= 0.5),
                    "with_context_correct_probability": float(aux_probs[0]),
                    "rag_harmful_probability": float(aux_probs[1]),
                    "with_context_correct_label": int(aux_labels[0]),
                    "rag_harmful_label": int(aux_labels[1]),
                    "rag_signal_label": int(true_class),
                    "rag_signal_label_name": (
                        rag_signal_classes[int(true_class)]
                        if true_class != RAG_CLASS_IGNORE_INDEX
                        else None
                    ),
                    "rag_signal_prediction": int(pred_class),
                    "rag_signal_prediction_name": rag_signal_classes[int(pred_class)],
                    "rag_signal_probabilities": {
                        name: float(value)
                        for name, value in zip(rag_signal_classes, class_prob)
                    },
                    "source_meta": item,
                })

    metrics = compute_metrics(y_true, y_score)
    roc_data = _write_roc_outputs(y_true, y_score, config.get("output", {}))
    rag_class_metrics = compute_classification_metrics(rag_class_true, rag_class_pred)
    log(f"Eval metrics: {metrics}")
    log(f"RAG signal eval metrics: {rag_class_metrics}")

    output_path = config["output"]["prediction_path"]
    write_jsonl(prediction_records, output_path)
    log(f"Predictions saved to {output_path}")
    return prediction_records, {
        "binary": metrics,
        "rag_signal": rag_class_metrics,
        "roc_curve": roc_data,
    }
