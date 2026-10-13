import html
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from ragprobe.prompts.qa import format_context
from ragprobe.training.dataset import _get_example, load_feature_meta
from ragprobe.utils.io import read_jsonl
from ragprobe.utils.logging import log

try:
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover - report generation should degrade gracefully.
    AutoTokenizer = None


TRANSITIONS = ["NN", "NY", "YN", "YY"]
RAG_CLASSES = [
    "Attention Bias",
    "Logic Drift",
    "Hierarchy Misalignment",
    "Over-correction",
    "Others",
    "Not Regression",
]
_TOKENIZER_CACHE = {}


def _safe_div(a, b):
    return a / b if b else 0.0


def _fmt(x, digits=4):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def _plot(fig, include_js=False):
    return fig.to_html(full_html=False, include_plotlyjs="cdn" if include_js else False)


def _prediction_path(config):
    output = config.get("output", {})
    if output.get("prediction_path"):
        return output["prediction_path"]
    if output.get("val_prediction_path"):
        return output["val_prediction_path"]
    raise KeyError("config.output must contain prediction_path or val_prediction_path")


def _report_path(config, prediction_path):
    output = config.get("output", {})
    if output.get("report_path"):
        return output["report_path"]
    stem = Path(prediction_path).stem.replace("_preds", "_diagnostic")
    return str(Path("outputs/reports") / f"{stem}.html")


def _load_meta_for_prediction(config):
    data = config.get("data", {})
    if data.get("meta_paths"):
        return load_feature_meta(data["meta_paths"], splits=data.get("splits"), partition=data.get("partition"))
    if data.get("validation_meta_paths"):
        return load_feature_meta(data["validation_meta_paths"], splits=data.get("validation_splits"), partition=data.get("validation_partition"))
    return []


def _enrich_records(records, config):
    if not records:
        return records
    if all(r.get("dataset") and r.get("question") for r in records):
        return records

    meta = _load_meta_for_prediction(config)
    enriched = []
    for record in records:
        out = dict(record)
        idx = out.get("idx")
        if idx is not None and 0 <= int(idx) < len(meta):
            source_meta = meta[int(idx)]
            ex = _get_example(source_meta)
            out.setdefault("source_meta", source_meta)
            out.setdefault("example_id", ex.get("id"))
            out.setdefault("dataset", ex.get("dataset"))
            out.setdefault("split", ex.get("split"))
            out.setdefault("question", ex.get("question"))
            out.setdefault("answers", ex.get("answers", []))
        enriched.append(out)
    return enriched


def _binary_metrics(records, threshold=0.5):
    y_true = [int(r.get("label", 1 if r.get("transition") == "NY" else 0)) for r in records]
    y_score = [float(r["helpful_probability"]) for r in records]
    y_pred = [1 if s >= threshold else 0 for s in y_score]
    out = {
        "n": len(records),
        "accuracy": accuracy_score(y_true, y_pred) if records else 0.0,
        "precision": precision_score(y_true, y_pred, zero_division=0) if records else 0.0,
        "recall": recall_score(y_true, y_pred, zero_division=0) if records else 0.0,
        "f1": f1_score(y_true, y_pred, zero_division=0) if records else 0.0,
        "threshold": threshold,
        "positive_rate": _safe_div(sum(y_pred), len(y_pred)),
    }
    try:
        out["auc"] = roc_auc_score(y_true, y_score)
    except ValueError:
        out["auc"] = None
    return out


def _optimal_threshold(records):
    y_true = [int(r.get("label", 1 if r.get("transition") == "NY" else 0)) for r in records]
    y_score = [float(r["helpful_probability"]) for r in records]
    if len(set(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = int(np.argmax(tpr - fpr))
    return float(thresholds[idx])


def _group_by(records, key):
    groups = defaultdict(list)
    for r in records:
        groups[r.get(key) or "UNKNOWN"].append(r)
    return dict(groups)


def _metrics_by_group(records, key, threshold=0.5):
    rows = []
    for name, group in sorted(_group_by(records, key).items()):
        m = _binary_metrics(group, threshold=threshold)
        rows.append({
            "group": name,
            **m,
            "transition_counts": dict(Counter(r.get("transition") or "UNKNOWN" for r in group)),
        })
    return rows


def _train_log_metrics(path):
    if not path or not Path(path).exists():
        return []
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    pattern = re.compile(
        r"Epoch\s+(\d+)\s+\|\s+loss=([0-9.]+)\s+\|\s+binary_loss=([0-9.]+)\s+"
        r"\|\s+rag_class_loss=([0-9.]+)\s+\|\s+auc=([0-9.]+)\s+\|\s+f1=([0-9.]+)\s+"
        r"\|\s+rag_acc=([0-9.]+)\s+\|\s+rag_macro_f1=([0-9.]+)\s+\|\s+lr=([0-9.eE+-]+)"
    )
    rows = []
    for m in pattern.finditer(text):
        rows.append({
            "epoch": int(m.group(1)),
            "loss": float(m.group(2)),
            "binary_loss": float(m.group(3)),
            "rag_class_loss": float(m.group(4)),
            "auc": float(m.group(5)),
            "f1": float(m.group(6)),
            "rag_acc": float(m.group(7)),
            "rag_macro_f1": float(m.group(8)),
            "lr": float(m.group(9)),
        })
    return rows


def _fig_training(rows):
    if not rows:
        return "<p>No training log metrics found.</p>"
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    x = [r["epoch"] for r in rows]
    for key, name in [("auc", "AUC"), ("f1", "F1"), ("rag_macro_f1", "RAG Macro F1")]:
        fig.add_trace(go.Scatter(x=x, y=[r[key] for r in rows], mode="lines+markers", name=name), secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=[r["binary_loss"] for r in rows], mode="lines", name="Binary Loss"), secondary_y=True)
    fig.add_trace(go.Scatter(x=x, y=[r["rag_class_loss"] for r in rows], mode="lines", name="RAG Class Loss"), secondary_y=True)
    fig.update_layout(title="Training Curve", height=420, margin=dict(l=40, r=40, t=50, b=40))
    fig.update_yaxes(title_text="Metric", secondary_y=False)
    fig.update_yaxes(title_text="Loss", secondary_y=True)
    return _plot(fig, include_js=True)


def _fig_roc_pr(records):
    y_true = [int(r.get("label", 1 if r.get("transition") == "NY" else 0)) for r in records]
    y_score = [float(r["helpful_probability"]) for r in records]
    if len(set(y_true)) < 2:
        return "<p>ROC/PR unavailable: only one class present.</p>"
    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)
    fig = make_subplots(rows=1, cols=2, subplot_titles=[f"ROC AUC={roc_auc:.4f}", f"PR AUC={pr_auc:.4f}"])
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Chance"), row=1, col=1)
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name="PR"), row=1, col=2)
    fig.update_xaxes(title_text="FPR", row=1, col=1)
    fig.update_yaxes(title_text="TPR", row=1, col=1)
    fig.update_xaxes(title_text="Recall", row=1, col=2)
    fig.update_yaxes(title_text="Precision", row=1, col=2)
    fig.update_layout(height=420, margin=dict(l=40, r=40, t=60, b=40))
    return _plot(fig)


def _fig_score_distribution(records):
    fig = go.Figure()
    for transition in TRANSITIONS:
        scores = [float(r["helpful_probability"]) for r in records if r.get("transition") == transition]
        if scores:
            fig.add_trace(go.Histogram(x=scores, name=transition, opacity=0.68, nbinsx=40))
    fig.update_layout(
        title="Helpful Probability Distribution by Transition",
        barmode="overlay",
        xaxis_title="helpful_probability",
        yaxis_title="Count",
        height=420,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return _plot(fig)


def _fig_box_by_dataset(records):
    fig = go.Figure()
    for dataset, group in sorted(_group_by(records, "dataset").items()):
        fig.add_trace(go.Box(
            y=[float(r["helpful_probability"]) for r in group],
            x=[r.get("transition") or "UNKNOWN" for r in group],
            name=dataset,
            boxmean=True,
        ))
    fig.update_layout(
        title="Score Distribution by Dataset and Transition",
        xaxis_title="Transition",
        yaxis_title="helpful_probability",
        boxmode="group",
        height=460,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return _plot(fig)


def _fig_confusion(records, threshold):
    labels = [0, 1]
    y_true = [int(r.get("label", 1 if r.get("transition") == "NY" else 0)) for r in records]
    y_pred = [1 if float(r["helpful_probability"]) >= threshold else 0 for r in records]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Pred Other", "Pred Helpful"],
        y=["GT Other", "GT Helpful"],
        text=cm,
        texttemplate="%{text}",
        colorscale="Blues",
    ))
    fig.update_layout(title=f"Binary Confusion Matrix @ threshold={threshold:.4f}", height=360)
    return _plot(fig)


def _fig_transition_action(records, threshold):
    z = []
    for transition in TRANSITIONS:
        group = [r for r in records if r.get("transition") == transition]
        z.append([
            sum(1 for r in group if float(r["helpful_probability"]) < threshold),
            sum(1 for r in group if float(r["helpful_probability"]) >= threshold),
        ])
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=["Pred Other/No Helpful", "Pred Helpful RAG"],
        y=TRANSITIONS,
        text=z,
        texttemplate="%{text}",
        colorscale="Viridis",
    ))
    fig.update_layout(title="Transition vs Binary Probe Decision", height=360)
    return _plot(fig)


def _fig_calibration(records, bins=10):
    rows = []
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        group = [
            r for r in records
            if lo <= float(r["helpful_probability"]) < hi or (i == bins - 1 and float(r["helpful_probability"]) == 1)
        ]
        if not group:
            continue
        rows.append({
            "center": (lo + hi) / 2,
            "confidence": sum(float(r["helpful_probability"]) for r in group) / len(group),
            "empirical": sum(int(r.get("label", 1 if r.get("transition") == "NY" else 0)) for r in group) / len(group),
            "count": len(group),
        })
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Perfect"))
    fig.add_trace(go.Scatter(
        x=[r["confidence"] for r in rows],
        y=[r["empirical"] for r in rows],
        text=[f"n={r['count']}" for r in rows],
        mode="lines+markers",
        name="Observed",
    ))
    fig.update_layout(title="Calibration: Predicted vs Empirical Helpful Rate", xaxis_title="Mean predicted p", yaxis_title="Empirical NY rate", height=420)
    return _plot(fig)


def _fig_dataset_bars(rows):
    if not rows:
        return "<p>No dataset rows.</p>"
    fig = go.Figure()
    x = [r["group"] for r in rows]
    for key in ["auc", "f1", "accuracy"]:
        fig.add_trace(go.Bar(x=x, y=[r[key] if r[key] is not None else 0 for r in rows], name=key))
    fig.update_layout(title="Performance by Dataset", barmode="group", yaxis_range=[0, 1], height=420)
    return _plot(fig)


def _rag_class_metrics(records):
    rows = [
        r for r in records
        if r.get("rag_signal_label") is not None and int(r.get("rag_signal_label")) >= 0
        and r.get("rag_signal_prediction") is not None
    ]
    if not rows:
        return None
    y_true = [int(r["rag_signal_label"]) for r in rows]
    y_pred = [int(r["rag_signal_prediction"]) for r in rows]
    return {
        "n": len(rows),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "true_counts": dict(Counter(y_true)),
        "pred_counts": dict(Counter(y_pred)),
    }


def _fig_rag_class_confusion(records):
    rows = [
        r for r in records
        if r.get("rag_signal_label") is not None and int(r.get("rag_signal_label")) >= 0
        and r.get("rag_signal_prediction") is not None
    ]
    if not rows:
        return "<p>No RAG signal classification outputs found.</p>"
    y_true = [int(r["rag_signal_label"]) for r in rows]
    y_pred = [int(r["rag_signal_prediction"]) for r in rows]
    labels = list(range(len(RAG_CLASSES)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=RAG_CLASSES,
        y=RAG_CLASSES,
        text=cm,
        texttemplate="%{text}",
        colorscale="Reds",
    ))
    fig.update_layout(title="RAG Signal Class Confusion Matrix", height=560, margin=dict(l=160, b=140, t=50, r=30))
    return _plot(fig)


def _checkpoint_head_weights(config):
    path = config.get("checkpoint_path") or config.get("output", {}).get("checkpoint_path")
    if not path or not Path(path).exists():
        return None
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint.get("state_dict", checkpoint)
        logits = state.get("head_selector.head_logits")
        if logits is None:
            return None
        weights = (2.0 * torch.sigmoid(logits)).squeeze().detach().cpu().numpy()
        return weights
    except Exception as exc:
        log(f"Could not load head weights from checkpoint: {exc}")
        return None


def _fig_head_weights(weights):
    if weights is None:
        return "<p>No head gate weights found in checkpoint.</p>"
    size = len(weights)
    cols = 16
    rows = math.ceil(size / cols)
    padded = np.full(rows * cols, np.nan)
    padded[:size] = weights
    matrix = padded.reshape(rows, cols)
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=[f"H{i}" for i in range(cols)],
        y=[f"L{i}" for i in range(rows)],
        colorscale="Viridis",
        hovertemplate="Layer %{y}<br>Head %{x}<br>Weight %{z:.4f}<extra></extra>",
    ))
    fig.update_layout(title="Global Attention Head Gate Weights", height=420, yaxis=dict(autorange="reversed"))
    return _plot(fig)


def _source_meta(record):
    return record.get("source_meta") or {}


def _tensor_path(record):
    meta = _source_meta(record)
    return meta.get("feature_extraction", {}).get("rag", {}).get("tensor_path")


def _select_case_records(records, threshold):
    out = {}
    for record in records:
        label = int(record.get("label", 1 if record.get("transition") == "NY" else 0))
        pred = 1 if float(record["helpful_probability"]) >= threshold else 0
        key = None
        if label == 1 and pred == 1:
            key = "TP"
        elif label == 0 and pred == 0:
            key = "TN"
        elif label == 0 and pred == 1:
            key = "FP"
        elif label == 1 and pred == 0:
            key = "FN"
        if key and key not in out and _tensor_path(record):
            out[key] = record
    return out


def _feature_context_text(record):
    example = _source_meta(record).get("example") or {}
    contexts = example.get("contexts") or example.get("context") or []
    if contexts:
        try:
            return format_context(contexts)
        except Exception:
            parts = []
            for ctx in contexts:
                if isinstance(ctx, dict):
                    text = ctx.get("text") or ctx.get("content") or ""
                else:
                    text = str(ctx)
                if text:
                    parts.append(text)
            return "\n".join(parts)
    return str(record.get("question") or "")


def _load_tokenizer(model_name):
    if not model_name or AutoTokenizer is None:
        return None
    if model_name in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[model_name]
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception as exc:
        log(f"Could not load local tokenizer for feature blocks ({model_name}): {exc}")
        tokenizer = None
    _TOKENIZER_CACHE[model_name] = tokenizer
    return tokenizer


def _fallback_text_tokens(text):
    tokens = re.findall(r"\s+|[^\s]+", text or "")
    return tokens or ["<empty>"]


def _context_tokens(record, config, max_tokens):
    text = _feature_context_text(record)
    tokenizer = _load_tokenizer(config.get("model_name"))
    if tokenizer is not None:
        try:
            token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            return [tokenizer.decode([tid]) for tid in token_ids[:max_tokens]]
        except Exception as exc:
            log(f"Could not tokenize context for feature blocks: {exc}")
    return _fallback_text_tokens(text)[:max_tokens]


def _display_token(token):
    shown = str(token).replace("\n", "↵").replace("\t", "⇥")
    if shown == "":
        shown = "∅"
    return html.escape(shown)


def _scale_values(values):
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr)
    lo = float(np.percentile(finite, 5))
    hi = float(np.percentile(finite, 95))
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _token_feature_strip(title, values, tokens, rgb, feature_indices):
    scaled = _scale_values(values)
    spans = []
    for token, value, rel, feature_idx in zip(tokens, values, scaled, feature_indices):
        alpha = 0.12 + 0.78 * float(rel)
        tooltip = (
            f"{title} | feature_idx={int(feature_idx)} | "
            f"value={float(value):.6f} | rel={float(rel):.3f} | token={str(token)}"
        )
        spans.append(
            f'<span class="token" style="background-color: rgba({rgb}, {alpha:.3f});" '
            f'title="{html.escape(tooltip)}">{_display_token(token)}</span>'
        )
    return (
        f"<div class='feature-strip'><h4>{html.escape(title)}</h4>"
        f"<div class='token-container'>{''.join(spans)}</div></div>"
    )


def _feature_channel_fig(record, config):
    path = _tensor_path(record)
    if not path or not Path(path).exists():
        return "<p>Feature tensor missing.</p>"
    try:
        features = torch.load(path, map_location="cpu", weights_only=True).float()
    except Exception as exc:
        return f"<p>Could not load tensor: {html.escape(str(exc))}</p>"
    if features.size(0) < 130:
        return "<p>Tensor does not include probability/stat channels.</p>"

    max_tokens = int(config.get("report", {}).get("feature_max_tokens", 220))
    n_features = int(features.size(1))
    rag_meta = _source_meta(record).get("feature_extraction", {}).get("rag", {})
    span_len = int(rag_meta.get("span_len") or n_features)
    actual_len = min(span_len, n_features)
    tokens = _context_tokens(record, config, min(max_tokens, actual_len))
    n = min(len(tokens), actual_len, max_tokens)
    if n == 0:
        return "<p>No displayable context tokens.</p>"

    # Feature tensors are stored in context-token order. Do not interpolate display
    # tokens across the padded 1024-column tensor; column i belongs to token i.
    feature_indices = np.arange(n)

    p_value = features[-2, feature_indices].numpy()
    entropy = features[-1, feature_indices].numpy()
    surprisal = (-torch.log(features[-2, feature_indices] + 1e-10)).numpy()
    attn_mean = features[:-2, feature_indices].mean(dim=0).numpy()
    display_tokens = tokens[:n]

    header = (
        f"<div class='feature-block-header'>"
        f"<strong>{html.escape(str(record.get('dataset')))} | {html.escape(str(record.get('transition')))} "
        f"| p={float(record['helpful_probability']):.4f}</strong>"
        f"<span>showing {n}/{actual_len} context tokens; tensor columns={n_features}</span></div>"
    )
    strips = [
        _token_feature_strip("Attention mean", attn_mean, display_tokens, "59, 130, 246", feature_indices),
        _token_feature_strip("P(value)", p_value, display_tokens, "16, 185, 129", feature_indices),
        _token_feature_strip("Entropy", entropy, display_tokens, "245, 158, 11", feature_indices),
        _token_feature_strip("Surprisal", surprisal, display_tokens, "239, 68, 68", feature_indices),
    ]
    return (
        "<div class='feature-blocks'>"
        f"{header}"
        "<p class='note'>Each block is one context token. Color intensity is percentile-scaled within the row; hover for exact values and aligned feature index.</p>"
        f"{''.join(strips)}</div>"
    )


def _html_table(rows, columns, max_rows=200):
    if not rows:
        return "<p>No rows.</p>"
    header = "".join(f"<th>{html.escape(str(col))}</th>" for col in columns)
    body = []
    for row in rows[:max_rows]:
        cells = []
        for col in columns:
            value = row.get(col)
            if isinstance(value, float):
                value = _fmt(value)
            cells.append(f"<td>{html.escape(str(value))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _summary_cards(primary_metrics, opt_metrics, train_rows, ood_metrics):
    best_train = max(train_rows, key=lambda r: r["auc"]) if train_rows else None
    items = [
        ("AUC @0.5", _fmt(primary_metrics.get("auc"))),
        ("F1 @0.5", _fmt(primary_metrics.get("f1"))),
        ("Opt Threshold", _fmt(opt_metrics.get("threshold"))),
        ("F1 @Opt", _fmt(opt_metrics.get("f1"))),
        ("Best Train AUC", _fmt(best_train.get("auc") if best_train else None)),
        ("Best Epoch", best_train.get("epoch") if best_train else "n/a"),
    ]
    if ood_metrics:
        items.append(("OOD PopQA AUC", _fmt(ood_metrics.get("auc"))))
        items.append(("OOD PopQA F1", _fmt(ood_metrics.get("f1"))))
    return "".join(
        f"<div class='metric'><div class='metric-label'>{html.escape(str(k))}</div>"
        f"<div class='metric-value'>{html.escape(str(v))}</div></div>"
        for k, v in items
    )


def _load_ood_records(config):
    report_cfg = config.get("report", {})
    paths = report_cfg.get("ood_prediction_paths")
    if paths is None:
        paths = ["outputs/predictions/popqa_probe_preds.jsonl"]
    records = []
    for path in paths:
        if Path(path).exists():
            for r in read_jsonl(path):
                item = dict(r)
                item.setdefault("ood_source_path", path)
                records.append(item)
    return records


def _load_triage_metrics(config):
    report_cfg = config.get("report", {})
    paths = report_cfg.get("triage_metrics_paths")
    if paths is None:
        paths = sorted(str(p) for p in Path("outputs/predictions").glob("*triage_metrics.json"))
    out = []
    for path in paths:
        if Path(path).exists():
            try:
                data = json.loads(Path(path).read_text(encoding="utf-8"))
                out.append((path, data))
            except json.JSONDecodeError:
                pass
    return out


def _triage_section(items):
    if not items:
        return "<p>No triage metrics JSON found.</p>"
    rows = []
    chart_parts = []
    for path, m in items:
        cards = [
            ("Triage Acc", m.get("triage_accuracy")),
            ("Always RAG Acc", m.get("always_rag_accuracy")),
            ("No RAG Acc", m.get("no_rag_accuracy")),
            ("NY Retention", m.get("ny_retention_rate")),
            ("YN Avoidance", m.get("yn_avoidance_rate")),
            ("RAG Usage", m.get("rag_usage_rate")),
        ]
        metric_cards = "".join(
            f"<div class='metric'><div class='metric-label'>{html.escape(label)}</div>"
            f"<div class='metric-value'>{html.escape(_fmt(value))}</div></div>"
            for label, value in cards
        )

        bar = go.Figure()
        bar.add_trace(go.Bar(
            x=["triage", "always_rag", "no_rag"],
            y=[
                m.get("triage_accuracy", 0),
                m.get("always_rag_accuracy", 0),
                m.get("no_rag_accuracy", 0),
            ],
            name="accuracy",
        ))
        bar.add_trace(go.Bar(
            x=["NY retention", "YN avoidance", "RAG usage"],
            y=[
                m.get("ny_retention_rate", 0),
                m.get("yn_avoidance_rate", 0),
                m.get("rag_usage_rate", 0),
            ],
            name="control",
        ))
        bar.update_layout(
            title=f"Triage Metrics: {Path(path).name}",
            yaxis_range=[0, 1],
            barmode="group",
            height=380,
            margin=dict(l=40, r=30, t=50, b=80),
        )

        action_counts = m.get("action_counts") or {}
        action_fig = go.Figure(data=go.Bar(
            x=list(action_counts.keys()),
            y=list(action_counts.values()),
            name="actions",
        ))
        action_fig.update_layout(
            title="Policy Action Counts",
            height=340,
            margin=dict(l=40, r=30, t=50, b=70),
        )

        action_by_transition = m.get("action_by_transition") or {}
        actions = sorted({a for counts in action_by_transition.values() for a in counts})
        transitions = [t for t in TRANSITIONS if t in action_by_transition]
        if transitions and actions:
            z = [[action_by_transition.get(t, {}).get(a, 0) for a in actions] for t in transitions]
            heat = go.Figure(data=go.Heatmap(
                z=z,
                x=actions,
                y=transitions,
                text=z,
                texttemplate="%{text}",
                colorscale="Viridis",
            ))
            heat.update_layout(
                title="Policy Action by Transition",
                height=360,
                margin=dict(l=70, r=30, t=50, b=80),
            )
            heat_html = _plot(heat)
        else:
            heat_html = "<p>No action-by-transition metrics found.</p>"

        chart_parts.append(
            f"<div class='triage-item'><h3>{html.escape(path)}</h3>"
            f"<div class='metrics'>{metric_cards}</div>"
            f"<div class='grid'>{_plot(bar)}{_plot(action_fig)}{heat_html}</div></div>"
        )
        rows.append({
            "path": path,
            "triage_accuracy": m.get("triage_accuracy"),
            "always_rag_accuracy": m.get("always_rag_accuracy"),
            "no_rag_accuracy": m.get("no_rag_accuracy"),
            "ny_retention_rate": m.get("ny_retention_rate"),
            "yn_avoidance_rate": m.get("yn_avoidance_rate"),
            "rag_usage_rate": m.get("rag_usage_rate"),
            "objective": m.get("best_objective"),
            "best_policy": json.dumps(m.get("best_policy", m.get("policy", {})), ensure_ascii=False),
        })
    table = _html_table(rows, [
        "path", "triage_accuracy", "always_rag_accuracy", "no_rag_accuracy",
        "ny_retention_rate", "yn_avoidance_rate", "rag_usage_rate", "objective", "best_policy",
    ])
    return "".join(chart_parts) + "<h3>Triage Metric Table</h3>" + table


def _case_feature_html(records, config, threshold):
    cases = _select_case_records(records, threshold=threshold)
    case_html = ""
    for name in ["TP", "TN", "FP", "FN"]:
        if name in cases:
            r = cases[name]
            case_html += (
                f"<div class='case'><h3>{name}: {html.escape(str(r.get('question', '')))}</h3>"
                f"<p>dataset={html.escape(str(r.get('dataset')))} | transition={html.escape(str(r.get('transition')))} "
                f"| p={float(r['helpful_probability']):.4f}</p>"
                f"{_feature_channel_fig(r, config)}</div>"
            )
    return case_html


def _prepare_score_rows(records, reverse=False, limit=30):
    rows = [dict(r) for r in sorted(records, key=lambda r: float(r["helpful_probability"]), reverse=reverse)[:limit]]
    for r in rows:
        r["p"] = float(r["helpful_probability"])
    return rows


def _slice_report_html(
    title,
    records,
    config,
    threshold,
    train_rows=None,
    head_weights=None,
    triage_items=None,
    include_training=False,
    include_triage=False,
):
    if not records:
        return f"<section><h2>{html.escape(title)}</h2><p>No records.</p></section>"

    opt_threshold = _optimal_threshold(records)
    primary_metrics = _binary_metrics(records, threshold=threshold)
    opt_metrics = _binary_metrics(records, threshold=opt_threshold)
    dataset_rows = _metrics_by_group(records, "dataset", threshold=threshold)
    split_rows = _metrics_by_group(records, "split", threshold=threshold)
    transition_counts = dict(Counter(r.get("transition") or "UNKNOWN" for r in records))
    rag_metrics = _rag_class_metrics(records)
    case_html = _case_feature_html(records, config, threshold)
    top_helpful = _prepare_score_rows(records, reverse=True)
    bottom_helpful = _prepare_score_rows(records, reverse=False)

    training_section = (
        f"<section><h2>Latest Training Result</h2>{_fig_training(train_rows or [])}</section>"
        if include_training else ""
    )
    head_section = (
        f"<div class='grid'>{_fig_head_weights(head_weights)}</div>"
        if head_weights is not None else ""
    )
    triage_section = (
        f"<section><h2>Triage RAG</h2>{_triage_section(triage_items or [])}</section>"
        if include_triage else ""
    )

    return f"""
  <section><h2>{html.escape(title)} Summary</h2><div class="metrics">{_summary_cards(primary_metrics, opt_metrics, train_rows or [], None)}</div></section>
  {training_section}
  <section><h2>Core Binary Performance</h2><div class="grid">{_fig_roc_pr(records)}{_fig_calibration(records)}{_fig_score_distribution(records)}{_fig_box_by_dataset(records)}</div></section>
  <section><h2>Confusion and Transition Diagnostics</h2><div class="grid">{_fig_confusion(records, threshold)}{_fig_confusion(records, opt_threshold)}{_fig_transition_action(records, threshold)}{_fig_transition_action(records, opt_threshold)}</div></section>
  <section><h2>Dataset / Split Slices</h2>{_fig_dataset_bars(dataset_rows)}{_html_table(dataset_rows, ["group", "n", "auc", "accuracy", "precision", "recall", "f1", "positive_rate", "transition_counts"])}<h3>Split Slices</h3>{_html_table(split_rows, ["group", "n", "auc", "accuracy", "precision", "recall", "f1", "positive_rate", "transition_counts"])}</section>
  <section><h2>Transition Distribution</h2>{_html_table([{"transition": k, "count": v} for k, v in sorted(transition_counts.items())], ["transition", "count"])}</section>
  <section><h2>RAG Signal Classification Head</h2><p class="note">This is diagnostic only; current triage policy may ignore this head.</p><div class="grid">{_fig_rag_class_confusion(records)}{_html_table([rag_metrics] if rag_metrics else [], ["n", "accuracy", "macro_f1", "weighted_f1", "true_counts", "pred_counts"])}</div></section>
  <section><h2>Head Gate and Feature Channels</h2>{head_section}{case_html or "<p>No feature tensors available for case telemetry.</p>"}</section>
  {triage_section}
  <section><h2>Top / Bottom Probe Scores</h2><h3>Top Helpful</h3>{_html_table(top_helpful, ["p", "transition", "dataset", "split", "question"], max_rows=30)}<h3>Bottom Helpful</h3>{_html_table(bottom_helpful, ["p", "transition", "dataset", "split", "question"], max_rows=30)}</section>
"""


def _tab_id(name):
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", str(name)).strip("-").lower()
    return cleaned or "tab"


def _tabs_html(tab_items):
    controls = []
    panels = []
    used_ids = set()
    for i, item in enumerate(tab_items):
        base_id = _tab_id(item["title"])
        tab_id = base_id
        suffix = 2
        while tab_id in used_ids:
            tab_id = f"{base_id}-{suffix}"
            suffix += 1
        used_ids.add(tab_id)
        checked = " checked" if i == 0 else ""
        controls.append(
            f'<input class="tab-input" type="radio" name="dataset-tabs" id="{tab_id}"{checked}/>'
            f'<label class="tab-label" for="{tab_id}">{html.escape(item["title"])}'
            f'<span>{int(item["count"])}</span></label>'
        )
        panels.append(f'<div class="tab-panel">{item["body"]}</div>')
    return f'<div class="tabs">{"".join(controls)}<div class="tab-panels">{"".join(panels)}</div></div>'


def make_probe_report(config: dict):
    prediction_path = _prediction_path(config)
    report_path = _report_path(config, prediction_path)
    records = _enrich_records(read_jsonl(prediction_path), config)
    if not records:
        raise RuntimeError(f"No prediction records found at {prediction_path}")

    threshold = float(config.get("report", {}).get("threshold", 0.5))
    train_log_path = config.get("report", {}).get("train_log_path", "outputs/training_runs/latest/train.log")
    train_rows = _train_log_metrics(train_log_path)

    ood_records = _load_ood_records(config)
    all_records = records + ood_records
    head_weights = _checkpoint_head_weights(config)
    triage_items = _load_triage_metrics(config)

    dataset_groups = _group_by(all_records, "dataset")
    tab_items = [{
        "title": "All",
        "count": len(all_records),
        "body": _slice_report_html(
            "All Datasets",
            all_records,
            config,
            threshold,
            train_rows=train_rows,
            head_weights=head_weights,
            triage_items=triage_items,
            include_training=True,
            include_triage=True,
        ),
    }]
    for dataset, group in sorted(dataset_groups.items()):
        tab_items.append({
            "title": str(dataset),
            "count": len(group),
            "body": _slice_report_html(
                str(dataset),
                group,
                config,
                threshold,
                train_rows=train_rows,
                head_weights=None,
                triage_items=None,
                include_training=False,
                include_triage=False,
            ),
        })
    tabs = _tabs_html(tab_items)

    css = """
    body { font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: #f7f8fb; color: #1f2937; }
    header { padding: 28px 36px; background: #111827; color: white; }
    main { padding: 24px 36px 48px; max-width: 1480px; margin: auto; }
    h1, h2, h3 { margin: 0 0 12px; }
    section { background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 18px; margin-bottom: 18px; box-shadow: 0 1px 3px rgba(15,23,42,0.05); }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap: 16px; align-items: start; }
    .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }
    .metric { background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; }
    .metric-label { font-size: 12px; color: #6b7280; }
    .metric-value { font-size: 24px; font-weight: 700; margin-top: 4px; }
    table { border-collapse: collapse; width: 100%; font-size: 13px; }
    th, td { border: 1px solid #e5e7eb; padding: 7px 8px; vertical-align: top; }
    th { background: #f3f4f6; text-align: left; }
    .note { color: #6b7280; font-size: 13px; }
    .case { border-top: 1px solid #e5e7eb; padding-top: 12px; margin-top: 12px; }
    .triage-item { border-top: 1px solid #e5e7eb; padding-top: 12px; margin-top: 12px; }
    .tabs { display: flex; flex-wrap: wrap; gap: 8px; align-items: flex-start; }
    .tab-input { position: absolute; opacity: 0; pointer-events: none; }
    .tab-label { display: inline-flex; align-items: center; gap: 8px; padding: 9px 13px; border: 1px solid #d1d5db; border-radius: 8px; background: #ffffff; color: #374151; font-size: 14px; font-weight: 700; cursor: pointer; }
    .tab-label span { padding: 1px 7px; border-radius: 999px; background: #eef2ff; color: #3730a3; font-size: 12px; }
    .tab-input:checked + .tab-label { background: #111827; color: white; border-color: #111827; }
    .tab-input:checked + .tab-label span { background: rgba(255,255,255,0.16); color: white; }
    .tab-panels { flex-basis: 100%; min-width: 0; }
    .tab-panel { display: none; margin-top: 12px; }
    .tabs > .tab-input:nth-of-type(1):checked ~ .tab-panels > .tab-panel:nth-child(1),
    .tabs > .tab-input:nth-of-type(2):checked ~ .tab-panels > .tab-panel:nth-child(2),
    .tabs > .tab-input:nth-of-type(3):checked ~ .tab-panels > .tab-panel:nth-child(3),
    .tabs > .tab-input:nth-of-type(4):checked ~ .tab-panels > .tab-panel:nth-child(4),
    .tabs > .tab-input:nth-of-type(5):checked ~ .tab-panels > .tab-panel:nth-child(5),
    .tabs > .tab-input:nth-of-type(6):checked ~ .tab-panels > .tab-panel:nth-child(6),
    .tabs > .tab-input:nth-of-type(7):checked ~ .tab-panels > .tab-panel:nth-child(7),
    .tabs > .tab-input:nth-of-type(8):checked ~ .tab-panels > .tab-panel:nth-child(8) { display: block; }
    .feature-blocks { margin-top: 10px; }
    .feature-block-header { display: flex; justify-content: space-between; gap: 12px; align-items: baseline; margin-bottom: 6px; }
    .feature-block-header span { color: #6b7280; font-size: 12px; }
    .feature-strip { margin: 12px 0 16px; }
    .feature-strip h4 { margin: 0 0 6px; font-size: 14px; }
    .token-container { display: flex; flex-wrap: wrap; gap: 2px; padding: 12px; border: 1px solid #e5e7eb; border-radius: 6px; background: #f9fafb; max-height: 260px; overflow-y: auto; line-height: 1.65; font-family: "Consolas", "Courier New", monospace; font-size: 13px; }
    .token { padding: 2px 4px; border-radius: 3px; border: 1px solid rgba(17,24,39,0.08); cursor: default; white-space: pre-wrap; transition: transform 0.12s ease, box-shadow 0.12s ease, border-color 0.12s ease; }
    .token:hover { border-color: #111827; transform: scale(1.08); z-index: 3; position: relative; box-shadow: 0 2px 8px rgba(17,24,39,0.18); }
    """

    html_out = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>RAG Probe Diagnostic Report</title>
  <style>{css}</style>
</head>
<body>
<header>
  <h1>RAG Probe Diagnostic Report</h1>
  <div class="note">prediction_path={html.escape(str(prediction_path))} | train_log={html.escape(str(train_log_path))}</div>
</header>
<main>
  {tabs}
</main>
<script>
  document.querySelectorAll(".tab-label").forEach((label) => {{
    label.addEventListener("click", () => {{
      window.setTimeout(() => {{
        if (window.Plotly) {{
          document.querySelectorAll(".plotly-graph-div").forEach((node) => {{
            window.Plotly.Plots.resize(node);
          }});
        }}
      }}, 80);
    }});
  }});
</script>
</body>
</html>"""

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(html_out, encoding="utf-8")
    log(f"Report saved to {report_path}")
