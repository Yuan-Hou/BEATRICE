"""Unified detection metrics shared by every baseline and the main probes.

A "prediction file" is a JSONL where each row has at least ``label`` (0/1) and a score
field (default ``helpful_probability``, falling back to ``score``). This is exactly the
schema ``scripts/evaluate_probe.py`` / the trainer already write, so the same evaluator
ranks the frozen-fusion probe and the new baselines on identical footing.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ragprobe.utils.io import read_jsonl


SCORE_KEYS = ("helpful_probability", "score", "without_context_correct_probability", "with_context_correct_probability")


def _pick_score(row: dict, score_key: str | None) -> float | None:
    if score_key is not None:
        val = row.get(score_key)
        return float(val) if val is not None else None
    for key in SCORE_KEYS:
        if row.get(key) is not None:
            return float(row[key])
    return None


def binary_detection_metrics(labels, scores, threshold: float = 0.5) -> dict:
    """AUROC / AUPRC plus thresholded accuracy/precision/recall/F1 for one group."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = int(labels.shape[0])
    pos = int(labels.sum())
    out = {
        "n": n,
        "positive_rate": float(pos / n) if n else 0.0,
    }
    if n == 0 or pos == 0 or pos == n:
        out.update({"auroc": None, "auprc": None, "note": "need both classes"})
        return out

    out["auroc"] = float(roc_auc_score(labels, scores))
    out["auprc"] = float(average_precision_score(labels, scores))
    # AUPRC of a random classifier == positive rate; report lift for honesty.
    out["auprc_lift"] = float(out["auprc"] - out["positive_rate"])

    preds = (scores >= threshold).astype(np.int64)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    out.update({
        "threshold": float(threshold),
        "accuracy": float((preds == labels).mean()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    })
    return out


def metrics_overall_and_by_dataset(rows: list[dict], score_key: str | None, threshold: float) -> dict:
    labels, scores, datasets = [], [], []
    for row in rows:
        score = _pick_score(row, score_key)
        label = row.get("label")
        if score is None or label is None:
            continue
        labels.append(int(label))
        scores.append(score)
        datasets.append(str(row.get("dataset", "unknown")))

    result = {"overall": binary_detection_metrics(labels, scores, threshold)}
    by_dataset = defaultdict(lambda: ([], []))
    for label, score, ds in zip(labels, scores, datasets):
        by_dataset[ds][0].append(label)
        by_dataset[ds][1].append(score)
    result["by_dataset"] = {
        ds: binary_detection_metrics(lab, sco, threshold)
        for ds, (lab, sco) in sorted(by_dataset.items())
    }
    return result


def evaluate_prediction_files(paths: list[str], score_key: str | None = None, threshold: float = 0.5) -> dict:
    """Evaluate one or more prediction JSONL files, keyed by file stem."""
    from pathlib import Path

    report = {}
    for path in paths:
        rows = list(read_jsonl(path))
        report[Path(path).name] = metrics_overall_and_by_dataset(rows, score_key, threshold)
    return report


def format_report_table(report: dict) -> str:
    """Compact text table: one row per prediction file, overall AUROC/AUPRC."""
    lines = [f"{'method/file':52s} {'n':>6s} {'pos%':>6s} {'AUROC':>7s} {'AUPRC':>7s} {'PRlift':>7s} {'F1':>6s}"]
    lines.append("-" * 96)
    for name, res in report.items():
        o = res["overall"]
        auroc = f"{o['auroc']:.4f}" if o.get("auroc") is not None else "  n/a"
        auprc = f"{o['auprc']:.4f}" if o.get("auprc") is not None else "  n/a"
        lift = f"{o['auprc_lift']:+.4f}" if o.get("auprc_lift") is not None else "  n/a"
        f1 = f"{o.get('f1', 0.0):.3f}" if o.get("auroc") is not None else " n/a"
        lines.append(
            f"{name[:52]:52s} {o['n']:6d} {o['positive_rate']*100:5.1f}% {auroc:>7s} {auprc:>7s} {lift:>7s} {f1:>6s}"
        )
    return "\n".join(lines)
