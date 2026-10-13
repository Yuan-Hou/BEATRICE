import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml
from sklearn.metrics import accuracy_score, auc, precision_recall_curve, precision_score, recall_score, roc_auc_score

from ragprobe.utils.io import read_jsonl, write_jsonl


def _example_id(record: dict[str, Any]) -> str:
    return str(record.get("example_id") or ((record.get("source_meta") or {}).get("example") or {}).get("id") or "")


def _dataset(record: dict[str, Any]) -> str:
    return str(record.get("dataset") or ((record.get("source_meta") or {}).get("example") or {}).get("dataset") or "UNKNOWN")


def _split(record: dict[str, Any]) -> str:
    return str(record.get("split") or ((record.get("source_meta") or {}).get("example") or {}).get("split") or "unknown")


def _load_split_ids(path: str | None, group: str) -> set[str]:
    if not path:
        return set()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ids = set(str(x) for x in ((data.get("ids") or {}).get(group) or []))
    for ds_data in (data.get("by_dataset") or {}).values():
        if isinstance(ds_data, dict):
            ids.update(str(x) for x in (ds_data.get(f"{group}_ids") or []))
    return ids


def _load(paths: list[str]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        if not path or not Path(path).exists():
            continue
        for r in read_jsonl(path):
            item = dict(r)
            item["prediction_source_path"] = path
            rows.append(item)
    return rows


def _is_rewritten(record: dict[str, Any]) -> bool:
    meta = record.get("source_meta") or {}
    rag = (meta.get("feature_extraction") or {}).get("rag") or {}
    return bool(meta.get("rewrite")) or "rewritten_context" in str(rag.get("tensor_path") or "")


def _score(record: dict[str, Any], preferred: str) -> float:
    if preferred == "with_context_correct_probability":
        return float(record.get("with_context_correct_probability", record.get("binary_probability", record.get("helpful_probability", 0.0))))
    if preferred == "without_context_correct_probability":
        return float(record.get("without_context_correct_probability", record.get("binary_probability", record.get("helpful_probability", 0.0))))
    return float(record.get(preferred, record.get("binary_probability", record.get("helpful_probability", 0.0))))


def _label(record: dict[str, Any], target: str) -> int:
    if target == "without_context_correct":
        if "without_context_correct_label" in record:
            return int(record["without_context_correct_label"])
        trans = record.get("transition") or ""
        return 1 if trans[:1] == "Y" else 0
    if target == "with_context_correct":
        if "with_context_correct_label" in record:
            return int(record["with_context_correct_label"])
        return int(record.get("label", 0))
    return int(record.get("label", 0))


def _partition(rows: list[dict[str, Any]], kind: str, split_ids_path: str | None = None, validation_group: str = "validation", test_group: str = "test") -> list[dict[str, Any]]:
    if kind == "validation":
        ids = _load_split_ids(split_ids_path, validation_group)
        return [r for r in rows if _split(r) == "validation" and _example_id(r) in ids]
    if kind == "test":
        ids = _load_split_ids(split_ids_path, test_group)
        return [r for r in rows if _split(r) == "validation" and _example_id(r) in ids]
    if kind == "ood":
        return [r for r in rows if _split(r) == "test"]
    return rows


def _best_threshold(rows: list[dict[str, Any]], target: str, score_key: str) -> float:
    y_true = [_label(r, target) for r in rows]
    y_score = [_score(r, score_key) for r in rows]
    if len(set(y_true)) < 2:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return 0.5
    best_i, best_f1 = 0, -1.0
    for i, thr in enumerate(thresholds):
        p, r = float(precision[i]), float(recall[i])
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        if f1 > best_f1:
            best_i, best_f1 = i, f1
    return float(thresholds[best_i])


def _metric_rows(rows: list[dict[str, Any]], target: str, score_key: str, threshold: float) -> dict[str, Any]:
    y_true = [_label(r, target) for r in rows]
    y_score = [_score(r, score_key) for r in rows]
    y_pred = [1 if s >= threshold else 0 for s in y_score]
    out = {
        "n": len(rows),
        "threshold": float(threshold),
        "positive_rate": sum(y_pred) / len(y_pred) if y_pred else 0.0,
        "accuracy": float(accuracy_score(y_true, y_pred)) if rows else 0.0,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)) if rows else 0.0,
        "recall": float(recall_score(y_true, y_pred, zero_division=0)) if rows else 0.0,
        "label_positive_rate": sum(y_true) / len(y_true) if y_true else 0.0,
    }
    try:
        out["auc"] = float(roc_auc_score(y_true, y_score)) if len(set(y_true)) >= 2 else None
    except ValueError:
        out["auc"] = None
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        out["pr_auc"] = float(auc(recall, precision)) if len(set(y_true)) >= 2 else None
    except ValueError:
        out["pr_auc"] = None
    return out


def _by_dataset(rows: list[dict[str, Any]], target: str, score_key: str, threshold: float) -> dict[str, Any]:
    groups = defaultdict(list)
    for r in rows:
        groups[_dataset(r)].append(r)
    return {name: _metric_rows(group, target, score_key, threshold) for name, group in sorted(groups.items())}


def _pr_curve(rows: list[dict[str, Any]], target: str, score_key: str) -> dict[str, list[float]]:
    y_true = [_label(r, target) for r in rows]
    y_score = [_score(r, score_key) for r in rows]
    if len(set(y_true)) < 2:
        return {"precision": [], "recall": [], "thresholds": []}
    p, r, t = precision_recall_curve(y_true, y_score)
    return {"precision": [float(x) for x in p], "recall": [float(x) for x in r], "thresholds": [float(x) for x in t]}


def _without_correct(record: dict[str, Any]) -> bool:
    meta = record.get("source_meta") or {}
    runs = meta.get("runs") or {}
    if isinstance(runs.get("without_context"), dict) and runs["without_context"].get("is_correct") is not None:
        return bool(runs["without_context"].get("is_correct"))
    trans = record.get("transition") or ""
    return trans[:1] == "Y"


def _with_correct(record: dict[str, Any]) -> bool:
    meta = record.get("source_meta") or {}
    runs = meta.get("runs") or {}
    if isinstance(runs.get("with_context"), dict) and runs["with_context"].get("is_correct") is not None:
        return bool(runs["with_context"].get("is_correct"))
    trans = record.get("transition") or ""
    return trans[1:2] == "Y"


def _choose_context_records(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped = defaultdict(list)
    for r in rows:
        grouped[_example_id(r)].append(r)
    chosen = {}
    for exid, items in grouped.items():
        rewritten = [r for r in items if _is_rewritten(r)]
        pool = rewritten or items
        chosen[exid] = sorted(pool, key=lambda r: str(r.get("prediction_source_path") or ""))[0]
    return chosen


def _flow_metrics(q_rows: list[dict[str, Any]], c_rows: list[dict[str, Any]], q_thr: float, c_thr: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    c_by_id = _choose_context_records(c_rows)
    traces = []
    for q in q_rows:
        exid = _example_id(q)
        c = c_by_id.get(exid)
        if c is None:
            continue
        q_prob = _score(q, "without_context_correct_probability")
        without_correct = _without_correct(q)
        with_correct = _with_correct(c)
        if q_prob >= q_thr:
            used_rag = False
            final_correct = without_correct
            c_prob = None
            prob_says_wrong = False
        else:
            used_rag = True
            final_correct = with_correct
            c_prob = _score(c, "with_context_correct_probability")
            prob_says_wrong = c_prob < c_thr
        traces.append({
            "example_id": exid,
            "dataset": _dataset(q),
            "split": _split(q),
            "transition": q.get("transition"),
            "question_probability": q_prob,
            "context_probability": c_prob,
            "context_probability_full": _score(c, "with_context_correct_probability"),
            "used_rag": used_rag,
            "context_is_rewritten": _is_rewritten(c),
            "without_context_correct": without_correct,
            "with_context_correct": with_correct,
            "final_correct": final_correct,
            "prob_says_wrong_after_rag": prob_says_wrong,
        })

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        wrong = [x for x in items if not x["final_correct"]]
        prob_wrong = [x for x in items if x["prob_says_wrong_after_rag"]]
        return {
            "n": len(items),
            "rag_usage_rate": sum(1 for x in items if x["used_rag"]) / len(items) if items else 0.0,
            "final_accuracy": sum(1 for x in items if x["final_correct"]) / len(items) if items else 0.0,
            "prob_wrong_precision": sum(1 for x in prob_wrong if not x["final_correct"]) / len(prob_wrong) if prob_wrong else 0.0,
            "prob_wrong_recall": sum(1 for x in wrong if x["prob_says_wrong_after_rag"]) / len(wrong) if wrong else 0.0,
            "rewritten_context_rate_when_rag": sum(1 for x in items if x["used_rag"] and x["context_is_rewritten"]) / max(sum(1 for x in items if x["used_rag"]), 1),
            "transition_counts": dict(Counter(x.get("transition") or "UNKNOWN" for x in items)),
        }

    groups = defaultdict(list)
    for t in traces:
        groups[t["dataset"]].append(t)
    metrics = {"overall": summarize(traces), "by_dataset": {k: summarize(v) for k, v in sorted(groups.items())}}
    return traces, metrics


def _write_pr_plot(curves: dict[str, dict[str, Any]], path: str) -> None:
    try:
        import matplotlib.pyplot as plt
        out = Path(path); out.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(7, 5))
        for name, curve in curves.items():
            if curve.get("recall"):
                plt.plot(curve["recall"], curve["precision"], label=name)
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curves")
        plt.legend(loc="best", fontsize=8); plt.tight_layout(); plt.savefig(out, dpi=160); plt.close()
    except Exception as exc:
        print(f"[baseline-flow] could not write PR plot {path}: {exc}")


def _html_table(title: str, data: dict[str, Any]) -> str:
    rows = []
    for name, m in data.items():
        rows.append("<tr>" + "".join([
            f"<td>{name}</td>",
            f"<td>{m.get('n', 0)}</td>",
            f"<td>{m.get('auc', '')}</td>",
            f"<td>{m.get('pr_auc', '')}</td>",
            f"<td>{m.get('accuracy', '')}</td>",
            f"<td>{m.get('precision', '')}</td>",
            f"<td>{m.get('recall', '')}</td>",
            f"<td>{m.get('positive_rate', '')}</td>",
        ]) + "</tr>")
    return f"<h2>{title}</h2><table><tr><th>group</th><th>n</th><th>AUC</th><th>PR AUC</th><th>Acc</th><th>Precision</th><th>Recall</th><th>Pred+Rate</th></tr>{''.join(rows)}</table>"


def _flow_table(title: str, data: dict[str, Any]) -> str:
    rows = []
    for name, m in data.items():
        rows.append("<tr>" + "".join([
            f"<td>{name}</td>", f"<td>{m.get('n',0)}</td>", f"<td>{m.get('rag_usage_rate',0):.4f}</td>",
            f"<td>{m.get('final_accuracy',0):.4f}</td>", f"<td>{m.get('prob_wrong_precision',0):.4f}</td>",
            f"<td>{m.get('prob_wrong_recall',0):.4f}</td>", f"<td>{m.get('rewritten_context_rate_when_rag',0):.4f}</td>",
        ]) + "</tr>")
    return f"<h2>{title}</h2><table><tr><th>group</th><th>n</th><th>RAG usage</th><th>Final acc</th><th>P(wrong|prob wrong)</th><th>P(prob wrong|wrong)</th><th>Rewritten rate when RAG</th></tr>{''.join(rows)}</table>"




def _reference_baselines_from_traces(traces: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(items),
            "always_rag_accuracy": sum(1 for x in items if x.get("with_context_correct")) / len(items) if items else 0.0,
            "never_rag_accuracy": sum(1 for x in items if x.get("without_context_correct")) / len(items) if items else 0.0,
        }
    groups = defaultdict(list)
    for item in traces:
        groups[item.get("dataset") or "UNKNOWN"].append(item)
    return {"overall": summarize(traces), "by_dataset": {k: summarize(v) for k, v in sorted(groups.items())}}


def _reference_table(title: str, data: dict[str, Any]) -> str:
    rows = []
    merged = {"overall": data.get("overall", {}), **(data.get("by_dataset") or {})}
    for name, m in merged.items():
        rows.append(
            "<tr>"
            f"<td>{name}</td>"
            f"<td>{m.get('n', 0)}</td>"
            f"<td>{m.get('always_rag_accuracy', 0):.4f}</td>"
            f"<td>{m.get('never_rag_accuracy', 0):.4f}</td>"
            "</tr>"
        )
    return f"<h2>{title}</h2><table><tr><th>group</th><th>n</th><th>Always RAG Acc</th><th>Never RAG Acc</th></tr>{''.join(rows)}</table>"


def _interactive_report_html(q_thr: float, c_thr: float, split_ids_path: str | None, test_traces: list[dict[str, Any]], ood_traces: list[dict[str, Any]]) -> str:
    payload = json.dumps({"test": test_traces, "ood": ood_traces}, ensure_ascii=False)
    return f"""
<h2>Interactive Threshold Explorer</h2>
<p>Adjust thresholds in steps of 0.01. The tables below recompute the two-stage policy and baselines locally in the browser.</p>
<div class='control-panel'>
  <label>Question-only threshold <span id='qThrText'></span><input id='qThr' type='range' min='0' max='1' step='0.01' value='{q_thr:.2f}'></label>
  <label>With-context threshold <span id='cThrText'></span><input id='cThr' type='range' min='0' max='1' step='0.01' value='{c_thr:.2f}'></label>
</div>
<div class='grid2'>
  <div><h3>In-domain Test</h3><div id='interactiveTest'></div></div>
  <div><h3>OOD Test</h3><div id='interactiveOod'></div></div>
</div>
<script id='flow-data' type='application/json'>{payload}</script>
<script>
function fmt(x) {{ return (Number.isFinite(x) ? x.toFixed(4) : '0.0000'); }}
function summarize(rows, qThr, cThr) {{
  const out = {{n: rows.length, rag: 0, final: 0, probWrong: 0, probWrongAndWrong: 0, wrong: 0, probWrongGivenWrong: 0, always: 0, never: 0}};
  for (const r of rows) {{
    const qProb = Number(r.question_probability || 0);
    const cProb = Number(r.context_probability_full || r.context_probability || 0);
    const noCorrect = !!r.without_context_correct;
    const ragCorrect = !!r.with_context_correct;
    out.always += ragCorrect ? 1 : 0;
    out.never += noCorrect ? 1 : 0;
    let finalCorrect, usedRag=false, probWrong=false;
    if (qProb >= qThr) {{
      finalCorrect = noCorrect;
    }} else {{
      usedRag = true;
      finalCorrect = ragCorrect;
      probWrong = cProb < cThr;
    }}
    out.rag += usedRag ? 1 : 0;
    out.final += finalCorrect ? 1 : 0;
    out.wrong += finalCorrect ? 0 : 1;
    out.probWrong += probWrong ? 1 : 0;
    out.probWrongAndWrong += (probWrong && !finalCorrect) ? 1 : 0;
  }}
  return {{
    n: out.n,
    rag_usage_rate: out.n ? out.rag / out.n : 0,
    final_accuracy: out.n ? out.final / out.n : 0,
    prob_wrong_precision: out.probWrong ? out.probWrongAndWrong / out.probWrong : 0,
    prob_wrong_recall: out.wrong ? out.probWrongAndWrong / out.wrong : 0,
    always_rag_accuracy: out.n ? out.always / out.n : 0,
    never_rag_accuracy: out.n ? out.never / out.n : 0,
  }};
}}
function byDataset(rows, qThr, cThr) {{
  const groups = {{overall: rows}};
  for (const r of rows) {{
    const k = r.dataset || 'UNKNOWN';
    if (!groups[k]) groups[k] = [];
    groups[k].push(r);
  }}
  const result = {{}};
  for (const [k, v] of Object.entries(groups)) result[k] = summarize(v, qThr, cThr);
  return result;
}}
function table(metrics) {{
  let html = '<table><tr><th>group</th><th>n</th><th>RAG usage</th><th>Final acc</th><th>P(wrong|prob wrong)</th><th>P(prob wrong|wrong)</th><th>Always RAG acc</th><th>Never RAG acc</th></tr>';
  for (const [k, m] of Object.entries(metrics)) {{
    html += `<tr><td>${{k}}</td><td>${{m.n}}</td><td>${{fmt(m.rag_usage_rate)}}</td><td>${{fmt(m.final_accuracy)}}</td><td>${{fmt(m.prob_wrong_precision)}}</td><td>${{fmt(m.prob_wrong_recall)}}</td><td>${{fmt(m.always_rag_accuracy)}}</td><td>${{fmt(m.never_rag_accuracy)}}</td></tr>`;
  }}
  html += '</table>';
  return html;
}}
function updateExplorer() {{
  const data = JSON.parse(document.getElementById('flow-data').textContent);
  const qThr = Number(document.getElementById('qThr').value);
  const cThr = Number(document.getElementById('cThr').value);
  document.getElementById('qThrText').textContent = qThr.toFixed(2);
  document.getElementById('cThrText').textContent = cThr.toFixed(2);
  document.getElementById('interactiveTest').innerHTML = table(byDataset(data.test || [], qThr, cThr));
  document.getElementById('interactiveOod').innerHTML = table(byDataset(data.ood || [], qThr, cThr));
}}
document.getElementById('qThr').addEventListener('input', updateExplorer);
document.getElementById('cThr').addEventListener('input', updateExplorer);
updateExplorer();
</script>
"""

def run(config: dict[str, Any]) -> dict[str, Any]:
    q_val = _load(config["input"]["question_validation_prediction_paths"])
    q_test_explicit = _load(config["input"].get("question_test_prediction_paths", []))
    q_ood = _load(config["input"].get("question_ood_prediction_paths", []))
    c_val = _load(config["input"]["context_validation_prediction_paths"])
    c_test_explicit = _load(config["input"].get("context_test_prediction_paths", []))
    c_ood = _load(config["input"].get("context_ood_prediction_paths", []))

    split_ids_path = config["input"].get("sample_split_path")
    validation_group = config["input"].get("validation_group", "validation")
    test_group = config["input"].get("test_group", "test")
    q_validation = _partition(q_val, "validation", split_ids_path, validation_group, test_group)
    q_test = q_test_explicit or _partition(q_val, "test", split_ids_path, validation_group, test_group)
    c_validation = _partition(c_val, "validation", split_ids_path, validation_group, test_group)
    c_test = c_test_explicit or _partition(c_val, "test", split_ids_path, validation_group, test_group)
    q_thr = _best_threshold(q_validation, "without_context_correct", "without_context_correct_probability")
    c_thr = _best_threshold(c_validation, "with_context_correct", "with_context_correct_probability")

    def model_block(name, cal, test, ood, target, score_key, threshold):
        all_test = test
        all_ood = ood
        curves = {"test_overall": _pr_curve(all_test, target, score_key), "ood_overall": _pr_curve(all_ood, target, score_key)}
        for ds, group in defaultdict(list, {}).items():
            pass
        return {
            "threshold_from_validation": threshold,
            "validation_overall": _metric_rows(cal, target, score_key, threshold),
            "test_overall": _metric_rows(all_test, target, score_key, threshold),
            "test_by_dataset": _by_dataset(all_test, target, score_key, threshold),
            "ood_overall": _metric_rows(all_ood, target, score_key, threshold),
            "ood_by_dataset": _by_dataset(all_ood, target, score_key, threshold),
            "pr_curves": curves,
        }

    q_block = model_block("question", q_validation, q_test, q_ood, "without_context_correct", "without_context_correct_probability", q_thr)
    c_block = model_block("context", c_validation, c_test, c_ood, "with_context_correct", "with_context_correct_probability", c_thr)

    flow_test_traces, flow_test = _flow_metrics(q_test, c_test, q_thr, c_thr)
    flow_ood_traces, flow_ood = _flow_metrics(q_ood, c_ood, q_thr, c_thr)
    reference_test = _reference_baselines_from_traces(flow_test_traces)
    reference_ood = _reference_baselines_from_traces(flow_ood_traces)

    metrics = {
        "thresholds": {"question_without_context_correct": q_thr, "context_with_context_correct": c_thr},
        "question_model": q_block,
        "context_model": c_block,
        "decision_flow_test": flow_test,
        "decision_flow_ood": flow_ood,
        "reference_baselines_test": reference_test,
        "reference_baselines_ood": reference_ood,
        "sample_split_path": split_ids_path,
        "counts": {
            "question_validation": len(q_validation), "question_test": len(q_test), "question_ood": len(q_ood),
            "context_validation": len(c_validation), "context_test": len(c_test), "context_ood": len(c_ood),
        },
    }

    out_cfg = config["output"]
    Path(out_cfg["metrics_path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(out_cfg["metrics_path"]).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    if out_cfg.get("test_trace_path"):
        write_jsonl(flow_test_traces + flow_ood_traces, out_cfg["test_trace_path"])
    if out_cfg.get("question_pr_plot_path"):
        _write_pr_plot({"question_test": q_block["pr_curves"]["test_overall"], "question_ood": q_block["pr_curves"]["ood_overall"]}, out_cfg["question_pr_plot_path"])
    if out_cfg.get("context_pr_plot_path"):
        _write_pr_plot({"context_test": c_block["pr_curves"]["test_overall"], "context_ood": c_block["pr_curves"]["ood_overall"]}, out_cfg["context_pr_plot_path"])
    if out_cfg.get("report_path"):
        html = f"""<!doctype html><html><head><meta charset='utf-8'><title>Baseline Probe Flow Report</title>
<style>body{{font-family:Arial,sans-serif;margin:24px;color:#111}}table{{border-collapse:collapse;margin:16px 0;width:100%}}td,th{{border:1px solid #ddd;padding:6px;font-size:13px}}th{{background:#f3f4f6}}code{{background:#f3f4f6;padding:2px 4px}}.cards{{display:flex;gap:12px;flex-wrap:wrap}}.card{{border:1px solid #ddd;padding:12px;min-width:220px}}.control-panel{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px;margin:18px 0;padding:14px;border:1px solid #ddd;background:#fafafa}}.control-panel label{{display:flex;flex-direction:column;gap:8px;font-weight:600}}.grid2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:20px}}</style></head><body>
<h1>Baseline Probe Flow Report</h1>
<div class='cards'><div class='card'><b>Question threshold</b><br>{q_thr:.6f}</div><div class='card'><b>Context threshold</b><br>{c_thr:.6f}</div><div class='card'><b>Validation split</b><br><code>{split_ids_path}</code><br>train/validation/test ids are explicitly recorded</div></div>
{_html_table('Question-only model: validation-test', {'overall': q_block['test_overall'], **q_block['test_by_dataset']})}
{_html_table('Question-only model: OOD', {'overall': q_block['ood_overall'], **q_block['ood_by_dataset']})}
{_html_table('With-context model: validation-test', {'overall': c_block['test_overall'], **c_block['test_by_dataset']})}
{_html_table('With-context model: OOD', {'overall': c_block['ood_overall'], **c_block['ood_by_dataset']})}
{_reference_table('Reference baselines: in-domain test', reference_test)}
{_reference_table('Reference baselines: OOD', reference_ood)}
{_flow_table('Decision flow: validation-test', {'overall': flow_test['overall'], **flow_test['by_dataset']})}
{_flow_table('Decision flow: OOD', {'overall': flow_ood['overall'], **flow_ood['by_dataset']})}
{_interactive_report_html(q_thr, c_thr, split_ids_path, flow_test_traces, flow_ood_traces)}
<p>Decision flow: if question-only predicts correct, answer without RAG. Otherwise retrieve and answer with context, then use the with-context probe threshold to flag likely wrong answers.</p>
</body></html>"""
        Path(out_cfg["report_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(out_cfg["report_path"]).write_text(html, encoding="utf-8")
    print(json.dumps({"metrics_path": out_cfg["metrics_path"], "report_path": out_cfg.get("report_path"), "thresholds": metrics["thresholds"], "counts": metrics["counts"]}, ensure_ascii=False, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    run(config)


if __name__ == "__main__":
    main()
