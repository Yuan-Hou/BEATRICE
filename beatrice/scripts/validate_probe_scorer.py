"""Scaled numerical validation of the probe scorer HTTP service on TEST-split examples.

Covers the full 2x2 matrix — {context, question} mode x {provided, generated} answer —
N examples per cell, via real HTTP requests to the running service:

  provided : send the training-time recorded answer; online p_correct must reproduce the
             offline eval prediction per-example (|dp| ~ 0), and subset AUC must match.
  generated: omit the answer; greedy decoding should reproduce the training-time answer
             text for most examples (determinism check), p on exact-matches must agree,
             and the online scores' subset AUC vs labels is compared to the offline one.

Usage:
  python scripts/validate_probe_scorer.py --url http://localhost:8008 -n 100
Writes per-cell jsonl + summary.md to outputs/reports/probe_scorer_validation/.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests
from sklearn.metrics import roc_auc_score

OFFLINE = {
    "context": "outputs/predictions/frozen_fusion_context_test_ctxL17rd_attn_p6t3_preds.jsonl",
    "question": "outputs/predictions/frozen_fusion_question_test_L17rd_attn_greedyk3_preds.jsonl",
}


def load_offline(mode: str, n: int) -> list[dict]:
    """Stratified across datasets (round-robin) — a sequential head slice is dominated
    by one dataset and its subset AUC is misleading."""
    by_ds: dict[str, list[dict]] = {}
    for line in open(OFFLINE[mode]):
        r = json.loads(line)
        sm = r.get("source_meta", {})
        ex = sm.get("example", {})
        ans = (sm.get("feature_extraction", {}).get("rag", {}).get("spans", {}) or {}).get("answer_text")
        if not ex.get("question") or not ans:
            continue
        if mode == "context" and not ex.get("contexts"):
            continue
        by_ds.setdefault(str(r.get("dataset") or "?"), []).append({
            "example_id": r["example_id"], "label": int(r["label"]),
            "offline_p": float(r["binary_probability"]),
            "question": ex["question"], "contexts": ex.get("contexts"),
            "train_answer": ans,
        })
    rows, i = [], 0
    while len(rows) < n and any(i < len(v) for v in by_ds.values()):
        for ds in sorted(by_ds):
            if i < len(by_ds[ds]) and len(rows) < n:
                rows.append(by_ds[ds][i])
        i += 1
    return rows


def score_batch(url: str, items: list[dict], chunk: int = 16) -> list[dict]:
    out = []
    for i in range(0, len(items), chunk):
        r = requests.post(f"{url}/batch_score", json={"items": items[i:i + chunk]}, timeout=1800)
        r.raise_for_status()
        out.extend(r.json()["items"])
        print(f"    {min(i + chunk, len(items))}/{len(items)} scored", flush=True)
    return out


def run_cell(url: str, mode: str, path: str, rows: list[dict]) -> tuple[dict, list[dict]]:
    items = []
    for row in rows:
        it = {"example_id": row["example_id"], "question": row["question"], "mode": mode}
        if mode == "context":
            it["contexts"] = row["contexts"]
        if path == "provided":
            it["answer"] = row["train_answer"]
        items.append(it)
    t0 = time.time()
    results = score_batch(url, items)
    elapsed = time.time() - t0

    recs, diffs, y, p_on, ans_match = [], [], [], [], 0
    fails = 0
    for row, res in zip(rows, results):
        rec = {**row, "online": res}
        recs.append(rec)
        if res.get("status") != "success":
            fails += 1
            continue
        y.append(row["label"])
        p_on.append(res["p_correct"])
        if path == "provided":
            diffs.append(abs(res["p_correct"] - row["offline_p"]))
        else:
            if res.get("answer", "").strip() == row["train_answer"].strip():
                ans_match += 1
                diffs.append(abs(res["p_correct"] - row["offline_p"]))
    ok = len(y)
    stats = {
        "mode": mode, "path": path, "n": len(rows), "success": ok, "failed": fails,
        "seconds_per_item": round(elapsed / max(len(rows), 1), 2),
        "auc_online": round(roc_auc_score(y, p_on), 4) if len(set(y)) > 1 else None,
        "auc_offline_subset": round(roc_auc_score(
            [r["label"] for r, res in zip(rows, results) if res.get("status") == "success"],
            [r["offline_p"] for r, res in zip(rows, results) if res.get("status") == "success"],
        ), 4) if len(set(y)) > 1 else None,
    }
    if diffs:
        stats["dp_mean"] = round(sum(diffs) / len(diffs), 5)
        stats["dp_max"] = round(max(diffs), 5)
        stats["dp_n"] = len(diffs)
    if path == "generated":
        stats["answer_exact_match_rate"] = round(ans_match / max(ok, 1), 3)
    return stats, recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8008")
    ap.add_argument("-n", type=int, default=100)
    ap.add_argument("--out", default="outputs/reports/probe_scorer_validation")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    all_stats = []
    for mode in ("context", "question"):
        rows = load_offline(mode, args.n)
        print(f"[{mode}] {len(rows)} test examples loaded", flush=True)
        for path in ("provided", "generated"):
            print(f"[{mode} / {path}] scoring ...", flush=True)
            stats, recs = run_cell(args.url, mode, path, rows)
            all_stats.append(stats)
            with open(out / f"{mode}_{path}.jsonl", "w") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  -> {stats}", flush=True)

    lines = ["# Probe scorer service validation (test split)", ""]
    lines.append("| mode | path | n | ok | s/item | AUC online | AUC offline(subset) | mean\\|dp\\| | max\\|dp\\| | ans match |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for s in all_stats:
        lines.append(
            f"| {s['mode']} | {s['path']} | {s['n']} | {s['success']} | {s['seconds_per_item']} "
            f"| {s.get('auc_online')} | {s.get('auc_offline_subset')} "
            f"| {s.get('dp_mean', '—')} | {s.get('dp_max', '—')} | {s.get('answer_exact_match_rate', '—')} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # hard gate: provided-path must reproduce offline eval
    bad = [s for s in all_stats if s["path"] == "provided" and s.get("dp_max", 1) > 0.02]
    fail_rate = any(s["failed"] > s["n"] * 0.05 for s in all_stats)
    if bad or fail_rate:
        print("VALIDATION FAILED")
        raise SystemExit(1)
    print("VALIDATION PASSED")


if __name__ == "__main__":
    main()
